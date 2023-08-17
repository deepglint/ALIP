import argparse
import os
import torch
from torch import distributed
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
import json
import os
from torch.utils.data import Subset, DataLoader
import numpy as np
import random
from src.open_alip import create_model, tokenize, factory
from dataloaders import cifar10, cifar100, dtd, food101, stanford_car, fgvc_aircraft, flowers102, oxford_pets, caltech101, sun397, imagenet


rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)

module_dict = {
    "food101": food101,  
    "cifar10": cifar10,
    "cifar100": cifar100,
    "sun397": sun397,
    "stanford_car": stanford_car,
    "aircraft": fgvc_aircraft,
    "dtd": dtd,
    "pets": oxford_pets,
    "flowers": flowers102,
    "caltech101": caltech101,
    "imagenet": imagenet
}

def metric_mean_per_class_accuracy(output, target, num_classes, topk=(1,)):
    with torch.no_grad():
        class_correct = list(0. for i in range(num_classes))
        class_total = list(0. for i in range(num_classes))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:1].reshape(-1).float()

        for i in range(batch_size):
            label = target.__getitem__(i)
            class_correct[label] += correct_k[i]
            class_total[label] += 1
        
        accuracy_total = 0
        for i in range(num_classes):
            accuracy_class_i =  class_correct[i] / class_total[i]
            accuracy_total += accuracy_class_i

        acc = accuracy_total / num_classes
        acc = np.array(acc)
        return acc


def zero_shot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] 
            texts = tokenize(texts).cuda() # tokenize
            class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


@torch.no_grad()
def metric_map(logits, gts, num_classes):
    mAP = []
    for i in range(num_classes):
        ap = average_precision_score(gts.astype(np.int), logits)
        mAP.append(ap)
    score = np.mean(mAP)
    score = np.array(score)
    return score


def metric_avg_acc1_acc5(output, target):
    with torch.no_grad():
        batch_size = target.size(0)
        #acc1
        _, pred_1 = output.topk(1, 1, True, True)
        pred_1 = pred_1.t()
        correct = pred_1.eq(target.view(1, -1).expand_as(pred_1))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_1 = correct_1.mul_(1.0 / batch_size)
        
        #acc5
        _, pred_5 = output.topk(5, 1, True, True)
        correct = pred_5.eq(target.view(batch_size, -1).expand_as(pred_5))
        correct_5 = correct.reshape(-1).float().sum(0, keepdim=True)
        correct_5 = correct_5.mul_(1.0 / batch_size)
        acc_1_5 = (correct_1 + correct_5) / 2
        acc_1_5 = np.array(acc_1_5)
        return acc_1_5[0]


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc1 =  float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    return acc1 / target.size(0)


def run(model, classifier, dataset, dataset_module, dataset_name, num_dataset, socre_list, args):
    if dataset_name == 'imagenet':
        dataloader = dataset

    else:
        n_data = len(dataset)
        idx_all_rank = list(range(n_data))
        num_local = n_data // world_size + int(rank < n_data % world_size)
        start = n_data // world_size * rank + min(rank, n_data % world_size)
        idx_this_rank = idx_all_rank[start: start + num_local]
        dataset_this_rank = Subset(dataset, idx_this_rank)
        dataloader = DataLoader(
            dataset_this_rank, args.batch_size,
            False, num_workers=4, drop_last=False)

    with torch.no_grad():
        if dataset_name =='imagenet':
            lenth = len(dataloader) * args.batch_size
        else:
            lenth = len(dataset_this_rank)
        logits_tensor = torch.zeros([lenth, dataset_module.num_classes], dtype=torch.long).to(local_rank) 
        
        if dataset_name == 'voc2007':
            target_tensor = torch.zeros([lenth, dataset_module.num_classes], dtype=torch.long).to(local_rank) 
        else:
            target_tensor = torch.zeros(lenth, dtype=torch.long).to(local_rank)

        idx = 0
        for images, target in dataloader:
            images = images.cuda()
            target = target.cuda()            
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100. * image_features @ classifier
            logits_tensor[idx: idx + logits.size(0)] = logits
            target_tensor[idx: idx + target.size(0)] = target
            idx += target.size(0)
        
        logits_tensor = logits_tensor.cpu()
        target_tensor = target_tensor.cpu()
        gather_list_logits = [None for i in range(world_size)]
        gather_list_target = [None for i in range(world_size)]
        distributed.all_gather_object(gather_list_logits, logits_tensor)
        distributed.all_gather_object(gather_list_target, target_tensor)

        if rank == 0:
            gather_logits = torch.cat(gather_list_logits, dim=0)
            gather_target = torch.cat(gather_list_target, dim=0)
            print('{} test dataset have {} data'.format(dataset_name, gather_logits.size(0)))

            if hasattr(dataset_module, "mean_per_class") and dataset_module.mean_per_class:
                acc1 = metric_mean_per_class_accuracy(gather_logits, gather_target, topk=(1,), num_classes=dataset_module.num_classes)
            elif hasattr(dataset_module, "bce") and dataset_module.bce:
                acc1 = metric_map(gather_logits.cpu().numpy(), gather_target.cpu().numpy(), dataset_module.num_classes)
            elif hasattr(dataset_module, "avg_acc1_acc5") and dataset_module.avg_acc1_acc5:
                acc1 = metric_avg_acc1_acc5(gather_logits, gather_target)
            elif hasattr(dataset_module, "roc_auc_score") and dataset_module.roc_auc_score:
                gather_logits = gather_logits.float()
                gather_logits = torch.softmax(gather_logits, dim = 1)
                gather_logits = gather_logits.cpu().detach().numpy()
                acc1 = roc_auc_score(gather_target.cpu().detach().numpy(), gather_logits[:,1])
            else:
                acc1 = accuracy(gather_logits, gather_target, topk=(1, ))  
                
            socre_list.append(str(100 * acc1))
            if len(socre_list) == num_dataset:
                str_data = ','.join(socre_list)
                with open(args.output_file, 'w') as f:
                    f.write(str_data + '\n')


def get_state_dict(model_weight):
    state_dict = torch.load(model_weight)
    state_dict_removed = {}
    for k, value in state_dict.items():
        if "module." in k:
            k_removed = k.split("module.")[-1]
            state_dict_removed[k_removed] = value
        else:
            state_dict_removed[k] = value
    return state_dict_removed
            

def main(args, dataset_list):
    setup_seed(1024, True)
    model_alip = create_model(args.model_name)

    state_dict = get_state_dict(args.model_weight)
    model_alip.load_state_dict(state_dict, strict=True)
    model_alip.eval()
    model_alip.cuda()
        
    dataset_templates = json.load(open('templates.json'))
    dataset_labels = json.load(open('label.json'))

    num_dataset = len(dataset_list)
    socre_list = []
    for num in range(num_dataset):
        dataset_name = dataset_list[num]
        dataset_module = module_dict[dataset_name]
        dataset_classnames = dataset_labels[dataset_name]
        dataset_template = dataset_templates[dataset_name]
        classifier = zero_shot_classifier(model_alip, dataset_classnames, dataset_template)
        classifier.cuda()

        transform = get_transform(args)

        # imagenet dataset saved as rec file
        if  dataset_name == 'imagenet':
            kwargs = {
            "batch_size": args.batch_size,
            "crop_size": args.input_size,
            "val_size": args.input_size,
            "workers": 8
            }
            test_dataset  = dataset_module.get_loader_test(**kwargs)

        else:
            kwargs = {
            "transform": transform,
            "batch_size": args.batch_size,
            "num_workers": 2,
            "seed": 3072}
            test_dataset  = dataset_module.get_loader_test(**kwargs)[0]
        run(model_alip, classifier, test_dataset, dataset_module, dataset_name, num_dataset, socre_list, args)


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_transform(args):
    transform = factory.image_transform(args.input_size, False)
    return transform


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ZeroShot")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--model-name", default="ViT-B/32", help="Name of the vision backbone to use.")
    parser.add_argument("--model-weight", default="", type=str, help="pretrain model weight path.")
    parser.add_argument("--input-size", default=224, type=int, help="Image resolution.")
    parser.add_argument("--output-file", default="", type=str, help="output results file")
    args = parser.parse_args()
    dataset_list = args.dataset.split(',')
    main(args, dataset_list)


            