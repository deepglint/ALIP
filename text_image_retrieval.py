import torch
import argparse
from src.open_alip import create_model, image_transform
from dataloaders import coco, flickr30k


dataset_dict = {'coco': coco,
                'flickr': flickr30k}

def compute_retrieval(similarity_scores, txt2img, img2txt):
    # compute text -> image
    t2i_similarity_score = similarity_scores.t()
    t2i_ranks = torch.zeros(t2i_similarity_score.shape[0])

    for index, score in enumerate(t2i_similarity_score):
        inds = torch.argsort(score, descending= True)
        t2i_ranks[index] = torch.where(inds == txt2img[index])[0][0]
        print('Evaluating batch {}/{}, {}'.format(index, t2i_similarity_score.shape[0], t2i_ranks[index]), end = "\r")

    # Compute metrics
    tr1 = 100.0 * len(torch.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
    tr5 = 100.0 * len(torch.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
    tr10 = 100.0 * len(torch.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
    t2i_report_dict = {"r1": tr1, "r5": tr5, "r10": tr10}

    #compute image -> text
    i2t_similarity_score = similarity_scores
    i2t_ranks = torch.zeros(i2t_similarity_score.shape[0])
    for index, score in enumerate(i2t_similarity_score):
        print('Evaluating batch {}/{}'.format(index, i2t_similarity_score.shape[0]), end = "\r")
        inds = torch.argsort(score, descending= True)
        # Score
        rank = 1e10
        for i in img2txt[index]:
            tmp = torch.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        i2t_ranks[index] = rank

    # Compute metrics
    ir1 = 100.0 * len(torch.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
    ir5 = 100.0 * len(torch.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
    ir10 = 100.0 * len(torch.where(i2t_ranks < 10)[0]) / len(i2t_ranks)
    i2t_report_dict = {"r1": ir1, "r5": ir5, "r10": ir10}
    return t2i_report_dict, i2t_report_dict


def get_image_feature(model, data_loader):
    image_features = []
    for batch_idx, batch in enumerate(data_loader):
        print('Evaluating batch {}/{}'.format(batch_idx, len(data_loader)), end = "\r")
        images, _ = batch
        image_emb = model.encode_image(images.cuda()) #embed with image encoder
        image_features.append(image_emb.detach().cpu())
    image_features = torch.cat(image_features, 0)
    
    print('Done image feature extract.')
    print(image_features.shape)

    # normalized features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


def get_text_feature(model, data_loader):
    text_features = []
    for batch_idx, batch in enumerate(data_loader):
        print('Evaluating batch {}/{}'.format(batch_idx, len(data_loader)), end = "\r")
        text = batch.squeeze()
        text_emb = model.encode_text(text.cuda())
        text_features.append(text_emb.detach().cpu())

    text_features = torch.cat(text_features, 0)
    print('Done text feature extract.')
    print(text_features.shape)

    # normalized features
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_transform(image_size):
    image_mean = (0.48145466, 0.4578275, 0.40821073)
    image_std = (0.26862954, 0.26130258, 0.27577711)
    preprocess = image_transform(image_size, is_train=False, mean=image_mean, std=image_std)
    return preprocess


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


def main(args):
    model_alip = create_model(args.model_name)
    state_dict = get_state_dict(args.model_weight)
    model_alip.load_state_dict(state_dict, strict=True)
    model_alip.eval()
    model_alip.cuda()

    transform = get_transform(args.input_size)
    dataset_module = dataset_dict[args.dataset]
    assert hasattr(dataset_module, "get_loader_image")
    assert hasattr(dataset_module, "get_loader_text")

    kwargs = {
        "batch_size": args.batch_size,
        "preprocess": transform}
    
    text_loader = dataset_module.get_loader_text(**kwargs)
    text_features = get_text_feature(model_alip, text_loader)

    image_loader, txt2img, img2txt = dataset_module.get_loader_image(**kwargs)
    image_features = get_image_feature(model_alip, image_loader)

    similarity_scores = image_features.cuda() @ text_features.cuda().t()
    similarity_scores = similarity_scores
    t2i_dict, i2t_dict = compute_retrieval(similarity_scores, txt2img, img2txt)
    print('Text retrieval', i2t_dict)
    print('Image retrieval', t2i_dict)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="ZeroShot")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--dataset", default="coco", type=str, help='coco or flickr')
    parser.add_argument("--model-name", default="ViT-B/32", help="Name of the vision backbone to use.")
    parser.add_argument("--model-weight", default= "/mnt/laion/clip/vit_b_16-laion400m_e32-55e67d44.pt")
    parser.add_argument("--input-size", default=224, type=int, help="Image resolution.")
    args = parser.parse_args()
    main(args)
