import argparse
import logging
import math
import os
import sys
import time
import torch
from dali import dali_dataloader
from torch import distributed, optim
from torch.utils.tensorboard import SummaryWriter
from src.open_alip import create_model, Adaptive_loss
from torch import distributed as dist


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--beta1", type=float, default=0.9, help="adamw")
parser.add_argument("--beta2", type=float, default=0.98, help="adamw")
parser.add_argument("--epochs", type=int, default=32)
parser.add_argument("--gradient-acc", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate.")
parser.add_argument("--lr-scheduler", default="cosine")
parser.add_argument("--input-size", default=224, type=int)
parser.add_argument("--model-name", type=str, default="ViT-B/32")
parser.add_argument("--local-loss",default=False,help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)")
parser.add_argument("--gather-with-grad",default=False,help="enable full distributed gradient for feature gather")
parser.add_argument("--horovod",default=False,action="store_true",help="Use horovod for distributed training.")
parser.add_argument("--optimizer", default="sgd")
parser.add_argument("--output", required=True)
parser.add_argument("--train-data", required=True)
parser.add_argument("--train-num-samples", type=int, required=True)
parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def main():
    os.makedirs(args.output, exist_ok=True)
    init_logging(rank, args.output)
    if rank == 0:
        summary_writer = SummaryWriter(os.path.join(args.output, "tensorboard"))
    else:
        summary_writer = None

    model_alip = create_model(args.model_name)
    model_alip.cuda()
    model_alip = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_alip)
    model_alip = torch.nn.parallel.DistributedDataParallel(
        module=model_alip,
        bucket_cap_mb=32,
        find_unused_parameters=True,
        static_graph=True)
    
    train_loader = dali_dataloader(args)

    global_step = GlobalStep()
    steps_per_epoch = args.train_num_samples // world_size // args.batch_size + 1
    steps_total = int(args.epochs * steps_per_epoch)

    contrastive_loss = Adaptive_loss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        use_horovod=args.horovod)

    opt = torch.optim.AdamW(
        params=[{"params": model_alip.parameters()}],
        lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))

    if args.lr_scheduler == "cosine":
        assert isinstance(args.epochs, int)
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=opt,
            max_lr=[args.lr],
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=0.1,
        )
    elif args.lr_scheduler == "linear":
        lr_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=opt, start_factor=1.0, end_factor=0.0,
            total_iters=int(args.epochs * steps_per_epoch))
    else:
        raise

    callback_func = SpeedCallBack(5, steps_total, args.batch_size)
    auto_scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=200)
    start_epoch = 0
        
    model_alip.train()
    momentum_sim_raw, momentum_sim_caption, momentum_sim = 1, 1, 1
    for epoch in range(start_epoch, math.ceil(args.epochs)):
        for _, (img, text_token) in enumerate(train_loader):
            text_token = text_token.long().cuda()
            img = img.cuda()
            with torch.cuda.amp.autocast(True):
                image_embeddings, text_embeddings_laion, text_embeddings_caption, logit_scale= model_alip(img, text_token)

            with torch.no_grad():
                gathered_raw_features = [torch.zeros_like(text_embeddings_laion) for _ in range(world_size)]
                gathered_caption_features = [torch.zeros_like(text_embeddings_caption) for _ in range(world_size)]
                gathered_raw_image_features = [torch.zeros_like(image_embeddings) for _ in range(world_size)]

                dist.all_gather(gathered_raw_features, text_embeddings_laion)
                dist.all_gather(gathered_caption_features, text_embeddings_caption)
                dist.all_gather(gathered_raw_image_features, image_embeddings)
                
                gathered_raw_features[rank] = text_embeddings_laion
                gathered_caption_features[rank] = text_embeddings_caption
                gathered_raw_image_features[rank] = image_embeddings

                gather_embeddings_laion = torch.cat(gathered_raw_features, dim=0)
                gather_embeddings_caption = torch.cat(gathered_caption_features, dim=0)
                gather_embeddings_image = torch.cat(gathered_raw_image_features, dim=0)  

                cur_sim = torch.sum(gather_embeddings_laion * gather_embeddings_caption, dim=1)
                cur_mean = torch.mean(cur_sim)
                momentum_sim = cur_mean * 0.01 + momentum_sim * 0.99
                weight_sample_raw = torch.exp(cur_sim - momentum_sim) ** 2
                ones_tensor = torch.ones_like(weight_sample_raw)
                weight_sample = torch.where(weight_sample_raw < 1, weight_sample_raw, ones_tensor)
                weight_sample_mean = torch.mean(weight_sample)

                cur_sim_raw = torch.sum(gather_embeddings_laion * gather_embeddings_image, dim=1)
                cur_mean_raw = torch.mean(cur_sim_raw)
                momentum_sim_raw = cur_mean_raw * 0.01 + momentum_sim_raw * 0.99
                weight_raw = torch.exp(cur_sim_raw - momentum_sim_raw) ** 2
                weight_raw = torch.where(weight_sample_raw < 1, weight_raw, ones_tensor)
                weight_raw_mean = torch.mean(weight_raw)

                cur_sim_caption = torch.sum(gather_embeddings_caption * gather_embeddings_image, dim=1)
                cur_mean_caption = torch.mean(cur_sim_caption)
                momentum_sim_caption = cur_mean_caption * 0.01 + momentum_sim_caption * 0.99
                weight_caption = torch.exp(cur_sim_caption - momentum_sim_caption) ** 2
                weight_caption = torch.where(weight_sample_raw < 1, weight_caption, ones_tensor)
                weight_caption_mean = torch.mean(weight_caption)
        
            
            loss_laion = contrastive_loss(image_embeddings.float(), text_embeddings_laion.float(), logit_scale, weight_raw, weight_sample)
            loss_caption = contrastive_loss(image_embeddings.float(), text_embeddings_caption.float(), logit_scale, weight_caption, weight_sample)
            loss = loss_laion + loss_caption
            auto_scaler.scale(loss).backward()

            if global_step.step % args.gradient_acc == 0:
                auto_scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model_alip.parameters(), 1)
                auto_scaler.step(opt)
                auto_scaler.update()
                opt.zero_grad()
            
            # Note: we clamp to 4.6052 = ln(100), as in the original paper.
            with torch.no_grad():
                unwrap_model(model_alip).logit_scale.clamp_(0, math.log(100))

            lr_scheduler.step()
            global_step.step += 1

            with torch.no_grad():
                callback_func(lr_scheduler, float(loss), float(loss_laion), float(loss_caption), float(cur_mean_raw), float(momentum_sim_raw), float(weight_raw_mean), float(cur_mean_caption), float(momentum_sim_caption), float(weight_caption_mean), float(cur_mean), float(momentum_sim), float(weight_sample_mean), global_step.step, auto_scaler.get_scale())
                if summary_writer is not None:
                    summary_writer.add_scalar(tag="loss", scalar_value=loss.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="loss_laion", scalar_value=loss_laion.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="loss_caption", scalar_value=loss_caption.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="current_raw_sim", scalar_value=cur_mean_raw.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="momentum_raw_sim", scalar_value=momentum_sim_raw.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="weight_raw_mean", scalar_value=weight_raw_mean.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="current_caption_sim", scalar_value=cur_mean_caption.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="momentum_caption_sim", scalar_value=momentum_sim_caption.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="weight_caption_mean", scalar_value=weight_caption_mean.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="current_text_sim", scalar_value=cur_mean.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="momentum_text_sim", scalar_value=momentum_sim.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="weight_text_mean", scalar_value=weight_sample_mean.item(), global_step=global_step.step)
                    summary_writer.add_scalar(tag="lr_backbone",
                                              scalar_value=lr_scheduler.get_last_lr()[0],
                                              global_step=global_step.step)
                    summary_writer.add_scalar(tag="logit_scale",
                                              scalar_value=logit_scale.item(),
                                              global_step=global_step.step)

            if global_step.step > steps_total:
                break

        train_loader.reset()
        if rank == 0: 
            torch.save(obj=model_alip.state_dict(), f=os.path.join(args.output, "model_{}.pt".format(str(epoch))))

    if summary_writer is not None:
        summary_writer.close()


def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)
        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")
        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_stream = logging.StreamHandler(sys.stdout)
        handler_file.setFormatter(formatter)
        handler_stream.setFormatter(formatter)
        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)
        log_root.info('rank_id: %d' % rank)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GlobalStep:
    def __init__(self, step: int = 0):
        self.step = int(step)

    def update(self):
        self.step += 1


class SpeedCallBack(object):
    def __init__(self, frequent, steps_total, batch_size):
        self.batch_size = batch_size
        self.frequent = frequent
        self.steps_total = steps_total
        self.loss_metric = AverageMeter()
        self.loss_laion_metric = AverageMeter()
        self.loss_caption_metric = AverageMeter()
        self.current_raw_sim = AverageMeter()
        self.momentum_raw_sim = AverageMeter()
        self.weight_raw_mean = AverageMeter()
        self.current_caption_sim = AverageMeter()
        self.momentum_caption_sim = AverageMeter()
        self.weight_caption_mean = AverageMeter()
        self.current_text_sim = AverageMeter()
        self.momentum_text_sim = AverageMeter()
        self.weight_text_mean = AverageMeter()
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.time_start = time.time()
        self.init = False
        self.tic = 0

    def __call__(
            self,
            lr_scheduler: optim.lr_scheduler._LRScheduler,
            loss,
            loss_laion,
            loss_caption,
            cur_mean_raw,
            momentum_sim_raw,
            weight_raw_mean,
            cur_sim_caption,
            momentum_sim_caption,
            weight_caption_mean,
            cur_mean,
            momentum_sim,
            weight_sample_mean,
            global_step,
            scale):#clip_loss,
        assert isinstance(loss, float)

        self.loss_metric.update(loss)
        self.loss_laion_metric.update(loss_laion)
        self.loss_caption_metric.update(loss_caption)

        self.current_raw_sim.update(cur_mean_raw)
        self.momentum_raw_sim.update(momentum_sim_raw)
        self.weight_raw_mean.update(weight_raw_mean)

        self.current_caption_sim.update(cur_sim_caption)
        self.momentum_caption_sim.update(momentum_sim_caption)
        self.weight_caption_mean.update(weight_caption_mean)

        self.current_text_sim.update(cur_mean)
        self.momentum_text_sim.update(momentum_sim)
        self.weight_text_mean.update(weight_sample_mean)

        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = (self.frequent * self.batch_size / (time.time() - self.tic))
                    self.tic = time.time()
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                loss_metric_format = f"[{self.loss_metric.avg :.3f}]"
                self.loss_metric.reset()
                loss_laion_metric_format = f"[{self.loss_laion_metric.avg :.3f}]"
                self.loss_laion_metric.reset()
                loss_caption_metric_format = f"[{self.loss_caption_metric.avg :.3f}]"
                self.loss_caption_metric.reset()

                current_raw_sim_format = f"[{self.current_raw_sim.avg :.3f}]"
                self.current_raw_sim.reset()
                momentum_raw_sim_format = f"[{self.momentum_raw_sim.avg :.3f}]"
                self.momentum_raw_sim.reset()
                weight_raw_mean_format = f"[{self.weight_raw_mean.avg :.3f}]"
                self.weight_raw_mean.reset()

                current_caption_sim_format = f"[{self.current_caption_sim.avg :.3f}]"
                self.current_caption_sim.reset()
                momentum_caption_sim_format = f"[{self.momentum_caption_sim.avg :.3f}]"
                self.momentum_caption_sim.reset()
                weight_caption_mean_format = f"[{self.weight_caption_mean.avg :.3f}]"
                self.weight_caption_mean.reset()

                current_text_sim_format = f"[{self.current_text_sim.avg :.3f}]"
                self.current_text_sim.reset()
                momentum_text_sim_format = f"[{self.momentum_text_sim.avg :.3f}]"
                self.momentum_text_sim.reset()
                weight_text_mean_format = f"[{self.weight_text_mean.avg :.3f}]"
                self.weight_text_mean.reset()


                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.steps_total)
                time_for_end = time_total - time_now
                lr_1 = lr_scheduler.get_last_lr()[0]
                msg = f"rank:{int(speed) :d} "
                msg += f"total:{int(speed_total) :d} "
                msg += f"lr:[{lr_1 :.8f}] "
                msg += f"step:{global_step :d} "
                msg += f"amp:{int(scale) :d} "
                msg += f"required:{time_for_end :.1f} hours "
                msg += f"loss:{loss_metric_format} "
                msg += f"laion_loss:{loss_laion_metric_format} "
                msg += f"caption_loss:{loss_caption_metric_format} "

                msg += f"current_raw_sim:{current_raw_sim_format} "
                msg += f"momentum_raw_sim:{momentum_raw_sim_format} "
                msg += f"weight_raw_mean:{weight_raw_mean_format} "

                msg += f"current_caption_sim:{current_caption_sim_format} "
                msg += f"momentum_caption_sim:{momentum_caption_sim_format} "
                msg += f"weight_caption_mean:{weight_caption_mean_format} "

                msg += f"current_text_sim:{current_text_sim_format} "
                msg += f"momentum_text_sim:{momentum_text_sim_format} "
                msg += f"weight_text_mean:{weight_text_mean_format} "

                if self.rank == 0:
                    logging.info(msg)
            else:
                self.init = True
                self.tic = time.time()
                

if __name__ == "__main__":
    main()
