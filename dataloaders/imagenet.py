
import os
from typing import List, Tuple
import torch
import os
from torch.utils.data import DataLoader, Dataset
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError(
        "Please install DALI from https://www.github.com/NVIDIA/DALI to run this example."
    )


device_memory_padding = 211025920
host_memory_padding = 140544512
root = ""
train_rec = os.path.join(root, "imagenet_train.rec")
train_idx = os.path.join(root, "imagenet_train.idx")
val_rec = os.path.join(root, "imagenet_val.rec")
val_idx = os.path.join(root, "imagenet_val.idx")

num_example_train_val = 1281167
num_classes = 1000

@torch.no_grad()
class DALIWarper(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict["data"]
        tensor_label: torch.Tensor = data_dict["label"][:, 0].long()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter.__len__()

    def reset(self):
        self.iter.reset()


def create_dali_pipeline(
    batch_size: int,
    rec_file: str,
    idx_file: str,
    crop_size: int,
    val_size: int,
    workers: int,
    name: str,
    is_training=True,
    mean: tuple = (0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255),
    std: tuple = (0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255)
):
    """
    Parameters
    ----------
    rec_files str
        Path to the RecordIO file.
    idx_file: str
        Path to the index file.
    rank: int
        Index of the shard to read.
    local_rank: int
        Id of GPU used by the pipeline.
    world_size: int
        Partitions the data into the specified number of parts (shards).
        This is typically used for multi-GPU or multi-node training.

    Returns
    -------
    pipe: pipeline
        The pipeline encapsulates the data processing graph and the execution engine.
    """
    val_size = int(val_size)
    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    device_id = int(os.getenv("LOCAL_RANK"))

    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=workers,
        device_id=device_id,
        prefetch_queue_depth=2,
        seed=rank + 18,
    )

    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file,
            index_path=idx_file,
            initial_fill=1024 << 4,
            shard_id=rank,
            num_shards=world_size,
            random_shuffle=is_training,
            pad_last_batch=False,
            name=name,
        )

        if is_training:
            images = fn.decoders.image_random_crop(
                jpegs,
                device="mixed",
                output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                random_aspect_ratio=[0.8, 1.25],
                random_area=[0.1, 1.0],
                num_attempts=100,
            )
            images = fn.resize(
                images,
                device="gpu",
                resize_x=crop_size,
                resize_y=crop_size,
                interp_type=types.INTERP_TRIANGULAR,
            )
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(
                images,
                device="gpu",
                size=int(256 / 224 * val_size),
                mode="not_smaller",
                interp_type=types.INTERP_TRIANGULAR,
            )
            mirror = False
        
        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(crop_size, crop_size),
            mean=mean,
            std=std,
            mirror=mirror,
        )
        labels = labels.gpu()
    pipe.set_outputs(images, labels)
    pipe.build()
    return pipe


def get_loader_trainval(batch_size, crop_size, val_size, workers,
) -> Tuple[Dataset, DataLoader]:
    pipe = create_dali_pipeline(
        batch_size=batch_size,
        rec_file=train_rec,
        idx_file=train_idx,
        crop_size=crop_size,
        val_size=val_size,
        workers=workers,
        name="loader_train",
        is_training=True,)
    return DALIWarper(DALIClassificationIterator(pipe, reader_name="loader_train", last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded=False))


def get_loader_test(batch_size, crop_size, val_size, workers,
) -> Tuple[Dataset, DataLoader]:
    pipe = create_dali_pipeline(
        batch_size=batch_size,
        rec_file=val_rec,
        idx_file=val_idx,
        crop_size=crop_size,
        val_size=val_size,
        workers=workers,
        name="loader_test",
        is_training=False,)
    return DALIWarper(DALIClassificationIterator(pipe, reader_name="loader_test", last_batch_policy = LastBatchPolicy.PARTIAL, last_batch_padded=False))