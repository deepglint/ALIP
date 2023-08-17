CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node 8 \
--master_port=13457 zero_shot.py \
    --model-name  ViT-B/32 \
    --batch-size  128 \
    --output-file output/zero_shot.txt \
    --dataset  cifar10,cifar100 \
    --model-weight weights/ALIP_YFCC15M_B32.pt