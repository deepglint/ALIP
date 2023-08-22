ip_list=(YOUR_ADDRESS) # e.g. one node ("1.1.1.1"); multi node ("1,1,1,1" "2,2,2,2")
model_name=ViT-B/32
train_num_samples=15061515
user_name=YOUR_NAME
output=YOUR_PATH
train_data=YOUR_DATA_PATH

for((node_rank=0;node_rank<${#ip_list[*]};node_rank++));
do
  ssh $user_name@${ip_list[node_rank]} "cd `pwd`;PATH=$PATH \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  NCCL_ALGO=Ring \
  NCCL_SOCKET_IFNAME=eth0 \
  NCCL_SOCKET_NTHREADS=8 \
  NCCL_NSOCKS_PERTHREAD=2 \
  torchrun --nproc_per_node 8 \
  --nnodes=${#ip_list[*]} \
  --node_rank=$node_rank \
  --master_addr=${ip_list[0]} \
  --master_port=23457  train.py \
    --batch-size 512 \
    --epochs 32 \
    --lr 1e-3 \
    --model-name $model_name \
    --optimizer adamw \
    --output $output \
    --train-data $train_data \
    --train-num-samples $train_num_samples \
    --weight-decay 0.2 " &
done
