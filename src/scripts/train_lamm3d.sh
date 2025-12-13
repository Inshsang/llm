#!/bin/bash
#numgpu=1

exp=$1
dataname=$2
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=/data/HTC/Data/model_zoo/llm_exe
mkdir -p ${ckpt_dir}/${exp}/log_rest/
deepspeed --include localhost:1,2 --master_addr 127.0.0.1 --master_port 28451 train.py \
    --train_stage 2 \
    --cfg /data/HTC/Project/llm/src/config/train_ds3.yaml \
    --data_path  /data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/WholeTrain/Classification3d_demo.json \
    --vision_root_path /data/HTC/Data/dataset \
    --vision_type pcl \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path /data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth \
    --vicuna_ckpt_path /data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0 \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
