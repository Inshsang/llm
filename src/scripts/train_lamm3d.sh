#!/bin/bash
#numgpu=1

exp=$1
dataname=$2
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

ckpt_dir=../ckpt
mkdir -p ${ckpt_dir}/${exp}/log_rest/
deepspeed --include localhost:4 --master_addr 127.0.0.1 --master_port 28451 train.py \
    --stage 1 \
    --cfg /media/kou/Data1/htc/LAMM/src/config/train_ds3.yaml \
    --data_path  /media/kou/Data1/htc/MYDATA/BenchMark/Task/Task_Reconstruct/Train/Classification.json \
    --vision_root_path /media/kou/Data1/htc/LAMM/data/3D_Instruct/ \
    --vision_type pcl \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path /media/kou/Data3/htc/epcl_scannet_vit-L-14_256tokens_latest.pth \
    --vicuna_ckpt_path /media/kou/Data3/htc/vicuna-7b/ \
    --vision_feature_type ${visfeat_type} \
    --num_vision_token 256 \
    --save_path  ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
