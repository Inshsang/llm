#!/bin/bash
#numgpu=1

exp=$1
dataname=$2
visfeat_type=local
now=$(date +"%Y%m%d_%H%M%S")

python_bin=/data/HTC/Library/lamm/bin/python
deepspeed_bin=/data/HTC/Library/lamm/bin/deepspeed
cfg_path=/data/HTC/Project/llm/src/config/train_ds3.yaml
data_root=/data/HTC/Data/dataset/Benchmark/data/LAMM/3D_Instruct
data_path=${data_root}/meta_file/LAMM_3dinstruct_10k.json
encoder_ckpt_path=/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth
vicuna_ckpt_path=/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0
ckpt_dir=/data/HTC/Data/model_zoo/llm_exe

mkdir -p ${ckpt_dir}/${exp}/log_rest/

if [ ! -x "${python_bin}" ]; then
    echo "Python not found: ${python_bin}" >&2
    exit 1
fi

if [ ! -x "${deepspeed_bin}" ]; then
    echo "DeepSpeed not found: ${deepspeed_bin}" >&2
    exit 1
fi

if [ ! -f "${data_path}" ]; then
    echo "Training meta file not found: ${data_path}" >&2
    exit 1
fi

if [ ! -f "${encoder_ckpt_path}" ]; then
    echo "EPCL checkpoint not found: ${encoder_ckpt_path}" >&2
    exit 1
fi

if [ ! -d "${vicuna_ckpt_path}" ]; then
    echo "Vicuna checkpoint directory not found: ${vicuna_ckpt_path}" >&2
    exit 1
fi

if [ ! -d "${data_root}/3rscan_pcls" ] || [ ! -d "${data_root}/shapenet_pcls" ]; then
    echo "OpenLAMM 3D data is still zipped. Please unzip 3rscan_pcls.zip and shapenet_pcls.zip under ${data_root} before training." >&2
    exit 1
fi

cd /data/HTC/Project/llm/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${deepspeed_bin}" --include localhost:1 --master_addr 127.0.0.1 --master_port 28451 train.py \
    --train_stage 3 \
    --cfg ${cfg_path} \
    --data_path ${data_path} \
    --vision_root_path ${data_root} \
    --vision_type pcl \
    --use_system \
    --model lamm_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path ${encoder_ckpt_path} \
    --vicuna_ckpt_path ${vicuna_ckpt_path} \
    --vision_feature_type ${visfeat_type} \
    --max_tgt_len 384 \
    --num_vision_token 256 \
    --save_path ${ckpt_dir}/${exp} \
    --log_path ${ckpt_dir}/${exp}/log_rest/ \
    2>&1 | tee ${ckpt_dir}/${exp}/log_rest/train_${now}.log
