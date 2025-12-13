dataset=VOC2012
exp=classfication_demo
base_data_path=../data/2D_Benchmark
token_num=256
layer=-2
answerdir=../answers
mkdir -p ${answerdir}/${exp}

python inference_3d.py \
    --model lamm_peft \
    --encoder_pretrain epcl \
    --encoder_ckpt_path /data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth \
    --vicuna_ckpt_path //data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0 \
    --delta_ckpt_path /data/HTC/Data/model_zoo/llm_exe/${exp}/pytorch_model.pt \
    --max_tgt_len 800 \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --num_vision_token ${token_num} \
    --vision_output_layer ${layer} \
    --conv_mode simple \
    --base-data-path ${base_data_path} \
    --inference-mode common \
    --bs 1 \
    --answers-dir ${answerdir}/${exp}