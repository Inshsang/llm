import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
	sys.path.append(str(SRC_ROOT))

from model.openlamm import LAMMPEFTModel
from datasets.system_msg import common_task2sysmsg  # type: ignore
from conversations import conv_templates  # type: ignore


# Use the same system message family as training/eval to reduce prompt mismatch.
SYS_MSG = common_task2sysmsg.get("Classification", "A chat between a curious human and an artificial intelligence assistant.")


# Align with src/inference_3d.py: classification uses a fixed query.
CLASSIFICATION_QUERY = "What's the 3D point cloud about?"


def _make_model_args(cli_args):
	vision_output_layer = -2 if cli_args.vision_feature_type == "local" else -1
	num_vision_token = 256 if cli_args.vision_feature_type == "local" else 1

	return {
		"model": cli_args.model,
		"vision_type": cli_args.vision_type,
		"encoder_pretrain": cli_args.encoder_pretrain,
		"encoder_ckpt_path": cli_args.encoder_ckpt_path,
		"vicuna_ckpt_path": cli_args.vicuna_ckpt_path,
		"train_stage": cli_args.train_stage,
		"lora_r": cli_args.lora_r,
		"lora_alpha": cli_args.lora_alpha,
		"lora_dropout": cli_args.lora_dropout,
		"lora_target_modules": cli_args.lora_target_modules,
		"vision_feature_type": cli_args.vision_feature_type,
		"vision_output_layer": vision_output_layer,
		"num_vision_token": num_vision_token,
		"max_tgt_len": cli_args.max_tgt_len,
		"use_system": False,
		"max_obj_len": cli_args.max_obj_len,
		"use_flash_attn": cli_args.use_flash_attn,
		"use_xformers": cli_args.use_xformers,
		"bs": 1,
		"local_rank": cli_args.gpu_id,
	}


def load_lamm(cli_args):
	available = torch.cuda.device_count()
	if cli_args.gpu_id >= available:
		raise ValueError(f"Requested gpu_id={cli_args.gpu_id} but only {available} CUDA device(s) visible")
	torch.cuda.set_device(cli_args.gpu_id)
	model_args = _make_model_args(cli_args)
	model = LAMMPEFTModel(**model_args)

	if cli_args.delta_ckpt_path:
		if not os.path.exists(cli_args.delta_ckpt_path):
			raise FileNotFoundError(f"delta checkpoint not found: {cli_args.delta_ckpt_path}")
		delta_ckpt = torch.load(cli_args.delta_ckpt_path, map_location=torch.device("cpu"))
		model.load_state_dict(delta_ckpt, strict=False)
		model.llama_model = model.llama_model.merge_and_unload()

	model = model.eval().half().cuda()
	return model


def _default_pcl_metadata(pcl_path: str):
	"""Load point cloud and build minimal metadata.

	Important: `model.openlamm.LAMMPEFTModel.prepare_generation_embedding()` expects
	`inputs['list_of_objpoints']` to be an (N,3) numpy array of points for
	Classification/DescriptionObj/ConversationObj.

	For `obj_list`, inference_3d.py uses a single Unknown bounding box placeholder.
	"""
	points = np.load(pcl_path)[:, :3]
	# Keep obj_list minimal; classification branch in openlamm ignores its class_list.
	obj_list = [{"name": "Unknown", "BoundingBox": [0, 0, 0, 2, 2, 2]}]
	return points.astype(np.float32), obj_list


def _build_prompt_with_history(cli_args, history, user_text: str) -> str:
	"""Build prompt text using the same conversation template as inference scripts.

	See src/inference_3d.py: generate_conversation_text() uses conv_templates[conv_mode]
	and formats as:
	  conv.system + (sep + role: msg + "\n")* + (sep + Human: input + "\n")
	Note: inference_3d DOES NOT append an explicit "Assistant:" stub line.
	"""
	conv_mode = getattr(cli_args, "conv_mode", "simple")
	conv = conv_templates[conv_mode]
	sep = conv.sep
	roles = conv.roles

	prompts = ""
	prompts += conv.system
	for q, a in history:
		prompts += "{} {}: {}\n{} {}: {}\n".format(sep, roles[0], q, sep, roles[1], a)
	prompts += "{} {}: {}\n".format(sep, roles[0], user_text)
	return prompts


def _extract_assistant_reply(reply: str) -> str:
	"""Robustly extract the latest Assistant span from a model output.

	Rules:
	- Prefer the text after the last "### Assistant:" (training template友好)。
	- Stop before the next role separator (### Human/Assistant) or generic ###.
	- Fallback: keep the substring before the first ###, else the whole string.
	"""
	if not isinstance(reply, str):
		return ""

	text = reply.strip()
	if not text:
		return ""

	# Align with inference_3d postprocess: take content before the first '###'.
	# This is important because the model often emits '\n###' as an end marker.
	if "###" in text:
		text = text.split("###", 1)[0].strip()

	# If the model echoed the system prompt, strip it.
	# (We take the last occurrence to be safe.)
	if SYS_MSG and SYS_MSG in text:
		text = text.split(SYS_MSG)[-1].strip()
	# Also strip the long classification system prompt if it leaked.
	leak_prefix = "You are a multimodal language model"
	if leak_prefix in text:
		text = text.split(leak_prefix)[-1].strip()

	# If the model output contains role markers, take the last assistant span.
	anchor = "### Assistant:"
	if anchor in text:
		text = text.rsplit(anchor, 1)[-1].strip()
		# Trim any following role markers if present
		for tok in ("### Human:", "Human:", "### Assistant:"):
			idx = text.find(tok)
			if idx != -1:
				text = text[:idx].strip()

	return text


def _looks_like_empty_or_question_stub(text: str) -> bool:
	if not text:
		return True
	low = text.strip().lower()
	# Common degenerate starts seen in debug logs
	return low in {"is", "are", "do", "does", "what", "why", "how", "natural", "language", "natural language"}


def _normalize_classification_label(text: str) -> str:
	"""Normalize model output to a single class label.

	We keep the first token-like span and strip punctuation.
	"""
	if not text:
		return ""
	clean = text.strip()
	clean = clean.splitlines()[0].strip()
	# Parse formats like "(A) apple" / "(Option) name"
	if clean.startswith("(") and ")" in clean:
		clean = clean.split(")", 1)[1].strip()
	# If the model answered in a sentence, try to pick a plausible label token.
	# Heuristic: prefer the last word on the first line (often the class name),
	# but fall back to the first word.
	first_line = clean
	tokens = [t.strip("\t\r\n \"'`.,;:!?()[]{}<>") for t in first_line.split() if t.strip()]
	if not tokens:
		return ""
	# If it contains a known filler phrase, treat as invalid.
	joined_low = " ".join(tokens).lower()
	if joined_low in {"natural language", "natural"}:
		return ""
	clean = tokens[-1] if len(tokens) > 1 else tokens[0]
	clean = clean.strip("\t\r\n \"'`.,;:!?()[]{}<>")
	return clean


def build_inputs(prompt: str, cli_args, history, image_path: Optional[str] = None, pcl_path: Optional[str] = None):
	task_type = cli_args.task_type
	list_of_objpoints = np.zeros((1, 3), dtype=np.float32)
	obj_list = [{"name": "Unknown", "BoundingBox": [0, 0, 0, 2, 2, 2]}]

	if pcl_path:
		list_of_objpoints, obj_list = _default_pcl_metadata(pcl_path)

	# For classification tasks: align with inference_3d.py
	# - disable history
	# - always use fixed query
	# - use conv template prompt builder
	if task_type.lower() == "classification":
		prompt_text = _build_prompt_with_history(cli_args, history=[], user_text=CLASSIFICATION_QUERY)
	else:
		# Other tasks: keep history.
		prompt_text = _build_prompt_with_history(cli_args, history, prompt)

	return {
		"prompt": [prompt_text],
		"image_paths": [image_path] if image_path else [],
		"pcl_paths": [pcl_path] if pcl_path else [],
		"top_p": cli_args.top_p,
		"temperature": cli_args.temperature,
		"max_tgt_len": cli_args.max_tgt_len,
		"modality_embeds": [],
		"obj_list": obj_list,
		"list_of_objpoints": list_of_objpoints,
		"task_type": task_type,
	}


def chat_once(model, prompt: str, cli_args, history, image_path: Optional[str] = None, pcl_path: Optional[str] = None):
	inputs = build_inputs(prompt, cli_args, history, image_path=image_path, pcl_path=pcl_path)
	outputs = model.generate(inputs)
	return outputs[0]


def parse_cli():
	parser = argparse.ArgumentParser(description="快速 LAMM 推理脚本")
	parser.add_argument("--prompt", default=None, help="可选：启动后先发送一次的用户输入（默认不发送，直接进入交互）")
	parser.add_argument("--image", dest="image_path", default=None, help="用于视觉条件的可选图像路径")
	parser.add_argument("--pcl", dest="pcl_path", default='/data/HTC/Data/dataset/object_1024_npy/0_apple60.npy', help="用于视觉条件的可选点云 .npy 路径（默认 apple）")
	parser.add_argument("--model", default="lamm_peft")
	parser.add_argument("--vision-type", dest="vision_type", choices=("image", "pcl"), default="pcl")
	parser.add_argument("--encoder-pretrain", dest="encoder_pretrain", choices=("clip", "epcl"), default="epcl")
	parser.add_argument("--encoder-ckpt-path", dest="encoder_ckpt_path", default="/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth")
	parser.add_argument("--vicuna-ckpt-path", dest="vicuna_ckpt_path", default="/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0")
	parser.add_argument("--delta-ckpt-path", dest="delta_ckpt_path", default="/data/HTC/Data/model_zoo/llm_exe/base_vicuna/pytorch_model_ep1.pt")
	parser.add_argument("--gpu-id", dest="gpu_id", type=int, default=1, help="使用的 CUDA 设备索引（从 0 开始，默认 2 即第 3 张卡）")
	parser.add_argument("--train-stage", dest="train_stage", type=int, default=2)
	parser.add_argument("--lora-r", dest="lora_r", type=int, default=32)
	parser.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=32)
	parser.add_argument("--lora-dropout", dest="lora_dropout", type=float, default=0.1)
	parser.add_argument("--lora-target-modules", dest="lora_target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
	parser.add_argument("--vision-feature-type", dest="vision_feature_type", choices=("local", "global"), default="local")
	parser.add_argument("--max-tgt-len", dest="max_tgt_len", type=int, default=1024)
	parser.add_argument("--top-p", dest="top_p", type=float, default=0.9)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--max-obj-len", dest="max_obj_len", type=int, default=30)
	parser.add_argument("--use-flash-attn", dest="use_flash_attn", action="store_true")
	parser.add_argument("--use-xformers", dest="use_xformers", action="store_true")
	parser.add_argument("--task-type", dest="task_type", default="Classification", help="任务类型：Classification/normal 等（传入模型 inputs['task_type']）")
	parser.add_argument("--conv-mode", dest="conv_mode", default="simple", help="对话模板名称（对齐 src/inference_3d.py，默认 simple）")
	parser.add_argument("--debug-raw", dest="debug_raw", action="store_true", help="打印模型原始输出，便于排查截断")
	parser.add_argument("--no-history", dest="no_history", action="store_true", help="不使用多轮历史（对分类任务建议开启）")
	parser.add_argument("--once", dest="once", action="store_true", help="单次推理：跑一轮后退出（便于快速验证分类输出）")
	args = parser.parse_args()

	if args.vision_feature_type == "global" and args.vision_type == "pcl":
		raise ValueError("Global vision features are not implemented for point clouds")
	if args.pcl_path and not os.path.exists(args.pcl_path):
		raise FileNotFoundError(f"Point cloud not found: {args.pcl_path}")
	if args.image_path and not os.path.exists(args.image_path):
		raise FileNotFoundError(f"Image not found: {args.image_path}")
	return args


def main():
	args = parse_cli()
	model = load_lamm(args)
	history = []
	current_pcl_path = args.pcl_path
	current_image_path = args.image_path

	print("Usage: type your question each round; type 'pcl=/path/to.npy' to switch point cloud; type 'exit' to quit.")
	if args.task_type.lower() == "classification":
		print(f"[INFO] Classification mode: your input is ignored; using fixed query: {CLASSIFICATION_QUERY}")

	def _run_one_turn(user_text: str):
		# In classification mode, follow inference_3d: always use the fixed question.
		effective_text = CLASSIFICATION_QUERY if args.task_type.lower() == "classification" else user_text
		local_history = [] if (args.no_history or args.task_type.lower() == "classification") else history
		# Align decode params with src/inference_3d.py::Class_response
		old_top_p = args.top_p
		old_temp = args.temperature
		old_max_len = args.max_tgt_len
		if args.task_type.lower() == "classification":
			args.top_p = 0.6
			args.temperature = 0.7
			args.max_tgt_len = 1800
		reply = chat_once(model, effective_text, args, local_history, image_path=current_image_path, pcl_path=current_pcl_path)
		if args.debug_raw:
			print(f"[DEBUG raw] {repr(reply)}")
		clean_reply = _extract_assistant_reply(reply)
		if args.task_type.lower() == "classification":
			clean_reply = _normalize_classification_label(clean_reply)
			# If generation collapses, auto retry once with lower temperature.
			if _looks_like_empty_or_question_stub(clean_reply):
				args.temperature = 0.2
				args.top_p = 0.1
				args.max_tgt_len = 512
				reply2 = chat_once(model, effective_text, args, [], image_path=current_image_path, pcl_path=current_pcl_path)
				if args.debug_raw:
					print(f"[DEBUG raw retry] {repr(reply2)}")
				# Restore after retry
				args.temperature = old_temp
				args.top_p = old_top_p
				args.max_tgt_len = old_max_len
				clean_reply2 = _normalize_classification_label(_extract_assistant_reply(reply2))
				if clean_reply2:
					clean_reply = clean_reply2
		# Restore decode params for non-classification or next turns.
		args.top_p = old_top_p
		args.temperature = old_temp
		args.max_tgt_len = old_max_len
		if not clean_reply:
			# Lightweight guardrail when generation collapses to a separator-only output.
			if args.vision_type == "pcl" and not current_pcl_path:
				clean_reply = "A point cloud is required (use --pcl to provide a .npy file path)."
			elif args.vision_type == "image" and not current_image_path:
				clean_reply = "An image is required (use --image to provide an image path)."
			else:
				clean_reply = "No valid answer was generated (the model may have emitted separators early). Try --debug-raw."
		print(f"助手: {clean_reply}")
		if not (args.no_history or args.task_type.lower() == "classification"):
			history.append((user_text, clean_reply))
		return clean_reply
	
	# Optional: send one initial prompt
	if args.prompt is not None and len(str(args.prompt).strip()) > 0:
		_run_one_turn(str(args.prompt).strip())
		if args.once:
			return

	if args.once:
		# If --once and no explicit prompt, still run one classification turn.
		_run_one_turn("")
		return

	while True:
		# Always ask user for each round's input.
		try:
			user_text = input("你: ").strip()
		except EOFError:
			break

		if user_text.lower() in {"quit", "exit", "q"}:
			break
		if len(user_text) == 0:
			continue

		# Allow switching point cloud in-chat: pcl=/abs/path.npy
		if user_text.startswith("pcl="):
			candidate = user_text[len("pcl=") :].strip().strip('"').strip("'")
			if candidate and os.path.exists(candidate):
				current_pcl_path = candidate
				print(f"[INFO] Switched point cloud: {current_pcl_path}")
			else:
				print(f"[WARN] Invalid point cloud path: {candidate}")
			continue

		_run_one_turn(user_text)


if __name__ == "__main__":
	main()
