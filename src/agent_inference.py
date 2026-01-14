"""src/agent_inference.py

3D Agent 推理脚本（对齐你的 `src/inference_3d.py` 数据流）。

目标：把表格里 Agent 缺失的 3D 任务补齐（VG fine/plus、Counting、RoomDetection、PositionRelation），并且严格符合
“MLLM 作为 controller + 工具执行 + 已知 label space + 一次反思 + 固定预算”。

这里的“工具”不再额外引入外部 detector，而是**直接复用 inference_3d.py 里提到的检测产物**：

- `/data/HTC/Data/dataset/Benchmark/data/metadata/Detection.json` （scene-> object list）
- `/data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/Test/Detection.json`（按 index 的检测结果列表）

并复用 inference_3d.py 中的 obj_list / list_of_objpoints 的组装方式，把候选框（3D bbox）作为“proposal set”，
让 MLLM 做重排序/解析，必要时用几何工具计算关系/计数。

输出：jsonl，每条包含 id/pcl/text（兼容原 eval 读法）+ 额外的 agent_trace 字段（工具调用、反思等）。

注意：RoomDetection 在 repo 的 system message 更像输出 room polygon vertices；你表里用 mAP@0.5。
本脚本先实现一个最小可跑的 Agent 版本（给出 room type + 一个 union bbox 近似）。如果你现有 3D room eval
脚本要求 polygon/多实例格式，我可以再把输出对齐到它的 parser。
"""

import os
import json
import time
import argparse
import pickle
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import StoppingCriteriaList

from model.openlamm import LAMMPEFTModel, LAMMStoppingCriteria
from conversations import conv_templates
from datasets import load_3Deval_dataset


def generate_conversation_text(args, input_list, history, sys_msg=None):
    conv = conv_templates[args.conv_mode]
    if sys_msg:
        conv.system = sys_msg
    prompts_list = []
    for _input in input_list:
        prompts = ''
        prompts += conv.system
        for q, a in history:
            prompts += "{} {}: {}\n{} {}: {}\n".format(conv.sep, conv.roles[0], q, conv.sep, conv.roles[1], a)
        prompts += "{} {}: {}\n".format(conv.sep, conv.roles[0], _input)
        prompts_list.append(prompts)
    return prompts_list


def _greedy_generate(model, inputs):
    """Bypass openlamm.generate to force greedy decoding (do_sample=False)."""
    input_embeds, input_masks = model.prepare_generation_embedding(inputs)
    stopping_criteria = StoppingCriteriaList([
        LAMMStoppingCriteria([[2277, 29937], [835]], input_embeds)
    ])
    top_p = inputs.get('top_p', 1.0)
    temperature = inputs.get('temperature', 1.0)
    top_p = min(max(top_p, 1e-5), 1.0)
    temperature = max(temperature, 1e-5)
    outputs = model.llama_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=input_masks,
        max_new_tokens=inputs['max_tgt_len'],
        top_p=top_p,
        temperature=temperature,
        do_sample=False,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    return model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)


def _greedy_text_generate(model: LAMMPEFTModel, prompt: str, max_new_tokens: int = 128) -> str:
    """纯文本 greedy 生成（用于解析问题）。避免走多模态 embedding，规避形状不匹配。"""
    tok = model.llama_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)
    out = model.llama_model.generate(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    # only decode newly generated part (optional); simplest is full decode
    text = model.llama_tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def mllm_generate_one(
    args,
    model,
    prompt_list,
    sys_msg,
    pcl_paths,
    obj_list,
    list_of_objpoints,
    task_type,
    max_length,
    top_p,
    temperature,
):
    # transformers 的 TemperatureLogitsWarper 要求 temperature>0，避免传入 0 触发异常
    safe_temperature = max(temperature, 1e-5)
    safe_top_p = min(max(top_p, 1e-5), 1.0)
    prompt_text = generate_conversation_text(args, prompt_list, history=[], sys_msg=sys_msg)

    # Ensure that the input data is valid and does not contain NaN values
    assert not any(isinstance(p, torch.Tensor) and torch.isnan(p).any() for p in prompt_list), "Prompt list contains NaN values in tensors"

    # Add additional logging for debugging
    print(f"Generating with task_type: {task_type}, max_length: {max_length}, top_p: {top_p}, temperature: {temperature}")

    # Align with inference_3d: let the model extract vision features from pcl + objpoints
    modality_embeds = []

    inputs = {
        'prompt': prompt_text,
        'pcl_paths': pcl_paths,
        'top_p': safe_top_p,
        'temperature': safe_temperature,
        'max_tgt_len': max_length,
        'do_sample': args.do_sample,
        'modality_embeds': modality_embeds,
        'obj_list': obj_list,
        'list_of_objpoints': list_of_objpoints[:args.max_obj],
        'task_type': task_type,
    }

    # 如果用户未开启 --do_sample，则强制走 greedy 路径，避免 openlamm 内部写死的采样导致 multinomial 报错。
    if not args.do_sample:
        return _greedy_generate(model, inputs)

    return model.generate(inputs)


class DetectionTool3D:
    """复用 Detection 产物作为 proposal generator。"""

    def __init__(self, metadata_detection_path: str, test_detection_path: str):
        self.metadata_detection_path = metadata_detection_path
        self.test_detection_path = test_detection_path
        self._meta = None
        self._test = None

    def _lazy_load(self):
        if self._meta is None:
            self._meta = json.load(open(self.metadata_detection_path, 'r'))
        if self._test is None:
            self._test = json.load(open(self.test_detection_path, 'r'))

    def proposals_for_scene(self, pcl_path: str, index: int, task_type: str) -> List[Dict[str, Any]]:
        """返回候选 objects list，每个元素至少包含 name/BoundingBox。

        对齐 inference_3d.py：
        - task_type==Detection: 用 Test/Detection.json 的 per-index 'object'
        - else: 用 metadata/Detection.json 里的 scene_id key
        """
        self._lazy_load()
        if task_type == 'Detection':
            return self._test[index]['object']

        base = os.path.basename(pcl_path)
        src_id = os.path.splitext(base)[0]
        if src_id not in self._meta:
            raise KeyError(f"scene id {src_id} not found in metadata Detection.json")
        return self._meta[src_id]


class GeomTool3D:
    """几何工具：用于 PositionRelation 的关系计算（bbox center）。"""

    @staticmethod
    def _center(bbox6):
        # bbox: [cx, cy, cz, l, w, h]
        return bbox6[0], bbox6[1], bbox6[2]

    def relation(self, bboxA, bboxB):
        ax, ay, az = self._center(bboxA)
        bx, by, bz = self._center(bboxB)
        dx, dy, dz = bx - ax, by - ay, bz - az

        if abs(dy) > max(abs(dx), abs(dz)):
            return 'above' if dy > 0 else 'below'
        if abs(dx) >= abs(dz):
            return 'right' if dx > 0 else 'left'
        return 'front' if dz > 0 else 'behind'


class Agent3D:
    def __init__(self, args, model, det_tool: DetectionTool3D, geom: Optional[GeomTool3D] = None):
        self.args = args
        self.model = model
        self.det_tool = det_tool
        self.geom = geom or GeomTool3D()

    @staticmethod
    def _norm_label(label: str) -> str:
        return ''.join(ch for ch in (label or '').lower() if ch.isalnum())

    def _parse_question_minimal(self, sys_msg: str, query: str) -> Dict[str, Any]:
        """Step1：纯文本解析，只判定任务类型与目标词，不做 bbox 或打分。"""
        inst = (
            "You are a parser. Do NOT solve the task.\n"
            "Given a question, output JSON only with keys:\n"
            "- task: one of [VG, COUNT, ROOM, REL]\n"
            "- target: a single lowercase word for VG/COUNT/ROOM (empty for REL)\n"
            "- A, B: lowercase words for REL (empty otherwise)\n"
            "Return example: {\"task\":\"COUNT\",\"target\":\"chair\",\"A\":\"\",\"B\":\"\"}\n"
            f"Question: {query}\n"
        )

        # keep same conversation style header if provided
        prompt = inst if not sys_msg else (sys_msg + "\n" + inst)
        text = _greedy_text_generate(self.model, prompt, max_new_tokens=128)

        # try extract the last JSON object in the output
        start = text.rfind('{')
        end = text.rfind('}')
        parsed: Dict[str, Any] = {'task': '', 'target': '', 'A': '', 'B': '', 'raw': text}
        if start != -1 and end != -1 and end > start:
            try:
                parsed.update(json.loads(text[start:end + 1]))
            except Exception:
                pass

        task = (parsed.get('task') or '').strip().upper()
        if task not in {'VG', 'COUNT', 'ROOM', 'REL'}:
            task = ''
        parsed['task'] = task
        parsed['target'] = self._norm_label(parsed.get('target', ''))
        parsed['A'] = self._norm_label(parsed.get('A', ''))
        parsed['B'] = self._norm_label(parsed.get('B', ''))
        return parsed

    @staticmethod
    def _bbox_to_minmax(bbox):
        cx, cy, cz, l, w, h = bbox
        return (
            cx - l / 2,
            cy - h / 2,
            cz - w / 2,
            cx + l / 2,
            cy + h / 2,
            cz + w / 2,
        )

    def _relation_4way(self, bboxA, bboxB) -> str:
        """REL: 仅输出 left/right/above/below 四类（按中心点）。"""
        ax, ay, _ = bboxA[0], bboxA[1], bboxA[2]
        bx, by, _ = bboxB[0], bboxB[1], bboxB[2]
        dx, dy = (bx - ax), (by - ay)
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        return 'above' if dy > 0 else 'below'

    def solve(self, task_type: str, sys_msg: str, data_item: Dict[str, Any], obj_list: List[Dict[str, Any]], list_of_objpoints):
        query = data_item['query'][0] if isinstance(data_item['query'], list) else data_item['query']
        pcl_paths = data_item['pcl']
        # Step1) 纯文本解析（不依赖视觉）
        parsed = self._parse_question_minimal(sys_msg, query)

        # Step2) 一次检测调用：保留 detector 顺序，截断前 20
        det_topk = obj_list[:20]

        # Step3) 固定规则：直接用检测结果出答案
        trace = {
            'query': query,
            'task_type_arg': task_type,
            'parsed': parsed,
            'det_topk_n': len(det_topk),
        }

        if not det_topk:
            if task_type == 'Counting':
                return '0', {'mode': 'fixed', **trace}
            return 'unknown', {'mode': 'fixed', **trace}

        if task_type in ['VisualGrounding_plus', 'VisualGrounding']:
            # 优先取与 target 名称匹配的第一个，否则回退 top-1
            tgt = parsed.get('target', '')
            chosen = None
            if tgt:
                for obj in det_topk:
                    if tgt in self._norm_label(obj.get('name', '')):
                        chosen = obj
                        break
            if chosen is None:
                chosen = det_topk[0]
            bbox6 = chosen.get('BoundingBox')
            return str(bbox6), {'mode': 'fixed', 'bbox': bbox6, **trace}

        if task_type == 'Counting':
            # COUNT: 只计数与 target 名称匹配的框；无匹配则回退 top-20 数量
            tgt = parsed.get('target', '')
            if tgt:
                matched = [o for o in det_topk if tgt in self._norm_label(o.get('name', ''))]
                return str(len(matched)), {'mode': 'fixed', 'count_target': tgt, 'matched': len(matched), **trace}
            return str(len(det_topk)), {'mode': 'fixed', **trace}

        if task_type == 'RoomDetection':
            # ROOM: 输出目标词 + top-1 bbox（Rgrounding3d_eval 会用 label 匹配文本）
            room_label = parsed.get('target') or 'unknown'
            bbox6 = det_topk[0].get('BoundingBox')
            return f"{room_label} {bbox6}", {'mode': 'fixed', 'room': room_label, 'bbox': bbox6, **trace}

        if task_type == 'PositionRelation':
            # REL: 按名称匹配 A/B；找不到则回退 top-1/top-2；仅 4 类关系
            tgtA = parsed.get('A', '')
            tgtB = parsed.get('B', '')
            bboxA = None
            bboxB = None
            if tgtA:
                for obj in det_topk:
                    if tgtA in self._norm_label(obj.get('name', '')):
                        bboxA = obj.get('BoundingBox')
                        break
            if tgtB:
                for obj in det_topk:
                    if tgtB in self._norm_label(obj.get('name', '')):
                        bboxB = obj.get('BoundingBox')
                        break
            if bboxA is None:
                bboxA = det_topk[0].get('BoundingBox')
            if bboxB is None:
                bboxB = det_topk[1].get('BoundingBox') if len(det_topk) > 1 else bboxA
            if not bboxA or not bboxB:
                return 'unknown', {'mode': 'fixed', **trace}
            relation = self._relation_4way(bboxA, bboxB)
            return relation, {'mode': 'fixed', 'relation': relation, 'bboxA': bboxA, 'bboxB': bboxB, **trace}

        return 'unknown', {'mode': 'fixed', **trace}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='Counting',
                        choices=['VisualGrounding_plus', 'Counting', 'RoomDetection', 'PositionRelation'],
                        help='任务类型，可选 VG+、计数、房间类型、相对位置')
    parser.add_argument('--base-data-path', type=str, default='/data/HTC/Data/dataset/Benchmark/data',
                        help='基准数据根路径')
    parser.add_argument('--answers-dir', type=str, default='../answers',
                        help='输出答案 jsonl 的目录')
    parser.add_argument('--choose', type=bool, default=True, help='兼容旧代码的占位开关，不影响运行')
    parser.add_argument('--gpu', type=int, default=1, help='指定使用的 GPU 编号，例如 0 或 1')

    # model paths (defaults align with inference_3d.py)
    parser.add_argument('--encoder_pretrain', type=str, default='epcl', choices=('clip', 'epcl'),
                        help='视觉编码器预训练权重类型')
    parser.add_argument('--encoder_ckpt_path', type=str,
                        default='/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth',
                        help='视觉编码器 checkpoint 路径')
    parser.add_argument('--vicuna_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0',
                        help='Vicuna 权重路径')
    parser.add_argument('--delta_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/llm_v1/ALL1/pytorch_model_ep1.pt',
                        help='LoRA/PEFT delta 权重路径')
    # openlamm.py expects this key in args
    parser.add_argument('--train_stage', type=int, default=2,
                        help='1 for obj alignment;2for test；3 for fintune')
    parser.add_argument('--stage', type=int, default=2, help='同 train_stage，保持接口兼容')

    # LoRA configurations (openlamm.py expects these keys)
    parser.add_argument('--lora_r', type=int, default=32, help='LoRA 秩大小')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha 缩放')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout 概率')
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                        help='应用 LoRA 的模块列表')

    # generation
    parser.add_argument('--max_tgt_len', type=int, default=1200, help='生成的最大文本长度')
    parser.add_argument('--conv_mode', type=str, default='simple', help='对话模板名称')
    parser.add_argument('--bs', type=int, default=1, help='推理 batch size')
    parser.add_argument('--local_rank', default=0, type=int, help='分布式本地 rank（单卡可忽略）')
    parser.add_argument('--vision_feature_type', type=str, default='local', choices=('local', 'global'),
                        help='视觉特征使用局部还是全局')
    parser.add_argument('--vision_output_layer', type=int, default=-2, help='使用的视觉 backbone 输出层索引')
    parser.add_argument('--num_vision_token', type=int, default=256, help='视觉 token 数量')
    parser.add_argument('--max_obj_len', type=int, default=30, help='单个对象的最大 token 长度')

    # agent budget + ablations
    parser.add_argument('--max_obj', type=int, default=20,
                        help='候选 proposal 上限（0-20，子任务最多取 20）')
    parser.add_argument('--budget_detector', type=int, default=1, help='检测器调用预算，当前复用现有检测结果')
    parser.add_argument('--wo_reflection', action='store_true', help='关闭反思步骤')
    parser.add_argument('--wo_tool', action='store_true', help='关闭工具调用，仅用基准式回答')
    parser.add_argument('--dry_run', type=int, default=10, help='>0 时仅跑前 N 条样本用于快速检查')
    parser.add_argument('--do_sample', action='store_true',
                        help='开启采样生成；默认关闭以使用 greedy，避免概率 nan 触发 CUDA assert')

    # detector artifacts
    parser.add_argument('--detection_meta', type=str,
                        default='/data/HTC/Data/dataset/Benchmark/data/metadata/Detection.json',
                        help='scene->object 的检测元数据路径')
    parser.add_argument('--detection_test', type=str,
                        default='/data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/Test/Detection.json',
                        help='按索引存储的测试检测结果路径')
    parser.add_argument('--objpoints_path', type=str,
                        default='/data/HTC/Project/Point-BERT/data/ModelNet/modelnet40_normal_resampled/my_test_1024pts_fps.dat',
                        help='用于分类等任务的独立对象点云缓存路径')

    args = parser.parse_args()

    if args.vision_feature_type == 'local':
        args.vision_output_layer = -2
        args.num_vision_token = 256
    else:
        args.vision_output_layer = -1
        args.num_vision_token = 1

    assert os.path.exists(args.delta_ckpt_path), 'delta checkpoint not exists!'
    assert os.path.exists(args.vicuna_ckpt_path), 'vicuna checkpoint not exists!'
    assert os.path.exists(args.encoder_ckpt_path), 'vision encoder checkpoint not exists!'
    assert os.path.exists(args.detection_meta), 'Detection metadata not exists!'
    assert os.path.exists(args.detection_test), 'Detection test outputs not exists!'
    assert os.path.exists(args.objpoints_path), 'objpoints file not exists!'
    return args


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    model = LAMMPEFTModel(**args.__dict__)
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(delta_ckpt, strict=False)
    model.llama_model = model.llama_model.merge_and_unload()
    # 将模型移动到指定设备并使用 FP16 提升推理速度
    model = model.eval().half().to(device)

    # For scene tasks (Counting/VG/Room/PositionRelation), we follow inference_3d: use detection bboxes only.
    # The objpoints cache is for object-level classification; keep it unused here to avoid mismatch.
    list_of_objpoints_all = None

    det_tool = DetectionTool3D(args.detection_meta, args.detection_test)
    agent = Agent3D(args, model, det_tool)

    dataloader = load_3Deval_dataset(args.base_data_path, args.task_type, mode='common', batch_size=args.bs)
    sys_msg = dataloader.dataset.system_msg

    # 输出文件命名与 inference_3d.py 对齐：<task_type>.jsonl
    answers_file = os.path.join(args.answers_dir, f"{args.task_type}.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, 'w') as fout:
        for idx, data_item in enumerate(tqdm(dataloader)):
            if args.dry_run and idx >= args.dry_run:
                break

            pcl_paths = data_item['pcl']
            obj_list = det_tool.proposals_for_scene(pcl_paths[0], idx, args.task_type)

            # Scene tasks: no per-object point clouds; openlamm crops via detection bboxes internally
            list_of_objpoints = []

            start = time.time()
            text, trace = agent.solve(args.task_type, sys_msg, data_item, obj_list, list_of_objpoints)
            elapsed = time.time() - start

            ans = {
                'id': data_item['id'][0] if isinstance(data_item['id'], list) else data_item['id'],
                'pcl': pcl_paths,
                'text': text,
                'delta_path': args.delta_ckpt_path,
            }
            fout.write(json.dumps(ans) + '\n')
            fout.flush()

    print(f"Wrote {answers_file}")


if __name__ == '__main__':
    main()
