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

    inputs = {
        'prompt': prompt_text,
        'pcl_paths': pcl_paths,
        'top_p': safe_top_p,
        'temperature': safe_temperature,
        'max_tgt_len': max_length,
        'do_sample': args.do_sample,
        'modality_embeds': [],
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

    def _budget(self):
        return {
            'detector_calls': self.args.budget_detector,
            'max_obj': self.args.max_obj,
            'reflection': 0 if self.args.wo_reflection else 1,
            'use_tool': 0 if self.args.wo_tool else 1,
        }

    def _reflection_check(self, task_type: str, result: Dict[str, Any]) -> bool:
        if self.args.wo_reflection:
            return True
        if task_type == 'Counting':
            return isinstance(result.get('count', None), int)
        if task_type in ['VisualGrounding_plus', 'VisualGrounding']:
            return result.get('selected_obj', None) is not None
        if task_type == 'PositionRelation':
            return result.get('relation', None) in {'left', 'right', 'front', 'behind', 'above', 'below'}
        return True

    def _rank_objects_by_query(self, sys_msg, pcl_paths, obj_list, list_of_objpoints, query: str, top_k: int) -> Dict[str, Any]:
        options = []
        for i, obj in enumerate(obj_list[:top_k]):
            name = obj.get('name', 'Unknown')
            bbox = obj.get('BoundingBox', obj.get('bbox', None))
            options.append({'idx': i, 'name': name, 'bbox': bbox})

        prompt = [
            "You are given a caption to locate an object in a 3D scene. "
            "Choose the best matching object from the candidate list and answer strictly in JSON.\n"
            f"Caption: {query}\n"
            f"Candidates(JSON list): {json.dumps(options)}\n"
            "Return: {\"idx\": <int>, \"reason\": <short>}\n"
        ]
        resp = mllm_generate_one(
            self.args,
            self.model,
            prompt,
            sys_msg,
            pcl_paths=pcl_paths,
            obj_list=obj_list,
            list_of_objpoints=list_of_objpoints,
            task_type='VisualGrounding_plus',
            max_length=256,
            top_p=0.9,
            temperature=0.0,
        )
        text = resp[0].split('###')[0] if isinstance(resp, (list, tuple)) else str(resp)
        try:
            chosen = json.loads(text)
        except Exception:
            chosen = {'idx': None, 'raw': text}

        idx = chosen.get('idx', None)
        if isinstance(idx, int) and 0 <= idx < len(options):
            sel = options[idx]
        else:
            sel = None
        return {'selected_obj': sel, 'raw': text, 'candidates': options}

    def solve(self, task_type: str, sys_msg: str, data_item: Dict[str, Any], obj_list: List[Dict[str, Any]], list_of_objpoints):
        budget = self._budget()
        query = data_item['query'][0] if isinstance(data_item['query'], list) else data_item['query']
        pcl_paths = data_item['pcl']
        trace = {'budget': budget, 'query': query, 'task_type': task_type}

        if budget['use_tool'] == 0:
            resp = mllm_generate_one(
                self.args,
                self.model,
                [query],
                sys_msg,
                pcl_paths=pcl_paths,
                obj_list=obj_list,
                list_of_objpoints=list_of_objpoints,
                task_type=task_type,
                max_length=self.args.max_tgt_len,
                top_p=0.9,
                temperature=0.8,
            )
            text = resp[0].split('###')[0]
            return text, {'mode': 'bench', 'raw': text, **trace}

        if task_type in ['VisualGrounding_plus', 'VisualGrounding']:
            res = self._rank_objects_by_query(sys_msg, pcl_paths, obj_list, list_of_objpoints, query, top_k=budget['max_obj'])
            res['reflection_pass'] = self._reflection_check(task_type, res)
            if res['selected_obj'] is None:
                text = "I cannot find the object."
            else:
                # IMPORTANT: common_eval_3d.py 的 VG_plus_acc 用 parse_bbox_2d_Vis 读取 [x,y,z]
                # 并与 gt['bbox'][:3] 做距离<=1 判断。
                # 所以这里直接输出所选 bbox 的中心点 [cx,cy,cz]。
                bbox6 = res['selected_obj'].get('bbox')
                if bbox6 and len(bbox6) >= 3:
                    cx, cy, cz = bbox6[0], bbox6[1], bbox6[2]
                    text = f"obj{res['selected_obj']['idx']} {res['selected_obj']['name']} [{cx}, {cy}, {cz}]"
                else:
                    text = f"obj{res['selected_obj']['idx']} {res['selected_obj']['name']}"
            return text, {'mode': 'agent', **res, **trace}

        if task_type == 'Counting':
            parse_prompt = [
                "Extract the object category to count from the question. "
                "Return only one lowercase word (e.g., chair, table, bed).\n"
                f"Question: {query}"
            ]
            resp = mllm_generate_one(
                self.args,
                self.model,
                parse_prompt,
                sys_msg,
                pcl_paths=pcl_paths,
                obj_list=obj_list,
                list_of_objpoints=list_of_objpoints,
                task_type=task_type,
                max_length=64,
                top_p=0.9,
                temperature=0.0,
            )
            label = resp[0].split('###')[0].strip().lower()
            matched_idx = [
                i for i, o in enumerate(obj_list[:budget['max_obj']])
                if label and label in o.get('name', '').lower()
            ]
            count = int(len(matched_idx))
            res = {'label': label, 'count': count, 'matched_idx': matched_idx}
            res['reflection_pass'] = self._reflection_check(task_type, res)
            text = str(count)
            return text, {'mode': 'agent', **res, **trace}

        if task_type == 'PositionRelation':
            ab_prompt = [
                "Extract two object categories A and B from the question and return JSON only. "
                "Format: {\"A\":\"...\",\"B\":\"...\"}.\n"
                f"Question: {query}"
            ]
            resp = mllm_generate_one(
                self.args,
                self.model,
                ab_prompt,
                sys_msg,
                pcl_paths=pcl_paths,
                obj_list=obj_list,
                list_of_objpoints=list_of_objpoints,
                task_type=task_type,
                max_length=128,
                top_p=0.9,
                temperature=0.0,
            )
            raw = resp[0].split('###')[0]
            try:
                ab = json.loads(raw)
            except Exception:
                ab = {'A': '', 'B': '', 'raw': raw}

            def pick_first(name: str):
                name = (name or '').lower().strip()
                for i, o in enumerate(obj_list[:budget['max_obj']]):
                    if name and name in o.get('name', '').lower():
                        return i, o
                return None, None

            a_idx, a_obj = pick_first(ab.get('A', ''))
            b_idx, b_obj = pick_first(ab.get('B', ''))

            relation = None
            if a_obj and b_obj:
                relation = self.geom.relation(a_obj['BoundingBox'], b_obj['BoundingBox'])
            res = {
                'A': ab.get('A', ''),
                'B': ab.get('B', ''),
                'A_idx': a_idx,
                'B_idx': b_idx,
                'relation': relation,
                'ab_raw': raw,
            }
            res['reflection_pass'] = self._reflection_check(task_type, res)
            # common_eval_3d.py 的 Positoinacc 是用 GPT 判断“句子语义是否一致”，
            # 因此直接输出一个完整句子更稳。
            if relation is None:
                text = 'unknown'
            else:
                text = f"The {ab.get('A','object A')} is {relation} of the {ab.get('B','object B')}."
            return text, {'mode': 'agent', **res, **trace}

        if task_type == 'RoomDetection':
            room_prompt = [
                "Given a 3D indoor scene candidate objects (name + bbox), predict the room type. "
                "Choose one from [bedroom,kitchen,livingroom,bathroom,diningroom,office]. "
                "Return JSON only: {\"room\":\"...\"}.\n"
                f"Objects: {json.dumps([{'name': o.get('name'), 'bbox': o.get('BoundingBox')} for o in obj_list[:budget['max_obj']]])}"
            ]
            resp = mllm_generate_one(
                self.args,
                self.model,
                room_prompt,
                sys_msg,
                pcl_paths=pcl_paths,
                obj_list=obj_list,
                list_of_objpoints=list_of_objpoints,
                task_type=task_type,
                max_length=128,
                top_p=0.9,
                temperature=0.0,
            )
            raw = resp[0].split('###')[0]
            try:
                room = json.loads(raw).get('room', None)
            except Exception:
                room = None

            # union bbox (axis-aligned, center-length form)
            bxs = [o.get('BoundingBox') for o in obj_list[:budget['max_obj']] if o.get('BoundingBox') is not None]
            union = None
            if bxs:
                xs = [b[0] for b in bxs]
                ys = [b[1] for b in bxs]
                zs = [b[2] for b in bxs]
                ls = [b[3] for b in bxs]
                ws = [b[4] for b in bxs]
                hs = [b[5] for b in bxs]

                minx = min(x - l / 2 for x, l in zip(xs, ls))
                maxx = max(x + l / 2 for x, l in zip(xs, ls))
                miny = min(y - h / 2 for y, h in zip(ys, hs))
                maxy = max(y + h / 2 for y, h in zip(ys, hs))
                minz = min(z - w / 2 for z, w in zip(zs, ws))
                maxz = max(z + w / 2 for z, w in zip(zs, ws))

                cx = (minx + maxx) / 2
                cy = (miny + maxy) / 2
                cz = (minz + maxz) / 2
                union = [cx, cy, cz, (maxx - minx), (maxz - minz), (maxy - miny)]

            res = {'room': room, 'room_bbox': union, 'raw': raw}
            res['reflection_pass'] = self._reflection_check(task_type, res)

            if union is None:
                text = room or 'unknown'
            else:
                # Rgrounding3d_eval: parse_bbox_3d_Vis(text) 读取 bbox；classification_acc(object_info['label'], text) 用 label 匹配。
                # 因此把 label(room) + bbox6 输出到同一行。
                text = f"{room} {union}"
            return text, {'mode': 'agent', **res, **trace}

        resp = mllm_generate_one(
            self.args,
            self.model,
            [query],
            sys_msg,
            pcl_paths=pcl_paths,
            obj_list=obj_list,
            list_of_objpoints=list_of_objpoints,
            task_type=task_type,
            max_length=self.args.max_tgt_len,
            top_p=0.9,
            temperature=0.8,
        )
        text = resp[0].split('###')[0]
        return text, {'mode': 'fallback', 'raw': text, **trace}


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
                        help='1/2 表示仅做目标对齐，3 表示全量阶段（与 inference_3d.py 一致）')
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
    parser.add_argument('--dry_run', type=int, default=0, help='>0 时仅跑前 N 条样本用于快速检查')
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
                        help='预存的对象点云缓存路径')

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

    with open(args.objpoints_path, 'rb') as f:
        objpoints_pack = pickle.load(f)
    list_of_objpoints_all = objpoints_pack[0]

    det_tool = DetectionTool3D(args.detection_meta, args.detection_test)
    agent = Agent3D(args, model, det_tool)

    dataloader = load_3Deval_dataset(args.base_data_path, args.task_type, mode='common', batch_size=args.bs)
    sys_msg = dataloader.dataset.system_msg

    out_name = f"Agent_{args.task_type}"
    if args.wo_tool:
        out_name += "_wotool"
    if args.wo_reflection:
        out_name += "_worefl"

    answers_file = os.path.join(args.answers_dir, out_name + '.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, 'w') as fout:
        for idx, data_item in enumerate(tqdm(dataloader)):
            if args.dry_run and idx >= args.dry_run:
                break

            pcl_paths = data_item['pcl']
            obj_list = det_tool.proposals_for_scene(pcl_paths[0], idx, args.task_type)

            # inference_3d.py: list_of_objpoints[0][index]
            list_of_objpoints = list_of_objpoints_all[idx]

            start = time.time()
            text, trace = agent.solve(args.task_type, sys_msg, data_item, obj_list, list_of_objpoints)
            elapsed = time.time() - start

            ans = {
                'id': data_item['id'][0] if isinstance(data_item['id'], list) else data_item['id'],
                'pcl': pcl_paths,
                'text': text,
                'delta_path': args.delta_ckpt_path,
                'agent_trace': {**trace, 'elapsed': elapsed},
            }
            fout.write(json.dumps(ans) + '\n')
            fout.flush()

    print(f"Wrote {answers_file}")


if __name__ == '__main__':
    main()
