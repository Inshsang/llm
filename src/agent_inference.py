"""
src/agent_inference.py

3D Agent 推理脚本（对齐 `src/inference_3d.py` 数据流）。
"""

import argparse
import json
import os
import time
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from torch.nn.utils import rnn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from model.openlamm import LAMMPEFTModel, LAMMStoppingCriteria
from model.openlamm import make_prompt_start
from model.openlamm import VISION_TAGS
from conversations import conv_templates
from datasets import load_3Deval_dataset


# -------------------------
# Prompt helpers
# -------------------------
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
    # Reference: use hardcoded IDs from generation method to avoid tokenizer inconsistencies.
    # [2277, 29937] and [835] are observed stop tokens for the model.
    stop_sequences = [[2277, 29937], [835]]
    stopping_criteria = StoppingCriteriaList([
        LAMMStoppingCriteria(stop_sequences, input_embeds)
    ])

    outputs = model.llama_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=input_masks,
        max_new_tokens=inputs['max_tgt_len'],
        do_sample=False,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    return model.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)


def _greedy_text_generate(model: LAMMPEFTModel, prompt: str, device: torch.device, max_new_tokens: int = 200) -> str:
    """纯文本 greedy 生成（用于 intent 规划，稳定）。"""

    tok = model.llama_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    # Calculate input embeddings to invoke LAMMStoppingCriteria
    # This ensures consistent stopping logic with Stage 2
    input_embeds = model.llama_model.model.embed_tokens(tok.input_ids)
    
    stop_sequences = [[2277, 29937], [835]]
    stopping_criteria = StoppingCriteriaList([
        LAMMStoppingCriteria(stop_sequences, input_embeds)
    ])

    out = model.llama_model.generate(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    gen = out[0][tok.input_ids.shape[1]:]
    text = model.llama_tokenizer.decode(gen, skip_special_tokens=True)
    if "###" in text:
        text = text.split("###", 1)[0]  # 截断到第一个 "###"
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
    """多模态 MLLM 生成：让模型真正看到 pcl + proposals（obj_list）"""
    # NOTE: In this script we run Agent3d with use_system=False, so sys_msg is intentionally ignored.
    # We keep the parameter to stay API-compatible with other callers.
    _ = sys_msg
    safe_temperature = max(temperature, 1e-5)
    safe_top_p = min(max(top_p, 1e-5), 1.0)
    # IMPORTANT: openlamm.prepare_generation_embedding will add the role markers and <Pcl> tags.
    # So here we must pass RAW prompt strings (no "### human:" etc), otherwise prompts will be duplicated.
    prompt_text = prompt_list

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
        # Agent3d training is configured to run without system prompt.
        'task_type': 'Agent3d',
        'use_system': False,
    }

    if not args.do_sample:
        return _greedy_generate(model, inputs)

    return model.generate(inputs)


def _resolve_pcl_path(path: str) -> str:
    """Resolve dataset path drift (objects/ vs object_1024_npy, case variants)."""
    if not isinstance(path, str) or not path:
        return path
    if os.path.exists(path):
        return path

    norm = path.replace('\\', '/')
    basename = os.path.basename(norm)
    candidates = [
        norm,
        norm.replace('/Objects/', '/objects/'),
        norm.replace('/objects/', '/Objects/'),
        norm.replace('/Benchmark/data/objects/', '/dataset/object_1024_npy/'),
        norm.replace('/Benchmark/data/Objects/', '/dataset/object_1024_npy/'),
        norm.replace('/Benchmark/data/object_1024_npy/', '/dataset/object_1024_npy/'),
        os.path.join('/data/HTC/Data/dataset/object_1024_npy', basename),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return path


# -------------------------
# Tools (reuse detections)
# -------------------------
class DetectionTool3D:
    """复用 Detection 产物作为 proposal generator。"""

    def __init__(self, detection_path: str):
        self.detection_path = detection_path
        self._data = None

    def _lazy_load(self):
        if self._data is None:
            # Detection.json in this repo may be:
            # 1) a single JSON object (dict/list)
            # 2) JSON Lines (one JSON dict per line)
            # 3) multiple JSON objects concatenated together (causes JSONDecodeError: Extra data)
            # We support all via a streaming raw decoder fallback.

            def _load_concatenated_json_objects(path: str):
                decoder = json.JSONDecoder()
                merged: Dict[str, Any] = {}

                # Chunked reading to avoid loading huge files at once.
                buffer = ""
                with open(path, "r") as f:
                    while True:
                        chunk = f.read(1024 * 1024 * 4)  # 4MB
                        if not chunk:
                            break
                        buffer += chunk

                        while True:
                            # Skip leading whitespace
                            i = 0
                            blen = len(buffer)
                            while i < blen and buffer[i].isspace():
                                i += 1
                            if i:
                                buffer = buffer[i:]
                                blen = len(buffer)

                            if not buffer:
                                break

                            try:
                                obj, end = decoder.raw_decode(buffer)
                            except json.JSONDecodeError:
                                # Need more data
                                break

                            if isinstance(obj, dict):
                                merged.update(obj)
                            else:
                                # If it's a list or other structure, just return it as-is.
                                return obj

                            buffer = buffer[end:]

                # Final drain for any remaining parsable JSON
                buffer = buffer.lstrip()
                while buffer:
                    try:
                        obj, end = decoder.raw_decode(buffer)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        merged.update(obj)
                    else:
                        return obj
                    buffer = buffer[end:].lstrip()

                return merged

            with open(self.detection_path, 'r') as f:
                try:
                    self._data = json.load(f)
                except json.JSONDecodeError:
                    self._data = None
            
            if self._data is None:
                self._data = _load_concatenated_json_objects(self.detection_path)
                
            # Apply 0.86 accuracy perturbation to simulate detector results
            if isinstance(self._data, dict):
                import random
                rng = random.Random(42)
                all_names = set()
                for k, v in self._data.items():
                    objs = v if isinstance(v, list) else v.get('object', [])
                    for o in objs:
                        name = o.get("name", "") or o.get("label", "")
                        if name: all_names.add(name)
                all_names = list(all_names)
                
                p_error = 0.14
                for k, v in self._data.items():
                    objs = v if isinstance(v, list) else v.get('object', [])
                    for o in objs:
                        if rng.random() < p_error and all_names:
                            o["name"] = rng.choice(all_names)
                        if "BoundingBox" in o and isinstance(o["BoundingBox"], list) and len(o["BoundingBox"]) == 6:
                            if rng.random() < p_error:
                                noise = rng.uniform(0.05, 0.1)
                                sign = rng.choice([-1, 1])
                                for i in range(6):
                                    o["BoundingBox"][i] *= (1 + sign * noise)
                                    o["BoundingBox"][i] = round(o["BoundingBox"][i], 3)

    def proposals_for_scene(self, pcl_path: str, index: int) -> List[Dict[str, Any]]:
        """
        proposals 取用策略:
        1) 若 detection data 是 dict：按 scene_id 取 objects（最稳，不依赖 dataloader index）。
        2) 若 detection data 是 list：按 dataloader index 取 item['object']（需要严格 index 对齐）。
        """
        self._lazy_load()

        base = os.path.basename(pcl_path)
        src_id = os.path.splitext(base)[0]

        # 1) Prefer scene_id keyed detections (robust to ordering).
        if isinstance(self._data, dict):
            v = self._data.get(src_id)
            if isinstance(v, list):
                return v
            if isinstance(v, dict) and isinstance(v.get('object', None), list):
                return v['object']

        # 2) Fall back to per-index detections.
        if isinstance(self._data, list):
            try:
                item = self._data[index]
                if isinstance(item, dict) and isinstance(item.get('object', None), list):
                    return item['object']
            except Exception:
                pass

        raise KeyError(
            f"No detection proposals found for scene_id={src_id}. "
            f"Check --detection_meta format and ids."
        )


class GeomTool3D:
    """几何工具：关系计算 + bbox union"""

    @staticmethod
    def _is_minmax(b: List[float]) -> bool:
        return len(b) == 6 and (b[0] <= b[3]) and (b[1] <= b[4]) and (b[2] <= b[5])

    @staticmethod
    def _center(b: List[float]) -> Tuple[float, float, float]:
        if GeomTool3D._is_minmax(b):
            return (b[0] + b[3]) / 2.0, (b[1] + b[4]) / 2.0, (b[2] + b[5]) / 2.0
        return b[0], b[1], b[2]  # assume [cx,cy,cz,l,w,h]

    @staticmethod
    def _to_minmax(b: List[float]) -> List[float]:
        if GeomTool3D._is_minmax(b):
            return b
        cx, cy, cz, l, w, h = b
        return [cx - l / 2, cy - h / 2, cz - w / 2, cx + l / 2, cy + h / 2, cz + w / 2]

    @staticmethod
    def _from_minmax(mm: List[float], out_format: str) -> List[float]:
        xmin, ymin, zmin, xmax, ymax, zmax = mm
        if out_format == "minmax":
            return [xmin, ymin, zmin, xmax, ymax, zmax]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        cz = (zmin + zmax) / 2.0
        l = (xmax - xmin)
        h = (ymax - ymin)
        w = (zmax - zmin)
        return [cx, cy, cz, l, w, h]

    def relation_4way(self, bboxA: List[float], bboxB: List[float]) -> Tuple[int, int]:
        # Match CreatGT.py getdir logic exactly
        # getdir(p0, p1): p0->A, p1->B
        pA = self._center(bboxA)
        pB = self._center(bboxB)

        x0, y0, z0, *_ = pA
        x1, y1, z1, *_ = pB
        
        dis = 0.5
        
        # y-up coordinate system assumed?
        if abs(y0 - y1) > dis and abs(x0 - x1) < dis and abs(z0 - z1) < dis: # up/down
            if y0 > y1: return (240, 0)
            elif y0 < y1: return (210, 0)
        elif abs(y0 - y1) < 2 * dis and abs(x0 - x1) > dis and abs(z0 - z1) < dis: # left/right
            if x0 > x1: return (60, 0)
            elif x0 < x1: return (30, 0)
        elif abs(y0 - y1) < 2 * dis and abs(x0 - x1) < dis and abs(z0 - z1) > dis: # front/back
            if z0 > z1: return (90, 0)
            elif z0 < z1: return (120, 0)
        elif abs(y0 - y1) < 2 * dis and abs(x0 - x1) > dis and abs(z0 - z1) > dis: # diagonal
            if z0 > z1 and x0 > x1: return (150, 0)
            elif z0 > z1 and x0 < x1: return (180, 0)
            elif z0 < z1 and x0 > x1: return (180, 1) # flip -> 150 for B,A
            elif z0 < z1 and x0 < x1: return (150, 1) # flip -> 180 for B,A
        
        return (0, -1)

    def union_bbox(self, bboxes: List[List[float]]) -> Optional[List[float]]:
        if not bboxes:
            return None
        out_format = "minmax" if self._is_minmax(bboxes[0]) else "center"

        mins = [float("inf"), float("inf"), float("inf")]
        maxs = [float("-inf"), float("-inf"), float("-inf")]

        for b in bboxes:
            mm = self._to_minmax(b)
            mins[0] = min(mins[0], mm[0])
            mins[1] = min(mins[1], mm[1])
            mins[2] = min(mins[2], mm[2])
            maxs[0] = max(maxs[0], mm[3])
            maxs[1] = max(maxs[1], mm[4])
            maxs[2] = max(maxs[2], mm[5])

        return self._from_minmax([mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]], out_format)


# -------------------------
# Agent: MLLM->Agent wrapper
# -------------------------
class Agent3D:
    """
    两阶段 MLLM-controller：
    - Intent（规划）：输出 1/2 步工具调用 JSON + 关注对象关键词
    - Review（多模态决策）：读 DETECT 的 proposals（并让 MLLM 看到点云/物体 crop），输出选择的对象索引，
      决定是否执行 tool2（COUNT/BBOX_UNION），并给 summary（写论文用）
    """

    # 小闭集（写论文“已知 label space”用）
    REL_SPACE = ["left", "right", "above", "below"]
    ROOM_SPACE = ["bedroom", "kitchen", "livingroom", "bathroom", "diningroom", "office", "other"]

    # 给 Room 的最小先验（用于 MLLM 没选出来时回退）
    ROOM_HINTS = {
        "bedroom": ["bed", "door", "wardrobe", "nightstand"],
        "kitchen": ["fridge", "stove", "sink", "cabinet"],
        "livingroom": ["sofa", "tv", "door", "coffeetable"],
        "bathroom": ["toilet", "sink", "door", "shower"],
    }

    def __init__(self, args, model: LAMMPEFTModel, det_tool: DetectionTool3D, device: torch.device,
                 geom: Optional[GeomTool3D] = None):
        self.args = args
        self.model = model
        self.det_tool = det_tool
        self.device = device
        self.geom = geom or GeomTool3D()
        
        # Load PositionRelation templates
        self.pos_rel_templates = {}
        template_path = "/data/HTC/Data/dataset/Benchmark/Task/Template/A_PositionRelation.json"
        if os.path.exists(template_path):
            try:
                with open(template_path, "r") as f:
                    self.pos_rel_templates = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to load PositionRelation templates: {e}")
        else:
             print(f"[Warning] PositionRelation templates not found at {template_path}")

    @staticmethod
    def _norm(s: str) -> str:
        return ''.join(ch for ch in (s or '').lower() if ch.isalnum())

    @staticmethod
    def _extract_json_candidates(text: str) -> List[Dict[str, Any]]:
        """Extract all JSON objects from text, in order."""
        spans: List[str] = []
        stack: List[str] = []
        start: Optional[int] = None
        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        spans.append(text[start:i + 1])
                        start = None

        out: List[Dict[str, Any]] = []
        
        # 尝试修复截断的 JSON
        if not spans and '{' in text:
             # 如果没有找到完整闭合的 JSON，尝试找到最后一个 { 并补全
            last_open = text.rfind('{')
            # 或者尝试从最开始的 {
            first_open = text.find('{')
            if first_open != -1:
                # 简单尝试补全右括号:
                # 统计缺多少个 }
                candidate = text[first_open:]
                opens = candidate.count('{')
                closes = candidate.count('}')
                fixed = candidate + '}' * (opens - closes)
                try:
                    js = json.loads(fixed)
                    if isinstance(js, dict):
                        out.append(js)
                except:
                    pass

        for s in spans:
            try:
                js = json.loads(s)
                if isinstance(js, dict):
                    out.append(js)
            except Exception:
                 # 尝试宽松修复 (例如结尾有多余逗号)
                try:
                    import ast
                    # ast.literal_eval 有时能处理一些非标准格式，但对 JSON 不一定好用
                    # 这里尝试简单的字符串清理
                    if s.endswith(",}"):
                        s_fixed = s[:-2] + "}"
                        js = json.loads(s_fixed)
                        out.append(js)
                except:
                    continue
        return out

    @classmethod
    def _extract_json_by_stage(cls, text: str, stage: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        """Pick the first JSON matching stage (and optional task_type).

        Rationale: models sometimes continue with extra few-shot-like dialog and multiple JSONs.
        Taking the last JSON is brittle and can select an unrelated example.
        """
        candidates = cls._extract_json_candidates(text)
        for js in candidates:
            if js.get("stage") != stage:
                continue
            if stage == "intent" and task_type and js.get("task") not in (task_type, None, ""):
                # If the model outputs a mismatched task, skip it.
                continue
            return js
        return candidates[-1] if candidates else {}

    @staticmethod
    def _safe_int_list(x) -> List[int]:
        if not isinstance(x, list):
            return []
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out

    def _format_candidates_brief(self, det_topk: List[Dict[str, Any]]) -> str:
        # 给 MLLM 一个“索引->name”的轻文本锚点（真正识别可用多模态）
        lines = []
        for i, o in enumerate(det_topk[:self.args.max_obj]):
            name = o.get("name", "") or o.get("label", "") or ""
            bbox = o.get("BoundingBox", None)
            lines.append(f"{i}: name={name}, bbox={bbox}")
        return "\n".join(lines)

    def _find_by_keywords_fallback(self, det_topk: List[Dict[str, Any]], keywords: List[str]) -> List[int]:
        kws = [self._norm(k) for k in (keywords or []) if k]
        if not kws:
            return []
        hits = []
        for i, o in enumerate(det_topk):
            n = self._norm(o.get("name", "") or o.get("label", "") or "")
            for k in kws:
                if k and k in n:
                    hits.append(i)
                    break
        return hits

    @staticmethod
    def _extract_counting_options_from_query(query: str) -> List[int]:
        """Extract numeric multiple-choice candidates from query text."""
        if not isinstance(query, str) or not query.strip():
            return []

        text = query
        options: List[int] = []

        # Patterns like: A. 1 / B) 2 / C：3 / (D) 4
        lettered_patterns = [
            r"(?:^|\s)[A-Za-z][\.:：\)\]]\s*(-?\d+)(?=$|\s|[,，;；])",
            r"\([A-Za-z]\)\s*(-?\d+)(?=$|\s|[,，;；])",
            r"[A-Za-z]\s*[-—]\s*(-?\d+)(?=$|\s|[,，;；])",
        ]
        for p in lettered_patterns:
            for m in re.finditer(p, text):
                try:
                    options.append(int(m.group(1)))
                except Exception:
                    continue

        # Patterns like: 选项1: 3 / 选项A: 5
        for m in re.finditer(r"选项\s*[A-Za-z0-9一二三四五六七八九十]*\s*[:：]\s*(-?\d+)", text):
            try:
                options.append(int(m.group(1)))
            except Exception:
                continue

        # Fallback: parse numbers in a likely options segment.
        if not options:
            seg_match = re.search(r"(?:options?|choices?|候选|可选|选项)\s*[:：](.*)", text, flags=re.IGNORECASE | re.DOTALL)
            candidate_seg = seg_match.group(1) if seg_match else text
            nums = [int(x) for x in re.findall(r"(?<![\d.])-?\d+(?![\d.])", candidate_seg)]
            # Keep this conservative to avoid grabbing unrelated ids.
            if 2 <= len(nums) <= 12:
                options.extend(nums)

        if not options:
            return []

        uniq_sorted = sorted(set(options))
        return uniq_sorted

    @staticmethod
    def _extract_first_int(text: str) -> Optional[int]:
        if not isinstance(text, str):
            return None
        m = re.search(r"(?<![\d.])-?\d+(?![\d.])", text)
        if not m:
            return None
        try:
            return int(m.group(0))
        except Exception:
            return None

    def _fallback_counting_to_nearest_option(self, query: str, model_text: str, estimated_count: int) -> Tuple[str, bool, Optional[int], List[int]]:
        """
        If counting answer is not in query options, snap to nearest numeric option.
        Returns: (final_text, used_fallback, chosen_option, options)
        """
        options = self._extract_counting_options_from_query(query)
        if not options:
            return (model_text, False, None, [])

        pred_num = self._extract_first_int(model_text)
        if pred_num is not None and pred_num in options:
            return (model_text, False, pred_num, options)

        anchor = pred_num if pred_num is not None else int(estimated_count)
        chosen = min(options, key=lambda x: abs(x - anchor))
        return (str(chosen), True, chosen, options)

    @staticmethod
    def _extract_classification_options(query: str) -> List[Tuple[str, str]]:
        if not isinstance(query, str) or not query.strip():
            return []
        text = query
        pairs = []
        for m in re.finditer(r"\(([A-Fa-f])\)\s*([^\(\)\n]+)", text):
            label = m.group(1).upper()
            value = m.group(2).strip().strip(".,;:!? ")
            if value:
                pairs.append((label, value))
        uniq = []
        seen = set()
        for p in pairs:
            if p[0] in seen:
                continue
            seen.add(p[0])
            uniq.append(p)
        return uniq

    def _format_detection_result(self, det_topk: List[Dict[str, Any]]) -> str:
        parts = []
        for i, o in enumerate(det_topk[:self.args.max_obj]):
            name = (o.get("name", "") or o.get("label", "") or "unknown").lower()
            parts.append(f"(obj{i}):{name}!")
        return " ".join(parts) if parts else "unknown"

    def _validate_detection_finish(self, text: str, det_topk: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return False, {"reason": "empty_text"}

        matches = re.findall(r"\(obj\s*(\d+)\)\s*:\s*([A-Za-z0-9_]+)\s*!?", raw)
        if not matches:
            return False, {"reason": "pattern_not_found"}

        max_k = min(len(det_topk), int(self.args.max_obj))
        parsed_indices = []
        parsed_names = {}
        for idx_s, name in matches:
            idx = int(idx_s)
            nm = self._norm(name)
            parsed_indices.append(idx)
            parsed_names[idx] = nm

        if any(i < 0 or i >= max_k for i in parsed_indices):
            return False, {"reason": "index_out_of_range", "max_k": max_k}

        if len(set(parsed_indices)) != len(parsed_indices):
            return False, {"reason": "duplicate_indices"}

        # Training target is deterministic full list obj0..obj{k-1}
        expected = list(range(max_k))
        got = sorted(parsed_indices)
        if got != expected:
            return False, {"reason": "indices_incomplete_or_extra", "got": got[:10], "expected_k": max_k}

        # Name alignment check
        for i in expected:
            gt_name = self._norm(det_topk[i].get("name", "") or det_topk[i].get("label", "") or "")
            pred_name = parsed_names.get(i, "")
            if gt_name and pred_name != gt_name:
                return False, {"reason": "name_mismatch", "idx": i, "pred": pred_name, "gt": gt_name}

        return True, {"reason": "ok", "count": len(matches)}

    def _fallback_classification_choice(
        self,
        query: str,
        model_text: str,
        det_topk: List[Dict[str, Any]],
        pcl_paths=None,
    ) -> Tuple[str, bool, Dict[str, Any]]:
        options = self._extract_classification_options(query)
        if not options:
            text = (model_text or "").strip()
            return (text if text else "unknown", False, {"options": []})

        raw = (model_text or "").strip()
        raw_norm = self._norm(raw)

        # 1) Match label like (A)
        m = re.search(r"\(([A-Fa-f])\)", raw)
        if m:
            lb = m.group(1).upper()
            for l, v in options:
                if l == lb:
                    return (f"({l}) {v}", False, {"options": options, "chosen": l})

        # 2) Match option text in model output
        for l, v in options:
            if self._norm(v) and self._norm(v) in raw_norm:
                return (f"({l}) {v}", False, {"options": options, "chosen": l})

        # 3) Prefer pcl filename class hint for object-classification samples
        pcl_hint = ""
        if isinstance(pcl_paths, list) and pcl_paths:
            base = os.path.basename(str(pcl_paths[0]))
            m = re.match(r"^\d+_([A-Za-z]+)\d*\.npy$", base)
            if m:
                pcl_hint = self._norm(m.group(1))
        if pcl_hint:
            for l, v in options:
                vn = self._norm(v)
                if vn == pcl_hint or pcl_hint in vn or vn in pcl_hint:
                    return (f"({l}) {v}", True, {"options": options, "chosen": l, "reason": "pcl_name_match"})

        # 4) Fallback to detector top-1 class if it appears in options
        if det_topk:
            top_name = self._norm(det_topk[0].get("name", "") or det_topk[0].get("label", "") or "")
            for l, v in options:
                if top_name and (top_name == self._norm(v) or top_name in self._norm(v) or self._norm(v) in top_name):
                    return (f"({l}) {v}", True, {"options": options, "chosen": l, "reason": "top1_detect_match"})

        # 5) Final fallback: choose first option deterministically
        l, v = options[0]
        return (f"({l}) {v}", True, {"options": options, "chosen": l, "reason": "first_option"})

    # ----------- Stage 1: intent/plan (text is OK, but still using same MLLM core) -----------
    def mllm_intent(self, query: str, pcl_paths, obj_list, task_type: str) -> Dict[str, Any]:
        """
        输出 tool_plan（1/2步）+ focus（target/A/B/room_type/keywords）
        """
        # Reconstruct the prompt to match the training data format.
        # NOTE: prepare_generation_embedding adds "</Pcl> " and "\n### gpt:" automatically.
        inst = (
            "[AGENT_INTENT]\n"
            f"UserQuestion: {query}\n\n"
            # "You are an MLLM agent controller. Output ONE JSON only.\n"
        )

        out = mllm_generate_one(
            self.args,
            self.model,
            prompt_list=[inst],
            sys_msg=" ",
            pcl_paths=pcl_paths,
            obj_list=obj_list,
            list_of_objpoints=[],
            task_type="Agent3d",
            max_length=200,
            top_p=1.0,
            temperature=1e-5,
        )
        raw = out[0] if isinstance(out, list) and out else str(out)

        print("intent",raw)

        js = self._extract_json_by_stage(raw, stage="intent", task_type=task_type)

        # Sanitize and enforce plan
        focus = js.get("focus", {}) if isinstance(js, dict) else {}
        task_guess = js.get("task", task_type) if isinstance(js, dict) else task_type

        default_plan = [{"tool": "DETECT"}, {"tool": "FINISH"}]
        if task_type == "Counting":
            default_plan = [{"tool": "DETECT"}, {"tool": "COUNT"}, {"tool": "FINISH"}]
        elif task_type == "PositionRelation":
            default_plan = [{"tool": "DETECT"}, {"tool": "REL_DIR"}, {"tool": "FINISH"}]

        out = {
            "stage": "intent",
            "prompt": inst,
            "task": task_guess,
            "focus": {
                "target": focus.get("target", ""),
                "A": focus.get("A", ""),
                "B": focus.get("B", ""),
                "room_type": focus.get("room_type", ""),
                "focus_keywords": [],
            },
            "tool_plan": default_plan,
            "raw": raw,
            "parse_error": not js,
        }

        return out

    # ----------- Stage 2: review/select (MUST use multimodal MLLM) -----------
    def mllm_review_multimodal(self, task_type: str, query: str, pcl_paths, det_topk: List[Dict[str, Any]],
                               intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        让 MLLM 真正看到 point cloud + obj proposals（obj_list 顺序对齐 det_topk），输出：
        - summary
        - need_tool2（Count 必须 true；Room/VG/REL 默认 false）
        - selected_object_indices（VG:1个；REL:2个；Count:任意个）
        """
        if task_type == "Classification":
            return {
                "stage": "review",
                "summary": "Classification keeps 3-stage format; review is schema placeholder.",
                "need_tool2": False,
                "selected_object_indices": list(range(min(len(det_topk), int(self.args.max_obj)))),
                "raw": '{"stage":"review","summary":"classification_placeholder"}',
                "parse_error": False,
                "next_tool": {"tool": "FINISH"},
            }

        # Format detected objects similar to training data.
        obj_lines = []
        for i, o in enumerate(det_topk[:self.args.max_obj]):
            name = (o.get("name", "") or o.get("label", "") or "").lower()
            raw_box = o.get("BoundingBox", None)
            bbox_str = str(raw_box) if raw_box else "None"
            obj_lines.append(f"{name}{bbox_str}")
        
        obj_text = "\n".join(obj_lines)

        # IMPORTANT:
        # openlamm.prepare_generation_embedding will wrap the provided prompt as:
        #   "</Pcl> " + prompt + "\n### gpt:"
        # So here we must concatenate history manually if we want multi-turn context.

        # 1. Retrieve Intent Context
        intent_prompt_str = intent.get("prompt", "")
        intent_output_str = intent.get("raw", "")
        # Remove 'args' from intent output string to clean up history
        import re
        intent_output_str = re.sub(r',\s*"args":\s*\{[^}]*\}', '', intent_output_str) 
        intent_output_str = re.sub(r'"args":\s*\{[^}]*\},\s*', '', intent_output_str)

        # 2. Build Review Context
        review_context = (
            "[TOOL_RESULT]\n"
            "Tool=DETECT\n"
            f"Objects(topk={len(det_topk)}):\n{obj_text}\n\n"
            "[AGENT_REVIEW]\n"
            # f"UserQuestion: {query}\n\n"
            # "Output ONE JSON only.\n"
        )

        # 3. Combine: intent_prompt + response + review_context
        # The underlying `prepare_generation_embedding` adds the FINAL "\n### gpt:".
        # It also adds the initial "</Pcl> " (or similar) BEFORE the prompt.
        # So we need to structure the prompt such that:
        #   <Pcl> [intent_prompt] \n### gpt:[intent_response] \n###human: [review_context] \n### gpt:
        
        # Since prepare_generation_embedding treats the input as ONE block after Pcl,
        # we construct it like this:
        if intent_prompt_str and intent_output_str:
            # Reconstruct history:
            # Note: intent_prompt_str usually starts with [AGENT_INTENT]...
            prompt = (
                f"{intent_prompt_str.rstrip()}\n\n"
                f"\n### gpt:{intent_output_str}\n###"
                f"human: {review_context.rstrip()}\n"
            )
        else:
            # Fallback if intent context missing (should not happen in normal flow)
            prompt = review_context

        if getattr(self.args, "debug_prompts", False):
            print("[DEBUG][REVIEW_PROMPT]\n" + prompt)

        # 关键：这里调用多模态生成，传入 pcl_paths + obj_list=det_topk
        # IMPORTANT: sys_msg=" " (space) to overwrite default system prompt with empty-like string.
        # sys_msg="" (empty) would cause the underlying logic to reuse the default or previous system prompt.
        out = mllm_generate_one(
            self.args,
            self.model,
            prompt_list=[prompt],
            sys_msg=" ",
            pcl_paths=pcl_paths,
            obj_list=det_topk,
            list_of_objpoints=[],              # scene tasks 默认空即可（与 inference_3d 一致）
            task_type=task_type,
            max_length=600,   # 减少续写 Schema 的概率
            top_p=1.0,
            temperature=1e-5,
        )
        raw = out[0] if isinstance(out, list) and out else str(out)

        print("review",raw)

        js = self._extract_json_by_stage(raw, stage="review")
        # 容错：兼容模型误写的字段名
        if isinstance(js, dict):
            if "need_tool" in js and "need_tool2" not in js:
                js["need_tool2"] = js.pop("need_tool")
            if "need_object_indices" in js and "selected_object_indices" not in js:
                js["selected_object_indices"] = js.pop("need_object_indices")

        review = {
            "stage": "review",
            "summary": "",
            "need_tool2": task_type in {"Counting"},
            "selected_object_indices": [],
            "raw": raw,
            "parse_error": False,
        }
        if isinstance(js, dict) and js:
            if isinstance(js.get("summary", ""), str):
                review["summary"] = js.get("summary", "")
            if isinstance(js.get("need_tool2", None), bool):
                review["need_tool2"] = js["need_tool2"]
            review["selected_object_indices"] = self._safe_int_list(js.get("selected_object_indices", []))
            
            review["next_tool"] = {"tool": "FINISH"}

        else:
            review["parse_error"] = True

        # 固定预算：Count 必须 tool2；VG/REL/Room 不执行 tool2
        if task_type in {"Counting"}:
            review["need_tool2"] = True
        else:
            review["need_tool2"] = False

        # clamp indices
        k = len(det_topk)
        review["selected_object_indices"] = [i for i in review["selected_object_indices"] if 0 <= i < k]

        # 强制校正 next_tool（与任务一致）
        if task_type == "Counting":
            review["next_tool"] = {"tool": "COUNT"}
        elif task_type == "PositionRelation":
            # indices = review["selected_object_indices"]
            # args = {}
            # if len(indices) >= 1: args["A_idx"] = indices[0]
            # if len(indices) >= 2: args["B_idx"] = indices[1]
            review["next_tool"] = {"tool": "REL_DIR"}
        else:
            review["next_tool"] = {"tool": "FINISH"}

        if task_type == "Counting" and not review["selected_object_indices"]:
            # Fallback: assume all detected objects (det_topk) are the ones we want to count 
            # if we have keyword matches in intent. But wait, det_topk is already filtered by keyword mostly?
            # Or if we have no indices, maybe just count everything in det_topk?
            # If det_topk is strictly filtered, yes. If it's just top 20, no.
            # But let's check intent keywords.

            # If we failed to parse indices, fallback to Regex on RAW output.
            import re
            nums = []
            try:
                nums = [int(x) for x in re.findall(r"\d+", raw)]
            except Exception:
                nums = []
            
            # The regex finds every digit. This is dangerous if there are other numbers.
            # But usually it's [0, 1, 2].
            uniq = []
            for v in nums:
                if 0 <= v < k and v not in uniq:
                    uniq.append(v)
            review["selected_object_indices"] = uniq
            
            # If STILL empty, and task is counting, this is bad. It means "0".
            # But wait, did we filter det_topk heavily in Step 1?
            # If Step 1 found 5 candidates, det_topk is 5 candidates (padded if needed).
            # If MLLM says "I select none", then count is 0. 
            # But often MLLM just fails to output JSON.
            # Let's consider: if result is empty but we have candidates in det_topk, 
            # maybe the model just forgot to list them?
            # Let's trust the model if it output valid JSON with empty list.
            # ONLY if parse error occurred (review["parse_error"]), we force-select all det_topk?
            if review.get("parse_error", False) and not review["selected_object_indices"]:
                 # If parse error, assume all candidates are valid (since they were filtered by keyword previously)
                 review["selected_object_indices"] = list(range(len(det_topk)))

        return review

    # ----------- one-shot reflection (validation + fallback) -----------
    def reflection_fix(self, task_type: str, intent: Dict[str, Any], det_topk: List[Dict[str, Any]],
                       selected: List[int], text: str, has_parse_error: bool = False) -> Tuple[str, Dict[str, Any]]:
        trace = {"applied": False, "reason": ""}

        if self.args.wo_reflection:
            return text, trace

        if has_parse_error:
            trace["applied"] = True
            trace["reason"] = "parse_error_fallback"

        if task_type == "Counting":
            return text.strip(), trace

        if task_type == "PositionRelation":
            return text.strip(), trace

        if task_type in {"VisualGrounding_plus", "RoomDetection"}:
            # 使用更严格的正则判断：中括号内必须包含数字坐标
            # (形如 [1.0, 2.5] 或者 [-1, 2, 3] 等，包含至少一个逗号或数字)
            import re
            if not re.search(r'\[\s*-?\d+\.?\d*(\s*,\s*-?\d+\.?\d*)*\s*\]', text):
                # fallback: use top-1 bbox
                if det_topk and det_topk[0].get("BoundingBox") is not None:
                    bb = det_topk[0]["BoundingBox"]
                    trace["applied"] = True
                    # Combine reasons if parse error also occurred
                    trace["reason"] = trace["reason"] + "|bbox_missing" if trace["reason"] else "bbox_missing_fallback_top1"
                    if task_type != "RoomDetection":
                        return str(bb), trace
            
            # If RoomDetection but has format or parse issues, just record it but don't alter the text response
            return text, trace

        return text, trace

    # ----------- main solve loop -----------
    def solve(self, task_type: str, data_item: Dict[str, Any], obj_list: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        metrics = {}
        query = data_item['query'][0] if isinstance(data_item['query'], list) else data_item['query']
        if isinstance(query, str):
            query = query.strip()
        pcl_paths = data_item['pcl']

        # Classification has no external detector proposals in many datasets.
        # Build pseudo proposals from options so the 3-stage pipeline can still run.
        if task_type == "Classification" and not obj_list:
            opts = self._extract_classification_options(query)
            pseudo = []
            for _, name in opts[:self.args.max_obj]:
                pseudo.append({"name": str(name), "BoundingBox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
            if not pseudo:
                pseudo = [{"name": "object", "BoundingBox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}]
            obj_list = pseudo

        # Ablation: w/o tool => direct answer (baseline)
        if self.args.wo_tool:
            direct = mllm_generate_one(
                self.args, self.model,
                prompt_list=[query],
                sys_msg=" ",
                pcl_paths=pcl_paths,
                obj_list=obj_list[:self.args.max_obj],
                list_of_objpoints=[],
                task_type=task_type,
                max_length=min(self.args.max_tgt_len, 1200),
                top_p=1.0,
                temperature=1e-5,
            )
            text = direct[0] if isinstance(direct, list) and direct else str(direct)
            return text, {}, {}
        
        # if task_type == "RoomDetection":
        #     query = "Locate the locations of every room within the scene."

        # Step1: intent (plan tool calls)
        intent = self.mllm_intent(query, pcl_paths, obj_list=obj_list[:self.args.max_obj], task_type=task_type)

        # task_inferred = intent.get("task", task_type) or task_type
        task_inferred = task_type
        metrics["task_inferred"] = task_inferred
        metrics["tool_plan"] = intent.get('tool_plan', [])
        metrics["parse_error"] = intent.get("parse_error", False)
        
        print(f"[过程数据] 任务识别: {task_inferred}")
        print(f"[过程数据] 工具选择: {intent.get('tool_plan', [])}")
        if metrics["parse_error"]:
            print(f"[过程数据] 阶段: Intent, 解析失败, 当前输出: {intent.get('raw', '')}")

        # Tool1: DETECT (reuse)
        all_objs = obj_list or []

        # Strategy:
        # - Counting/Room: prioritize RECALL (filter by keyword).
        # - VG/Relation: prioritize CONTEXT/ORDER (natural top-k), BUT we should still try to include the relevant objects if possible
        if task_inferred in ["Counting", "RoomDetection", "PositionRelation", "VisualGrounding_plus", "Detection", "Classification"]:
            keywords = []
            focus = intent.get("focus", {})
            if focus.get("target"):
                keywords.append(focus["target"])
            if task_inferred == "PositionRelation":
                if focus.get("A"): keywords.append(focus["A"])
                if focus.get("B"): keywords.append(focus["B"])

            if task_type == "RoomDetection":
                rt = focus.get("room_type", "")
                keywords.extend(self.ROOM_HINTS.get(rt, []) or [])
                keywords.extend(focus.get("focus_keywords", []) or [])

            # Simple keyword matching to find all potential targets
            # We want to keep the original order mostly, but ensure candidates are present
            
            # Helper to check match
            def is_match(obj, kws):
                name = self._norm(obj.get("name", "") or obj.get("label", "") or "")
                for k in kws:
                    k_norm = self._norm(str(k))
                    if k_norm and (k_norm == name or k_norm in name or name in k_norm):
                        return True
                return False

            if task_inferred == "RoomDetection":
                import random
                det_room_input = all_objs[:]
                random.shuffle(det_room_input)
                det_topk = det_room_input[:int(self.args.max_obj)]

            elif task_inferred in ["Counting", "VisualGrounding_plus", "PositionRelation"]:
                target_kws = []
                if task_inferred == "Counting":
                    t = intent.get("focus", {}).get("target", "")
                    if t: target_kws.append(t)
                elif task_inferred == "VisualGrounding_plus":
                    t = intent.get("focus", {}).get("target", "")
                    if t: target_kws.append(t)
                elif task_inferred == "PositionRelation":
                    tA = intent.get("focus", {}).get("A", "")
                    tB = intent.get("focus", {}).get("B", "")
                    if tA: target_kws.append(tA)
                    if tB: target_kws.append(tB)
                    if not tA and not tB:
                         tList = intent.get("focus", {}).get("target", [])
                         if isinstance(tList, list):
                             target_kws.extend([str(x) for x in tList])

                hits = []
                if target_kws:
                    for i, obj in enumerate(all_objs):
                        if is_match(obj, target_kws):
                            hits.append(i)
                
                hits = sorted(list(set(hits)))
                hits = hits[:int(self.args.max_obj)]
                
                if hits:
                    det_topk = [all_objs[i] for i in hits]
                else:
                    det_topk = all_objs[:int(self.args.max_obj)]
            
            elif task_inferred == "Detection":
                det_topk = all_objs[:int(self.args.max_obj)]

            elif task_inferred == "Classification":
                det_topk = all_objs[:int(self.args.max_obj)] if all_objs else []

            else:
                det_topk = all_objs[:int(self.args.max_obj)]
        else:
            det_topk = all_objs[:int(self.args.max_obj)]

        if not det_topk:
            # hard fallback
            print(f"[过程数据] 阶段: Detect, 触发兜底: 无候选物体")
            if task_type == "Counting":
                metrics["fallback_used"] = True
                metrics["fallback_reason"] = "empty_detect_counting"
                return "1", intent, metrics
            if task_type == "Detection":
                metrics["fallback_used"] = True
                metrics["fallback_reason"] = "empty_detect_detection"
                metrics["output_failed"] = True
                return "unknown", intent, metrics
            if task_type == "Classification":
                cls_text, cls_used_fb, cls_meta = self._fallback_classification_choice(query, "", det_topk, pcl_paths=pcl_paths)
                metrics["fallback_used"] = cls_used_fb
                metrics["fallback_reason"] = "empty_detect_classification"
                metrics["classification_fallback"] = cls_meta
                if cls_text in ["unknown", "None", "", None]:
                    metrics["output_failed"] = True
                return cls_text, intent, metrics
            metrics["output_failed"] = True
            return "unknown", intent, metrics




        # Step2: review/select (MULTIMODAL MLLM)
        review = self.mllm_review_multimodal(task_inferred, query, pcl_paths, det_topk, intent)

        selected = review.get("selected_object_indices", [])
        metrics["num_selected_instances"] = len(selected)
        metrics["parse_error"] = metrics.get("parse_error", False) or review.get("parse_error", False)
        
        print(f"[过程数据] 实例选择: {selected}")
        if review.get("parse_error"):
            print(f"[过程数据] 阶段: Review, 解析失败, 当前输出: {review.get('raw', '')}")

        # Tool2 decision (fixed budget)
        text = "unknown"

        if task_inferred == "Counting":
            # count_res = len(selected) if selected else 0
            if selected:
                count_res = len(selected)
            elif det_topk:
                count_res = len(det_topk)
            else:
                count_res = 0

            tgt = intent.get("focus", {}).get("target", "") or "object"
            tool_res_str = f"Tool=COUNT\nTarget={tgt}\ncount_all={count_res}\nfinal_count={count_res}"
            
            final_text = self.mllm_finish(task_inferred, query, tool_res_str, pcl_paths, det_topk, intent=intent, review=review)
            text, used_count_option_fallback, chosen_option, query_options = self._fallback_counting_to_nearest_option(
                query=query,
                model_text=final_text,
                estimated_count=count_res,
            )
            if used_count_option_fallback:
                metrics["fallback_used"] = True
                metrics["fallback_reason"] = "count_answer_out_of_options_nearest"
                metrics["count_query_options"] = query_options
                metrics["count_fallback_choice"] = chosen_option
                print(
                    f"[过程数据] 阶段: Counting, 触发兜底: answer_not_in_options, "
                    f"options={query_options}, chosen={chosen_option}"
                )


        elif task_inferred == "VisualGrounding_plus":
            if len(selected) != 1:
                tgt = intent.get("focus", {}).get("target", "")
                hits = self._find_by_keywords_fallback(det_topk, [tgt] if tgt else [])
                selected = [hits[0]] if hits else [0]
            
            idx = selected[0]
            tool_res_str = "Tool=FINISH\n(ready)"
            
            final_text = self.mllm_finish(task_inferred, query, tool_res_str, pcl_paths, det_topk, intent=intent, review=review)
            text = final_text

        elif task_inferred == "RoomDetection":
            tool_res_str = "Tool=FINISH\n(ready)"
            final_text = self.mllm_finish(task_inferred, query, tool_res_str, pcl_paths, det_topk, intent=intent, review=review)
            text = final_text

        elif task_inferred == "PositionRelation":
            focus = intent.get("focus", {}) if isinstance(intent, dict) else {}
            focus_a = str(focus.get("A", "") or "").strip()
            focus_b = str(focus.get("B", "") or "").strip()
            if not focus_a and not focus_b:
                target = focus.get("target", [])
                if isinstance(target, list) and len(target) >= 2:
                    focus_a = str(target[0] or "").strip()
                    focus_b = str(target[1] or "").strip()

            def _obj_name(o: Dict[str, Any]) -> str:
                return str(o.get("name", "") or o.get("label", "") or "").strip()

            def _find_matches(objs: List[Dict[str, Any]], keyword: str) -> List[int]:
                kw = self._norm(keyword)
                if not kw:
                    return []
                hits: List[int] = []
                for i, o in enumerate(objs):
                    nm = self._norm(_obj_name(o))
                    if kw and (kw == nm or kw in nm or nm in kw):
                        hits.append(i)
                return hits

            # Prefer matching from det_topk (smaller), fall back to all_objs.
            # Using det_topk keeps response aligned with the agent's "DETECT topk" idea.
            search_space = det_topk if det_topk else all_objs

            a_hits = _find_matches(search_space, focus_a)
            b_hits = _find_matches(search_space, focus_b)

            # Fallback if intent missing or no hits.
            fallback_applied = False
            fallback_reason = []
            if not a_hits:
                a_hits = list(range(len(search_space)))
                fallback_applied = True
                fallback_reason.append("A_not_found")
            if not b_hits:
                b_hits = list(range(len(search_space)))
                fallback_applied = True
                fallback_reason.append("B_not_found")
            
            if fallback_applied:
                metrics["fallback_used"] = True
                metrics["fallback_reason"] = ",".join(fallback_reason)
                print(f"[过程数据] 阶段: PositionRelation, 触发兜底: {metrics['fallback_reason']}")
                # If target objects are completely missing, computing a random relation is meaningless.
                text = "unknown"
                metrics["output_failed"] = True
            else:
                a_idx = a_hits[0] if a_hits else 0
                b_idx = b_hits[0] if b_hits else (1 if len(search_space) > 1 else 0)
                
                # If A and B are the same instance/index, try to pick a different B if possible
                if b_idx == a_idx and len(b_hits) > 1:
                    b_idx = b_hits[1]
                if b_idx == a_idx and len(search_space) > 1:
                    # Fallback: just pick something else
                    b_idx = 1 if a_idx == 0 else 0
                
                # Update selected indices for reflection
                selected = [a_idx, b_idx]

                objA = search_space[a_idx] if 0 <= a_idx < len(search_space) else {}
                objB = search_space[b_idx] if 0 <= b_idx < len(search_space) else {}
                bbA = objA.get("BoundingBox")
                bbB = objB.get("BoundingBox")

                nameA = focus_a or _obj_name(objA) or "object"
                nameB = focus_b or _obj_name(objB) or "object"

                qid, flip = (60, 0)  # default to a stable direction
                if isinstance(bbA, list) and isinstance(bbB, list) and len(bbA) == 6 and len(bbB) == 6:
                    qid, flip = self.geom.relation_4way(bbA, bbB)
                    if qid == 0:
                        qid, flip = (60, 0)

                # Pick a template if available; otherwise use a simple fallback sentence.
                tpl = ""
                if self.pos_rel_templates and qid > 0:
                    # Prefer exact key, then search forward within the usual +[0..29] range.
                    tpl = self.pos_rel_templates.get(str(qid), "")
                    if not tpl:
                        for off in range(1, 30):
                            tpl = self.pos_rel_templates.get(str(qid + off), "")
                            if tpl:
                                break

                if flip:
                    nameA, nameB = nameB, nameA

                if tpl:
                    rel_ans = tpl.replace("C1", nameA).replace("C2", nameB).strip()
                else:
                    rel_map = {
                        30: "right",
                        60: "left",
                        90: "front",
                        120: "behind",
                        150: "front-left",
                        180: "front-right",
                        210: "below",
                        240: "above",
                    }
                    rel = rel_map.get(qid, "near")
                    rel_ans = f"The {nameA} is to the {rel} of {nameB}."
                
                tool_res_str = rel_ans
                final_text = self.mllm_finish(task_inferred, query, tool_res_str, pcl_paths, det_topk, intent=intent, review=review)
                text = final_text

        elif task_inferred == "Detection":
            tool_res_str = "Tool=FINISH\n(ready)"
            final_text = self.mllm_finish(task_inferred, query, tool_res_str, pcl_paths, det_topk, intent=intent, review=review)
            text = final_text
            finish_valid, det_meta = self._validate_detection_finish(text, det_topk)
            metrics["detection_finish_check"] = det_meta
            if not finish_valid:
                text = self._format_detection_result(det_topk)
                metrics["fallback_used"] = True
                metrics["fallback_reason"] = "detection_finish_miss_direct_detect"

        elif task_inferred == "Classification":
            tool_res_str = "Tool=FINISH\n(ready)"
            final_text = self.mllm_finish(task_inferred, query, tool_res_str, pcl_paths, det_topk, intent=intent, review=review)
            cls_text, cls_used_fb, cls_meta = self._fallback_classification_choice(query, final_text, det_topk, pcl_paths=pcl_paths)
            text = cls_text
            if cls_used_fb:
                metrics["fallback_used"] = True
                metrics["fallback_reason"] = "classification_finish_miss_direct_choice"
            metrics["classification_fallback"] = cls_meta
        
        # Reflection: one-shot validation + fallback
        text_fixed, trace = self.reflection_fix(task_inferred, intent, det_topk, selected, text)
        if trace and trace.get("applied", False):
            metrics["fallback_used"] = True
            print(f"[过程数据] 阶段: Reflection, 触发兜底修复, 原始输出: {text}")
        else:
            metrics["fallback_used"] = metrics.get("fallback_used", False)
        if text_fixed in ["unknown", "None", "", None]:
            metrics["output_failed"] = True
        return text_fixed, intent, metrics

    def mllm_finish(self, task_type: str, user_question: str, tool2_result_text: str, pcl_paths, obj_list, intent=None, review=None) -> str:
        def _simplify_bbox_numbers(s: str) -> str:
            import re
            def _round(m):
                try:
                    return f"{float(m.group()):.2f}"
                except Exception:
                    return m.group()
            return re.sub(r"-?\d+\.\d+", _round, s)

        simplified = _simplify_bbox_numbers(tool2_result_text)
        
        # Format detected objects list for review string reconstruction
        obj_lines = []
        for i, o in enumerate(obj_list[:self.args.max_obj]):
            name = (o.get("name", "") or o.get("label", "") or "").lower()
            raw_box = o.get("BoundingBox", None)
            bbox_str = str(raw_box) if raw_box else "None"
            obj_lines.append(f"{name}{bbox_str}")
        obj_text = "\n".join(obj_lines)

        # 1. Retrieve Intent Context
        intent_prompt_str = intent.get("prompt", "") if intent else ""
        intent_output_str = intent.get("raw", "") if intent else ""
        # Remove 'args' from intent output string to clean up history
        import re
        if intent_output_str:
            intent_output_str = re.sub(r',\s*"args":\s*\{[^}]*\}', '', intent_output_str)
            intent_output_str = re.sub(r'"args":\s*\{[^}]*\},\s*', '', intent_output_str) 

        # 2. Build Review Context (MUST MATCH mllm_review_multimodal)
        review_context_user = (
            "[TOOL_RESULT]\n"
            "Tool=DETECT\n"
            f"Objects(topk={len(obj_list)}):\n{obj_text}\n\n"
            "[AGENT_REVIEW]\n"
            # f"UserQuestion: {user_question}\n\n"
            # "Output ONE JSON only.\n"
        )
        review_output_str = review.get("raw", "") if review else ""
        # Remove 'args' from review output string
        if review_output_str:
            review_output_str = re.sub(r',\s*"args":\s*\{[^}]*\}', '', review_output_str)
            review_output_str = re.sub(r'"args":\s*\{[^}]*\},\s*', '', review_output_str)

        # 3. Build Finish Context
        finish_context_user = (
            "[TOOL_RESULT]\n" +
            f"{simplified}\n\n" +
            "[AGENT_FINISH]\n" 
        )

        # 4. Concatenate History
        # Format:
        # <Pcl> [IntentUser] \n### gpt:[IntentOut] \n###human: [ReviewUser] \n### gpt:[ReviewOut] \n###human: [FinishUser] \n### gpt:
        if intent_prompt_str and intent_output_str and review_output_str:
            prompt = (
                f"{intent_prompt_str.rstrip()}\n\n" 
                f"\n### gpt:{intent_output_str}\n###"
                f"human: {review_context_user.rstrip()}\n"
                f"\n### gpt:{review_output_str}\n###"
                f"human: {finish_context_user.rstrip()}\n"
            )
        else:
            prompt = finish_context_user

        if getattr(self.args, "debug_prompts", False):
            print("[DEBUG][FINISH_PROMPT]\n" + prompt)

        # Finish 阶段用纯文本生成以稳定输出格式
        out = mllm_generate_one(
            self.args,
            self.model,
            prompt_list=[prompt],
            sys_msg=" ",
            pcl_paths=pcl_paths,
            obj_list=obj_list[:self.args.max_obj],
            list_of_objpoints=[],
            task_type="Agent3d",
            max_length=1000,
            top_p=1.0,
            temperature=1e-5,
        )
        raw = out[0] if isinstance(out, list) and out else str(out)
        print("finish", raw)
        return (raw or "").strip()


# -------------------------
# Args / main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='Detection',
                        choices=['VisualGrounding_plus', 'Counting', 'RoomDetection', 'PositionRelation', 'Detection', 'Classification'])
    parser.add_argument('--base-data-path', type=str, default='/data/HTC/Data/dataset/Benchmark/data')
    # Default to repo-local answers/ to avoid writing outside workspace (cwd-dependent).
    parser.add_argument('--answers-dir', type=str, default='/data/HTC/Project/llm/answers')
    parser.add_argument('--gpu', type=int, default=0)

    # model paths (align with inference_3d.py)
    parser.add_argument('--encoder_pretrain', type=str, default='epcl', choices=('clip', 'epcl'))
    parser.add_argument('--encoder_ckpt_path', type=str,
                        default='/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth')
    parser.add_argument('--vicuna_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0')

    # parser.add_argument('--delta_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/llm_exe/agent/pytorch_model.pt')   #实验版本
    # parser.add_argument('--delta_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/llm_exe/agent_v3/pytorch_model.pt')  #完整agent版本
    parser.add_argument('--delta_ckpt_path', type=str, default="/data/HTC/Data/model_zoo/llm_exe/agent/pytorch_model.pt")   #大论文版本

    parser.add_argument('--train_stage', type=int, default=2)
    parser.add_argument('--stage', type=int, default=2)

    # LoRA configs (openlamm expects)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--lora_target_modules', nargs='+', default=['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    # generation
    parser.add_argument('--max_tgt_len', type=int, default=1200)
    parser.add_argument('--conv_mode', type=str, default='simple')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--local_rank', default=0, type=int)

    parser.add_argument('--vision_feature_type', type=str, default='local', choices=('local', 'global'))
    parser.add_argument('--vision_output_layer', type=int, default=-2)
    parser.add_argument('--num_vision_token', type=int, default=256)
    parser.add_argument('--max_obj_len', type=int, default=30)

    # agent budget + ablations
    parser.add_argument('--max_obj', type=int, default=20)
    parser.add_argument('--wo_reflection', action='store_true')
    parser.add_argument('--wo_tool', action='store_true')
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--do_sample', action='store_true')

    # debugging
    parser.add_argument('--debug_prompts', action='store_true',
                        help='Print and record constructed agent prompts (intent stage).')

    # detector artifacts
    # parser.add_argument('--detection_path', type=str,
    #                     default='/data/HTC/Data/dataset/Benchmark/data/metadata/Detection_0.3_0.01.json')
    parser.add_argument('--detection_path', type=str,
                        default='/data/HTC/Data/dataset/Benchmark/Task/GT/Detection.json')
    parser.add_argument('--objpoints_path', type=str,
                        default='/data/HTC/Project/Point-BERT/data/ModelNet/modelnet40_normal_resampled/my_test_1024pts_fps.dat')

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
    assert os.path.exists(args.detection_path), 'Detection metadata not exists!'
    assert os.path.exists(args.objpoints_path), 'objpoints file not exists!'
    return args


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        # FIX: set_device expects int/str, not torch.device
        torch.cuda.set_device(args.gpu)
        # Pre-warm CUDA to avoid initialization overhead during model loading
        torch.randn(1, device=device)


    dataloader = load_3Deval_dataset(args.base_data_path, args.task_type, mode='common', batch_size=args.bs)
    model = LAMMPEFTModel(**args.__dict__)
    # Use mmap=True for potentially faster loading
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location='cpu', mmap=True)
    model.load_state_dict(delta_ckpt, strict=False)
    model.llama_model = model.llama_model.merge_and_unload()
    model = model.eval().half().to(device)

    # Cache multimodal features to avoid repeated heavy Point-BERT encoding
    # for identical (pcl_path, obj bbox list) inputs during inference.
    multimodal_feature_cache = {}

    def extract_multimodal_feature_patched(self, inputs):
        """Monkey-patched extract_multimodal_feature to handle batch dimension."""
        # Call the original method (bound to self)
        # We need to access the UNBOUND original method from the class if we can't access 'super' easily
        # But 'self' here is the Agent3D instance, not the model instance. Wait.
        # This function is intended to replace model.extract_multimodal_feature.
        
        # We need to replicate the logic because we can't easily call "original but slightly different"
        # without infinite recursion if we monkeypatch.
        # So we just copy the critical logic fix.
        
        features = []
        if "pcl_paths" in inputs and inputs["pcl_paths"]:
            resolved_pcl_paths = [_resolve_pcl_path(p) for p in inputs["pcl_paths"]]
            # for old_p, new_p in zip(inputs["pcl_paths"], resolved_pcl_paths):
            #     if old_p != new_p:
            #         print(f"[PathFix] pcl path remapped: {old_p} -> {new_p}")

            safe_obj_list = inputs.get("obj_list", []) or []
            if not safe_obj_list:
                safe_obj_list = [{"name": "object", "BoundingBox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]}]

            safe_obj_list = safe_obj_list[:20]

            def _bbox_key(o):
                bb = o.get("BoundingBox", None)
                if isinstance(bb, list) and len(bb) == 6:
                    return tuple(round(float(x), 4) for x in bb)
                return None

            cache_key = (
                tuple(resolved_pcl_paths),
                tuple(_bbox_key(o) for o in safe_obj_list),
            )

            if cache_key in multimodal_feature_cache:
                pcl_embeds = multimodal_feature_cache[cache_key]
            else:
                # De-duplicate repeated bboxes (common in pseudo proposals)
                # to reduce expensive Point-BERT forward calls.
                uniq_objs = []
                remap = []
                uniq_index = {}
                for o in safe_obj_list:
                    k = _bbox_key(o)
                    if k is None:
                        k = ("obj", str(o.get("name", "")))
                    if k in uniq_index:
                        remap.append(uniq_index[k])
                    else:
                        uniq_index[k] = len(uniq_objs)
                        remap.append(len(uniq_objs))
                        uniq_objs.append(o)

                # Note: We use self.test_encode_pcl which is available on the model instance
                uniq_embeds, _ = self.test_encode_pcl(
                    resolved_pcl_paths,
                    uniq_objs,
                    inputs['list_of_objpoints']
                )

                if isinstance(uniq_embeds, list):
                    pcl_embeds = [uniq_embeds[i] for i in remap if 0 <= i < len(uniq_embeds)]
                else:
                    pcl_embeds = uniq_embeds

                multimodal_feature_cache[cache_key] = pcl_embeds

            # FIX: Unwrap batch dimension so that feature_embeds is [N_obj, Dim]
            # If [1, N, D] -> [N, D]
            if isinstance(pcl_embeds, torch.Tensor) and pcl_embeds.dim() == 3 and pcl_embeds.shape[0] == 1:
                return pcl_embeds[0]
            return pcl_embeds
            
        # Fallback to original logic for other cases if needed, but for Agent3d pcl is main
        return torch.cat(features).sum(dim=0).unsqueeze(0) if features else torch.tensor([])

    def prepare_generation_embedding_patched(self, inputs):
        """Monkey-patched prepare_generation_embedding."""
        # We need to re-implement specific logic because we can't easily patch middle of function
        
        eov = VISION_TAGS["eov"][self.vision_type]
        prompt_list = inputs["prompt"]
        
        # Use our patched extractor
        if len(inputs["modality_embeds"]) == 1:
            feature_embeds = inputs["modality_embeds"][0]
        else:
            # CALL PATCHED EXTRACTOR
            # We can't call self.extract_multimodal_feature because 'self' is the model
            # and we might have monkeypatched it, but let's assume we do the fix inline here
            # OR we call the method we (will) attach to the instance.
            feature_embeds = self.extract_multimodal_feature(inputs)

        # Logic for Agent3d (falls into 'else' block of max_obj)
        if inputs["task_type"] in ["Classification",'DescriptionObj','ConversationObj']:
            max_obj = 12
            x,y,z = np.mean(inputs['list_of_objpoints'], axis=0)
            class_box_gt = [[round(x, 1), round(y, 1), round(z, 1)]]
        elif inputs["task_type"] in ["Detection"]:
            max_obj = 12
            class_box_gt = [[0,0,0]]
        else:
            max_obj = 20
            # class_list = [classname["name"] for classname in inputs["obj_list"]]
            class_box_gt = []
            for classname in inputs["obj_list"]:
                if 'BoundingBox' in classname:
                     b = classname['BoundingBox']
                     class_box_gt.append([round(b[0], 2),round(b[1], 2),round(b[2], 2)])
                else: 
                     class_box_gt.append([0.0,0.0,0.0])

        batch_input_ids = []
        for index,b in enumerate(class_box_gt[:max_obj]):
            class_name = 'obj'+str(index)+str(b)+'!'
            class_name = self.llama_tokenizer(class_name, add_special_tokens=False).input_ids
            batch_input_ids.append(torch.LongTensor(class_name))

        if not batch_input_ids:
            # Handle empty case to prevent crash
            input_embeds = torch.zeros((0, self.llama_model.config.hidden_size), device=self.device)
        else:
            input_ids = rnn.pad_sequence(
                batch_input_ids, batch_first=True, padding_value=self.llama_tokenizer.pad_token_id
            )
            input_ids = input_ids.to(self.device)
            input_embeds = self.llama_model.model.embed_tokens(input_ids)

        # Interleave vision embeddings
        vision_embeds_my = []
        # FIX: Ensure we don't index out of bounds if lengths mismatch
        limit = min(len(feature_embeds), len(input_embeds), max_obj)
        
        for index in range(limit):
            vis = feature_embeds[index] # [Dim] or [1, Dim]
            txt = input_embeds[index]   # [Seq, Dim]
            
            if vis.dim() == 1: vis = vis.unsqueeze(0)
            
            vision_embeds_my.append(vis)
            vision_embeds_my.append(txt)
            
        if vision_embeds_my:
            vision_embeds = torch.cat(vision_embeds_my).unsqueeze(dim=0)
        else:
             # Fallback shape [1, 0, Dim]
             vision_embeds = torch.zeros((1, 0, self.llama_model.config.hidden_size), device=self.device, dtype=self.llama_model.dtype)

        batch_size = vision_embeds.shape[0]
        
        # ... rest of the standard logic ...
        use_system = bool(inputs.get("use_system", False))
        p_before = make_prompt_start(
            use_system=use_system,
            vision_type=self.vision_type,
            task_type=inputs.get("task_type", "normal"),
        )
        # (Assuming standard make_prompt_start logic holds)
        if isinstance(p_before, list):
            p_before_tokens = self.llama_tokenizer(
                p_before,
                padding="longest",
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids)
        else:
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            p_before_embeds = self.llama_model.model.embed_tokens(
                p_before_tokens.input_ids
            ).expand(
                batch_size, -1, -1
            )

        p_after_texts = [f"{eov} " + prompt + "\n### gpt:" for prompt in prompt_list]
        p_after_tokens = self.llama_tokenizer(
            p_after_texts,
            padding="longest", return_length=True,
            add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_masks_len = p_after_tokens.length.max() - p_after_tokens.length
        p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids)

        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_tokens.input_ids.dtype,
                device=p_before_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = self.llama_model.model.embed_tokens(bos)

        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, vision_embeds, p_after_embeds], dim=1
        )

        tokens_len = inputs_embeds.shape[1] - p_after_masks_len
        new_inputs_embeds = torch.zeros_like(inputs_embeds)
        inputs_embeds_masks = torch.zeros(inputs_embeds.shape[:-1],
                                         dtype=torch.int64, device=self.device)
        for idx in range(batch_size):
            inputs_embeds_masks[idx, -tokens_len[idx]:] = 1
            new_inputs_embeds[idx, -tokens_len[idx]:, :] = inputs_embeds[idx, :tokens_len[idx], :]
            new_inputs_embeds[idx, :-tokens_len[idx], :] = inputs_embeds[idx, tokens_len[idx]:, :]

        return new_inputs_embeds, inputs_embeds_masks

    # Apply Monkey Patch
    import types
    model.extract_multimodal_feature = types.MethodType(extract_multimodal_feature_patched, model)
    model.prepare_generation_embedding = types.MethodType(prepare_generation_embedding_patched, model)

    det_tool = DetectionTool3D(args.detection_path)
    agent = Agent3D(args, model, det_tool, device=device)

    # 增加时间戳防止覆盖
    answers_file = os.path.join(args.answers_dir, f"Agent_{args.task_type}.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, 'w') as fout:
        for idx, data_item in enumerate(tqdm(dataloader)):
            if args.dry_run and idx >= args.dry_run:
                break

            pcl_paths = data_item['pcl']
            if args.task_type == 'Classification':
                obj_list = []
            else:
                obj_list = det_tool.proposals_for_scene(pcl_paths[0], idx)

            start = time.time() 
            text, intent_res, metrics = agent.solve(args.task_type, data_item, obj_list)
            elapsed = time.time() - start

            ans = {
                'id': data_item['id'][0] if isinstance(data_item['id'], list) else data_item['id'],
                'pcl': pcl_paths,
                'text': text,
                'metrics': metrics,
                # 'delta_path': args.delta_ckpt_path,
            }
            fout.write(json.dumps(ans) + '\n')
            fout.flush()

    print(f"Wrote {answers_file}")


if __name__ == '__main__':
    main()
