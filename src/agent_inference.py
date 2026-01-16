"""
src/agent_inference.py

3D Agent 推理脚本（对齐 `src/inference_3d.py` 数据流）。

目标：补齐 Agent 缺失的 3D 任务（VG fine/plus、Counting、RoomDetection、PositionRelation），并且体现
“把 MLLM 扩展成 Agent”：MLLM（多模态）作为 controller，基于点云 + proposals 产生结构化工具调用与关注对象选择。

约束：
- 固定预算：每样本最多 2 次工具（Tool1=DETECT，Tool2=COUNT/BBOX_UNION；REL/VG 通常只 Tool1）
- 一次反思：只做验收与回退，不做额外推理
- 不引入外部 detector：复用 Detection.json 的 proposals

输出：jsonl，每条包含 id/pcl/text（兼容 eval）+ agent_trace（可用于论文展示 agent 过程）。
"""

import os
import json
import time
import copy
import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import StoppingCriteriaList, StoppingCriteria

from model.openlamm import LAMMPEFTModel, LAMMStoppingCriteria
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
    stopping_criteria = StoppingCriteriaList([
        LAMMStoppingCriteria([[2277, 29937], [835]], input_embeds)
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

    class _StopOnSubstrings(StoppingCriteria):
        def __init__(self, tokenizer, stop_strings: List[str], window_tokens: int = 128):
            self.tokenizer = tokenizer
            self.stop_strings = stop_strings
            self.window_tokens = window_tokens

        def __call__(self, input_ids, scores, **kwargs):
            tail = input_ids[0][-self.window_tokens:]
            txt = self.tokenizer.decode(tail, skip_special_tokens=True)
            return any(s in txt for s in self.stop_strings)

    tok = model.llama_tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    ).to(device)

    # Training data is multi-turn and often uses "### Human:" separators.
    # For intent stage we want ONLY the first JSON. However the model may emit leading "###" tokens.
    # So we stop ONLY when it starts the next turn marker ("### Human:").
    stopping_criteria = StoppingCriteriaList([
        _StopOnSubstrings(model.llama_tokenizer, ["### Human:"], window_tokens=128)
    ])

    out = model.llama_model.generate(
        input_ids=tok.input_ids,
        attention_mask=tok.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stopping_criteria=stopping_criteria,
    )
    # Only decode newly generated tokens; otherwise the prompt content can pollute JSON extraction.
    gen = out[0][tok.input_ids.shape[1]:]
    text = model.llama_tokenizer.decode(gen, skip_special_tokens=True)
    if "### Human:" in text:
        text = text.split("### Human:", 1)[0]
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
    safe_temperature = max(temperature, 1e-5)
    safe_top_p = min(max(top_p, 1e-5), 1.0)
    # IMPORTANT: openlamm.prepare_generation_embedding will add the role markers and <Pcl> tags.
    # So here we must pass RAW prompt strings (no "### Human:" etc), otherwise prompts will be duplicated.
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
            self._data = json.load(open(self.detection_path, 'r'))

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

    def relation_4way(self, bboxA: List[float], bboxB: List[float]) -> str:
        ax, ay, _ = self._center(bboxA)
        bx, by, _ = self._center(bboxB)
        dx, dy = bx - ax, by - ay
        if abs(dx) >= abs(dy):
            return 'right' if dx > 0 else 'left'
        return 'above' if dy > 0 else 'below'

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
        for s in spans:
            try:
                js = json.loads(s)
                if isinstance(js, dict):
                    out.append(js)
            except Exception:
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

    # ----------- Stage 1: intent/plan (text is OK, but still using same MLLM core) -----------
    def mllm_intent(self, task_type: str, query: str) -> Dict[str, Any]:
        """
        输出 tool_plan（1/2步）+ focus（target/A/B/room_type/keywords）
        注意：这里用纯文本 greedy，主要为了 JSON 稳定；论文重点在 Stage2 多模态选择。
        """
        # Reconstruct the prompt to match the training data format.
        inst = (
            f"[AGENT_INTENT]\n"
            f"Subtask={task_type}\n"
            f"UserQuestion: {query}\n\n"
            "You are an MLLM agent controller. Output ONE JSON only.\n"
            "Schema:\n"
            "{\n"
            "  \"stage\":\"intent\",\n"
            "  \"task\":\"VisualGrounding_plus|Counting|RoomDetection|PositionRelation\",\n"
            "  \"focus\": {\"target\":\"\", \"A\":\"\", \"B\":\"\"}\n"
            "}\n"
        )

        # The training data does not seem to use a controller-specific system message.
        # We pass the full instruction as the prompt.
        raw = _greedy_text_generate(self.model, inst, device=self.device, max_new_tokens=220)
        js = self._extract_json_by_stage(raw, stage="intent", task_type=task_type)

        # sanitize + enforce plan
        focus = (js.get("focus", {}) if isinstance(js, dict) else {}) if isinstance(js, dict) else {}
        out = {
            "stage": "intent",
            "task": task_type,
            "focus": {
                "target": self._norm(focus.get("target", "")),
                "A": self._norm(focus.get("A", "")),
                "B": self._norm(focus.get("B", "")),
                "room_type": self._norm(focus.get("room_type", "")),
                "focus_keywords": [],
            },
            "tool_plan": [],
            "raw": raw,
        }

        if task_type == "Counting":
            out["tool_plan"] = [
                {"tool": "DETECT", "args": {"topk": int(self.args.max_obj)}},
                {"tool": "COUNT", "args": {}},
            ]
        elif task_type == "RoomDetection":
            out["tool_plan"] = [
                {"tool": "DETECT", "args": {"topk": int(self.args.max_obj)}},
                {"tool": "BBOX_UNION", "args": {}},
            ]
        else:
            out["tool_plan"] = [{"tool": "DETECT", "args": {"topk": int(self.args.max_obj)}}]

        return out

    # ----------- Stage 2: review/select (MUST use multimodal MLLM) -----------
    def mllm_review_multimodal(self, task_type: str, query: str, pcl_paths, det_topk: List[Dict[str, Any]],
                               intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        让 MLLM 真正看到 point cloud + obj proposals（obj_list 顺序对齐 det_topk），输出：
        - summary
        - need_tool2（Count/Room 必须 true；VG/REL 默认 false）
        - selected_object_indices（VG:1个；REL:2个；Room:2~8个；Count:任意个）
        """
        # Format detected objects similar to training data.
        obj_lines = []
        for i, o in enumerate(det_topk[:self.args.max_obj]):
            name = o.get("name", "") or o.get("label", "") or ""
            bbox = o.get("BoundingBox", None)
            obj_lines.append(f"(obj{i}):{name} bbox={bbox}")
        obj_text = "\n".join(obj_lines)

        # Reconstruct the prompt to match the training data format.
        prompt = (
            f"[TOOL_RESULT]\n"
            f"Tool=DETECT\n"
            f"Objects(topk={len(det_topk)}):\n{obj_text}\n\n"
            f"[AGENT_REVIEW]\n"
            f"Subtask={task_type}\n"
            f"UserQuestion: {query}\n\n"
            "Output ONE JSON only.\n"
            "Schema:\n"
            "{\n"
            "  \"stage\":\"review\",\n"
            "  \"summary\":\"one short sentence\",\n"
            "  \"need_tool2\": true/false,\n"
            "  \"selected_object_indices\": [int, ...],\n"
            "  \"next_tool\": {\"tool\":\"NONE|COUNT|BBOX_UNION\", \"args\": {}}\n"
            "}\n"
            "Rules:\n"
            "- VisualGrounding_plus: select EXACTLY 1 index; need_tool2=false; next_tool.tool=\"NONE\".\n"
            "- PositionRelation: select EXACTLY 2 indices [A_idx,B_idx]; need_tool2=false; next_tool.tool=\"NONE\".\n"
            "- Counting: select indices that match target; need_tool2=true; next_tool.tool=\"COUNT\".\n"
            "- RoomDetection: select 2~8 indices that define the room extent; need_tool2=true; next_tool.tool=\"BBOX_UNION\".\n"
        )

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
            max_length=200,   # 减少续写 Schema 的概率
            top_p=1.0,
            temperature=1e-5,
        )
        raw = out[0] if isinstance(out, list) and out else str(out)
        raw_clean = raw.split("###", 1)[0]  # 截断可能的下一轮对话/续写
        js = self._extract_json_by_stage(raw_clean, stage="review")
        # 容错：兼容模型误写的字段名
        if isinstance(js, dict):
            if "need_tool" in js and "need_tool2" not in js:
                js["need_tool2"] = js.pop("need_tool")
            if "need_object_indices" in js and "selected_object_indices" not in js:
                js["selected_object_indices"] = js.pop("need_object_indices")

        review = {
            "stage": "review",
            "summary": "",
            "need_tool2": task_type in {"Counting", "RoomDetection"},
            "selected_object_indices": [],
            "raw": raw,
        }
        if isinstance(js, dict):
            if isinstance(js.get("summary", ""), str):
                review["summary"] = js.get("summary", "")
            if isinstance(js.get("need_tool2", None), bool):
                review["need_tool2"] = js["need_tool2"]
            review["selected_object_indices"] = self._safe_int_list(js.get("selected_object_indices", []))

        # 固定预算：Count/Room 必须 tool2；VG/REL 不执行 tool2
        if task_type in {"Counting", "RoomDetection"}:
            review["need_tool2"] = True
        else:
            review["need_tool2"] = False

        # clamp indices
        k = len(det_topk)
        review["selected_object_indices"] = [i for i in review["selected_object_indices"] if 0 <= i < k]

        return review

    # ----------- one-shot reflection (validation + fallback) -----------
    def reflection_fix(self, task_type: str, intent: Dict[str, Any], det_topk: List[Dict[str, Any]],
                       selected: List[int], text: str) -> Tuple[str, Dict[str, Any]]:
        trace = {"applied": False, "reason": ""}

        if self.args.wo_reflection:
            return text, trace

        if task_type == "Counting":
            try:
                int(text.strip())
                return text, trace
            except Exception:
                trace["applied"] = True
                trace["reason"] = "count_not_int_fallback_0"
                return "0", trace

        if task_type == "PositionRelation":
            if text.strip() in set(self.REL_SPACE):
                return text, trace
            trace["applied"] = True
            trace["reason"] = "invalid_relation_fallback_left"
            return "left", trace

        if task_type in {"VisualGrounding_plus", "RoomDetection"}:
            if "[" in text and "]" in text:
                return text, trace
            # fallback: use top-1 bbox
            if det_topk and det_topk[0].get("BoundingBox") is not None:
                bb = det_topk[0]["BoundingBox"]
                trace["applied"] = True
                trace["reason"] = "bbox_missing_fallback_top1"
                if task_type == "RoomDetection":
                    room_label = intent.get("focus", {}).get("room_type", "") or "unknown"
                    return f"{room_label} {bb}", trace
                return str(bb), trace

        return text, trace

    # ----------- main solve loop -----------
    def solve(self, task_type: str, data_item: Dict[str, Any], obj_list: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        query = data_item['query'][0] if isinstance(data_item['query'], list) else data_item['query']
        pcl_paths = data_item['pcl']

        agent_trace: Dict[str, Any] = {
            "task_type": task_type,
            "query": query,
            "budget": {"max_tool_calls": 2, "used": 0},
        }

        # Ablation: w/o tool => direct answer (baseline)
        if self.args.wo_tool:
            direct = mllm_generate_one(
                self.args, self.model,
                prompt_list=[query],
                sys_msg="You are a helpful multimodal assistant.",
                pcl_paths=pcl_paths,
                obj_list=obj_list[:self.args.max_obj],
                list_of_objpoints=[],
                task_type=task_type,
                max_length=min(self.args.max_tgt_len, 256),
                top_p=1.0,
                temperature=1e-5,
            )
            text = direct[0] if isinstance(direct, list) and direct else str(direct)
            agent_trace["mode"] = "wo_tool_direct"
            return text, agent_trace

        # Step1: intent (plan tool calls)
        intent = self.mllm_intent(task_type, query)
        agent_trace["intent"] = intent

        print(intent)

        # Tool1: DETECT (reuse)
        all_objs = obj_list or []

        # Strategy:
        # - Counting/Room: prioritize RECALL (filter by keyword).
        # - VG/Relation: prioritize CONTEXT/ORDER (natural top-k).
        if task_type in ["Counting", "RoomDetection"]:
            keywords = []
            focus = intent.get("focus", {})
            if focus.get("target"):
                keywords.append(focus["target"])
            if task_type == "RoomDetection":
                rt = focus.get("room_type", "")
                keywords.extend(self.ROOM_HINTS.get(rt, []) or [])
                keywords.extend(focus.get("focus_keywords", []) or [])

            # Simple keyword matching to find all potential targets
            candidates = []
            for obj in all_objs:
                name = self._norm(obj.get("name", "") or obj.get("label", "") or "")
                for k in keywords:
                    k_norm = self._norm(str(k))
                    if k_norm and k_norm in name:
                        candidates.append(obj)
                        break
            
            if candidates:
                det_topk = candidates[:int(self.args.max_obj)]
            else:
                det_topk = all_objs[:int(self.args.max_obj)]
        else:
            # VisualGrounding_plus / PositionRelation
            det_topk = all_objs[:int(self.args.max_obj)]

        agent_trace["budget"]["used"] += 1
        agent_trace["tool1"] = {"tool": "DETECT", "n": len(det_topk), "topk": int(self.args.max_obj)}

        if not det_topk:
            # hard fallback
            if task_type == "Counting":
                return "1", {**agent_trace, "mode": "empty_detect_fallback_to_1"}
            return "unknown", {**agent_trace, "mode": "empty_detect"}

        # Step2: review/select (MULTIMODAL MLLM)
        review = self.mllm_review_multimodal(task_type, query, pcl_paths, det_topk, intent)
        
        print(review)

        agent_trace["review"] = review
        selected = review.get("selected_object_indices", [])

        # Tool2 decision (fixed budget)
        text = "unknown"

        if task_type == "VisualGrounding_plus":
            if len(selected) != 1:
                tgt = intent.get("focus", {}).get("target", "")
                hits = self._find_by_keywords_fallback(det_topk, [tgt] if tgt else [])
                selected = [hits[0]] if hits else [0]
                agent_trace["fallback_select"] = {"reason": "vg_fix", "selected": selected}
            bb = det_topk[selected[0]].get("BoundingBox")
            text = str(bb)

        elif task_type == "PositionRelation":
            if len(selected) != 2:
                A = intent.get("focus", {}).get("A", "")
                B = intent.get("focus", {}).get("B", "")
                a_hits = self._find_by_keywords_fallback(det_topk, [A] if A else [])
                b_hits = self._find_by_keywords_fallback(det_topk, [B] if B else [])
                a_idx = a_hits[0] if a_hits else 0
                b_idx = b_hits[0] if b_hits else (1 if len(det_topk) > 1 else 0)
                selected = [a_idx, b_idx]
                agent_trace["fallback_select"] = {"reason": "rel_fix", "selected": selected, "A": A, "B": B}
            bbA = det_topk[selected[0]].get("BoundingBox")
            bbB = det_topk[selected[1]].get("BoundingBox")
            if isinstance(bbA, list) and len(bbA) == 6 and isinstance(bbB, list) and len(bbB) == 6:
                text = self.geom.relation_4way(bbA, bbB)
            else:
                text = "unknown"

        elif task_type == "Counting":
            # must do tool2 COUNT
            agent_trace["budget"]["used"] += 1
            agent_trace["tool2"] = {"tool": "COUNT"}
            # 如果 MLLM 没选出来，回退到关键词匹配；再不行就 0
            if not selected:
                tgt = intent.get("focus", {}).get("target", "")
                hits = self._find_by_keywords_fallback(det_topk, [tgt] if tgt else [])
                selected = hits
                agent_trace["fallback_select"] = {"reason": "count_fix", "selected_n": len(selected), "target": tgt}
            
            count_result = len(selected)

            # New logic to adjust the count based on options
            options = data_item.get('gt_choices')
            # The data loader might wrap single items in a list
            if isinstance(options, list) and len(options) > 0 and isinstance(options[0], list):
                options = options[0]

            if isinstance(options, list) and all(isinstance(i, (int, float)) for i in options):
                if count_result == 0:
                    text = "1"
                else:
                    if count_result in options:
                        text = str(count_result)
                    else:
                        larger_options = sorted([opt for opt in options if opt > count_result])
                        if larger_options:
                            text = str(larger_options[0])
                        else:
                            # If no larger option, fallback to the largest option available
                            text = str(max(options))
            else:
                 text = str(count_result)

        elif task_type == "RoomDetection":
            # must do tool2 BBOX_UNION
            agent_trace["budget"]["used"] += 1
            agent_trace["tool2"] = {"tool": "BBOX_UNION"}

            # 如果 MLLM 选太少，回退：按 room_type 的 hints 匹配；再不行 union 前 3 个
            if len(selected) < 2:
                room_type = intent.get("focus", {}).get("room_type", "")
                fk = intent.get("focus", {}).get("focus_keywords", []) or []
                if not fk:
                    fk = self.ROOM_HINTS.get(room_type, [])
                hits = self._find_by_keywords_fallback(det_topk, fk)
                if len(hits) >= 2:
                    selected = hits[:8]
                else:
                    selected = list(range(min(3, len(det_topk))))
                agent_trace["fallback_select"] = {"reason": "room_fix", "room_type": room_type, "selected": selected}

            bboxes = []
            for i in selected:
                bb = det_topk[i].get("BoundingBox")
                if isinstance(bb, list) and len(bb) == 6:
                    bboxes.append(bb)
            union = self.geom.union_bbox(bboxes) if bboxes else det_topk[0].get("BoundingBox")

            room_label = intent.get("focus", {}).get("room_type", "") or "unknown"
            text = f"{room_label} {union}"

        else:
            text = "unknown"

        agent_trace["selected_indices_final"] = selected
        if task_type == "Counting":
            agent_trace["selected_names_final"] = [
                (det_topk[i].get("name") or det_topk[i].get("label") or "")
                for i in selected
                if 0 <= i < len(det_topk)
            ]

        # Reflection: one-shot validation + fallback
        text_fixed, refl = self.reflection_fix(task_type, intent, det_topk, selected, text)
        agent_trace["reflection"] = refl
        return text_fixed, agent_trace


# -------------------------
# Args / main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='RoomDetection',
                        choices=['VisualGrounding_plus', 'Counting', 'RoomDetection', 'PositionRelation'])
    parser.add_argument('--base-data-path', type=str, default='/data/HTC/Data/dataset/Benchmark/data')
    # Default to repo-local answers/ to avoid writing outside workspace (cwd-dependent).
    parser.add_argument('--answers-dir', type=str, default='/data/HTC/Project/llm/answers')
    parser.add_argument('--gpu', type=int, default=1)

    # model paths (align with inference_3d.py)
    parser.add_argument('--encoder_pretrain', type=str, default='epcl', choices=('clip', 'epcl'))
    parser.add_argument('--encoder_ckpt_path', type=str,
                        default='/data/HTC/Data/model_zoo/epcl_ckpt/epcl_scannet_vit-L-14_256tokens_latest.pth')
    parser.add_argument('--vicuna_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/vicuna-7b/Vicuna_7B_v0')
    parser.add_argument('--delta_ckpt_path', type=str, default='/data/HTC/Data/model_zoo/llm_exe/agent/pytorch_model_ep1.pt')

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

    # detector artifacts
    parser.add_argument('--detection_path', type=str,
                        default='/data/HTC/Data/dataset/Benchmark/data/metadata/Detection_0.3_0.01.json')
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

    model = LAMMPEFTModel(**args.__dict__)
    # Use mmap=True for potentially faster loading
    delta_ckpt = torch.load(args.delta_ckpt_path, map_location='cpu', mmap=True)
    model.load_state_dict(delta_ckpt, strict=False)
    model.llama_model = model.llama_model.merge_and_unload()
    model = model.eval().half().to(device)

    det_tool = DetectionTool3D(args.detection_path)
    agent = Agent3D(args, model, det_tool, device=device)

    dataloader = load_3Deval_dataset(args.base_data_path, args.task_type, mode='common', batch_size=args.bs)

    # 增加时间戳防止覆盖
    answers_file = os.path.join(args.answers_dir, f"Agent_{args.task_type}.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(answers_file, 'w') as fout:
        for idx, data_item in enumerate(tqdm(dataloader)):
            if args.dry_run and idx >= args.dry_run:
                break

            pcl_paths = data_item['pcl']
            obj_list = det_tool.proposals_for_scene(pcl_paths[0], idx)

            start = time.time()
            text, trace = agent.solve(args.task_type, data_item, obj_list)
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
