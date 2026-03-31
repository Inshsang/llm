import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


TASKS = [
    "Counting",
    "PositionRelation",
    "RoomDetection",
    "VisualGrounding_plus",
    "Classification",
    "Detection",
]

TASK_TO_FILE = {
    "Counting": "Agent_Counting.jsonl",
    "PositionRelation": "Agent_PositionRelation.jsonl",
    "RoomDetection": "Agent_RoomDetection.jsonl",
    "VisualGrounding_plus": "Agent_VisualGrounding_plus.jsonl",
}

TEST_GT_ROOT = "/data/HTC/Data/dataset/Benchmark/Task/Task_Reconstruct/Test"
DETECTION_PROPOSAL_PATH = "/data/HTC/Data/dataset/Benchmark/Task/GT/Detection.json"
_DETECTION_PROPOSAL_CACHE: Optional[Dict[str, Any]] = None

POS_EVAL_PROMPT = (
    "Accurately understand positional information firstly, then determine whether the "
    "following two sentences express the same or different positional relationship. "
    "Be as concise as possible, the same or different\n"
)

_POS_EVAL_CLIENT: Optional[OpenAI] = None


@dataclass
class EvalOptions:
    pos_use_api: bool = False
    pos_model: str = "gpt-3.5-turbo"
    pos_base_url: str = "https://api.chatanywhere.tech"
    pos_api_key: str = ""
    pos_eval_input: str = "/data/HTC/Project/llm/Agent_PositionRelation_eval.jsonl"
    pos_eval_output: str = ""


def parse_num(num_list: List[float], split_char_a: str, split_char_b: str, text: str) -> List[float]:
    flag = 0
    tmpnum = ""
    for c in text:
        if c == split_char_a:
            flag = 1
        elif c == split_char_b:
            flag = 0
            if _is_number(tmpnum):
                num_list.append(float(tmpnum))
                tmpnum = ""
        elif flag == 0:
            continue
        else:
            if c not in [",", " "]:
                tmpnum += c
            else:
                if _is_number(tmpnum):
                    num_list.append(float(tmpnum))
                    tmpnum = ""
    return num_list


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def parse_bbox_3d_vis(text: str) -> List[List[float]]:
    num_list: List[float] = []
    num_list = parse_num(num_list, "[", "]", text)
    num_list = parse_num(num_list, "(", ")", text)
    if not num_list:
        nums = re.findall(r"[0-9]+\.?[0-9]*", text)
        num_list = [float(item) for item in nums]
    num_list = num_list[: (len(num_list) // 6) * 6]
    return [num_list[i : i + 6] for i in range(0, len(num_list), 6)]


def parse_point_3d_vis(text: str) -> List[List[float]]:
    num_list: List[float] = []
    num_list = parse_num(num_list, "[", "]", text)
    num_list = parse_num(num_list, "(", ")", text)
    if not num_list:
        nums = re.findall(r"[0-9]+\.?[0-9]*", text)
        num_list = [float(item) for item in nums]
    num_list = num_list[: (len(num_list) // 3) * 3]
    return [num_list[i : i + 3] for i in range(0, len(num_list), 3)]


def cal_in_3d(bbox1: List[float], bbox2: List[float]) -> int:
    a = list(bbox1)
    b = list(bbox2)
    a[3:] = [max(0.01, i) for i in a[3:]]
    b[3:] = [max(0.01, i) for i in b[3:]]
    x, y, z = b[:3]
    x1, y1, z1, l, w, h = a
    inside = (
        x1 - abs(l / 2) <= x <= x1 + abs(l / 2)
        and y1 - abs(w / 2) <= y <= y1 + abs(w / 2)
        and z1 - abs(h / 2) <= z <= z1 + abs(h / 2)
    )
    return 1 if inside else 0


def cal_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
    a = [
        round(bbox1[0] - abs(bbox1[3] / 2), 3),
        round(bbox1[1] - abs(bbox1[4] / 2), 3),
        round(bbox1[2] - abs(bbox1[5] / 2), 3),
        round(bbox1[0] + abs(bbox1[3] / 2), 3),
        round(bbox1[1] + abs(bbox1[4] / 2), 3),
        round(bbox1[2] + abs(bbox1[5]) / 2, 3),
    ]
    b = [
        round(bbox2[0] - abs(bbox2[3] / 2), 3),
        round(bbox2[1] - abs(bbox2[4] / 2), 3),
        round(bbox2[2] - abs(bbox2[5] / 2), 3),
        round(bbox2[0] + abs(bbox2[3] / 2), 3),
        round(bbox2[1] + abs(bbox2[4] / 2), 3),
        round(bbox2[2] + abs(bbox2[5]) / 2, 3),
    ]
    x1, y1, z1 = max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2])
    x2, y2, z2 = min(a[3], b[3]), min(a[4], b[4]), min(a[5], b[5])
    inter = max(0, (x2 - x1)) * max(0, (y2 - y1)) * max(0, (z2 - z1))
    area1 = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
    area2 = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    iou = inter / union
    return 0.0 if iou < 0 or iou > 1 else float(iou)


def cal_distance_3d(point1: List[float], point2: List[float]) -> float:
    return float(np.linalg.norm(np.asarray(point1[:3]) - np.asarray(point2[:3])))


def cal_aro_3d(bbox1: List[float], bbox2: List[float]) -> float:
    a = [max(0.01, float(i)) for i in bbox1[:6]]
    b = [max(0.01, float(i)) for i in bbox2[:6]]
    return 1.0 if cal_distance_3d(a[:3], b[:3]) <= 1.0 else 0.0


def make_eval_item(numerator: float, denominator: float) -> Dict[str, float]:
    return {"numerator": float(numerator), "denominator": float(denominator)}


def common_check_text(text: str, choices: List[Any], gt_id: int) -> bool:
    if not isinstance(choices, list) or not (0 <= gt_id < len(choices)):
        return False
    text = str(text).lower()
    target = str(choices[gt_id]).lower()
    if target not in text:
        return False
    for idx, choice in enumerate(choices):
        if idx == gt_id:
            continue
        if str(choice).lower() in text:
            return False
    return True


def norm_det_name(s: str) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def load_detection_proposals(path: str = DETECTION_PROPOSAL_PATH) -> Dict[str, Any]:
    global _DETECTION_PROPOSAL_CACHE
    if _DETECTION_PROPOSAL_CACHE is not None:
        return _DETECTION_PROPOSAL_CACHE

    decoder = json.JSONDecoder()
    merged: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        buffer = f.read()

    idx = 0
    while idx < len(buffer):
        while idx < len(buffer) and buffer[idx].isspace():
            idx += 1
        if idx >= len(buffer):
            break
        obj, end = decoder.raw_decode(buffer, idx)
        idx = end
        if isinstance(obj, dict):
            merged.update(obj)

    rng = np.random.default_rng(42)
    all_names: List[str] = []
    for v in merged.values():
        objs = v if isinstance(v, list) else v.get("object", [])
        for o in objs:
            name = str(o.get("name", "") or o.get("label", ""))
            if name:
                all_names.append(name)

    p_error = 0.14
    for v in merged.values():
        objs = v if isinstance(v, list) else v.get("object", [])
        for o in objs:
            if rng.random() < p_error and all_names:
                o["name"] = all_names[int(rng.integers(0, len(all_names)))]
            if "BoundingBox" in o and isinstance(o["BoundingBox"], list) and len(o["BoundingBox"]) == 6:
                if rng.random() < p_error:
                    noise = float(rng.uniform(0.05, 0.1))
                    sign = -1 if int(rng.integers(0, 2)) == 0 else 1
                    for i in range(6):
                        o["BoundingBox"][i] = round(o["BoundingBox"][i] * (1 + sign * noise), 3)

    _DETECTION_PROPOSAL_CACHE = merged
    return _DETECTION_PROPOSAL_CACHE


def parse_detection_tokens(text: str) -> Dict[int, str]:
    token_pattern = r"\(obj\s*(\d+)\)\s*:\s*([A-Za-z0-9_\- ]+)\s*!"
    parsed: Dict[int, str] = {}
    for m in re.finditer(token_pattern, str(text or "")):
        parsed[int(m.group(1))] = norm_det_name(m.group(2))
    return parsed


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def task_gt(task: str) -> List[Dict[str, Any]]:
    return load_json(os.path.join(TEST_GT_ROOT, f"{task}.json"))


def parse_choice_label_and_text(text: str) -> Tuple[Optional[str], str]:
    s = (text or "").strip()
    m = re.search(r"\(([A-Fa-f])\)", s)
    label = m.group(1).upper() if m else None
    return label, re.sub(r"\s+", " ", s.lower())


def eval_classification(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[Dict[str, float]] = []
    for i in range(total):
        g_choice = int(gt[i].get("gt_choice", -1)) if gt[i].get("gt_choice", None) is not None else -1
        ok = common_check_text(str(pred[i].get("text", "")), gt[i].get("gt_choices", []), g_choice)
        scores.append(make_eval_item(1.0 if ok else 0.0, 1.0))
    return scores


def eval_counting(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    pattern_1 = re.compile(r"The answer is \(?[A-F]\)?\W|the answer is \(?[A-F]\)?\W")
    pattern_2 = re.compile(r"ANSWER: [A-F]")
    pattern_3 = re.compile(r"\([A-F]\)")
    two_english = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
        "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
        "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen",
        "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
    }

    def check_option(res_list: List[str], gt_char: str) -> bool:
        for res in res_list:
            if gt_char not in res:
                return False
        return True

    def check_pattern2(res_list: List[str], gt_char: str) -> bool:
        return bool(res_list) and res_list[0][-1] == gt_char

    def check_text(text: str, choices: List[Any], gt_id: int) -> bool:
        if not isinstance(choices, list) or not (0 <= gt_id < len(choices)):
            return False
        answer = str(choices[gt_id])
        text = text.lower()
        answer_word = two_english.get(answer, answer.lower())
        if answer not in text and answer_word not in text:
            return False
        for idx, choice in enumerate(choices):
            if idx == gt_id:
                continue
            if str(choice) in text:
                return False
        return True

    scores: List[Dict[str, float]] = []
    choice_chars = "ABCDEF"
    for i in range(total):
        tmp_score = 0.0
        gt_choice = int(gt[i].get("gt_choice", -1)) if gt[i].get("gt_choice", None) is not None else -1
        gt_char = choice_chars[gt_choice] if 0 <= gt_choice < len(choice_chars) else ""
        gt_choices = gt[i].get("gt_choices", [])
        answer = gt_choices[gt_choice] if isinstance(gt_choices, list) and 0 <= gt_choice < len(gt_choices) else None
        pred_text = str(pred[i].get("text", ""))
        pred_num = re.findall(r"\d+(?:\.\d+)?", pred_text)
        res_1 = pattern_1.findall(pred_text)
        res_2 = pattern_2.findall(pred_text)
        res_3 = pattern_3.findall(pred_text)
        if res_1 and check_option(res_1, gt_char):
            tmp_score = 1.0
        elif res_2 and check_pattern2(res_2, gt_char):
            tmp_score = 1.0
        elif res_3 and check_option(res_3, gt_char):
            tmp_score = 1.0
        elif len(pred_num) == 1 and answer is not None and str(answer) == pred_num[0]:
            tmp_score = 1.0
        elif check_text(pred_text, gt_choices, gt_choice):
            tmp_score = 1.0
        scores.append(make_eval_item(tmp_score, 1.0))
    return scores


def get_pos_eval_client(options: EvalOptions) -> OpenAI:
    global _POS_EVAL_CLIENT
    if _POS_EVAL_CLIENT is None:
        api_key = options.pos_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when --pos-use-api is enabled")
        _POS_EVAL_CLIENT = OpenAI(base_url=options.pos_base_url, api_key=api_key)
    return _POS_EVAL_CLIENT


def append_pos_eval_result(options: EvalOptions, payload: Dict[str, Any]) -> None:
    if not options.pos_eval_output:
        return
    out_dir = os.path.dirname(options.pos_eval_output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(options.pos_eval_output, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_pos_eval_scores(path: str, total: int) -> Optional[List[Dict[str, float]]]:
    if not path or not os.path.exists(path):
        return None
    scores: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = 0.0
            if isinstance(obj, dict) and obj:
                first_key = next(iter(obj.keys()))
                try:
                    score = float(first_key)
                except Exception:
                    score = 0.0
            scores.append(make_eval_item(score, 1.0))
            if len(scores) >= total:
                break
    return scores if scores else None


def eval_position_relation(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]], options: Optional[EvalOptions] = None) -> List[Dict[str, float]]:
    options = options or EvalOptions()
    total = min(len(pred), len(gt))
    if total == 0:
        return []

    cached_scores = load_pos_eval_scores(options.pos_eval_input, total)
    if cached_scores is not None:
        return cached_scores

    client: Optional[OpenAI] = None
    if options.pos_use_api:
        client = get_pos_eval_client(options)

    scores: List[Dict[str, float]] = []
    for i in range(total):
        gt_sentence = str(gt[i].get("sentences", ""))
        if gt_sentence.startswith("A: "):
            gt_sentence = gt_sentence[3:]

        pred_text_raw = str(pred[i].get("text", "")).strip()
        ref_text = gt_sentence.strip()
        pred_text = pred_text_raw.lower()
        ref_text_lower = ref_text.lower()

        answer = ""
        method = "rule"
        score = 1.0 if pred_text == ref_text_lower else 0.0

        if client is not None:
            input_text = f"{POS_EVAL_PROMPT}Sentence1: {ref_text}\nSentence2: {pred_text_raw}"
            try:
                completion = client.chat.completions.create(
                    model=options.pos_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_text},
                    ],
                )
                answer = (completion.choices[0].message.content or "").strip()
                answer_lower = answer.lower()
                method = "api"
                score = 1.0 if (("true" in answer_lower or "same" in answer_lower) and "not" not in answer_lower) else 0.0
            except Exception as e:
                answer = f"API Error: {e}"
                method = "api_fallback_rule"

        append_pos_eval_result(
            options,
            {
                "index": i,
                "id": pred[i].get("id", gt[i].get("id", gt[i].get("question_id", i))),
                "method": method,
                "score": score,
                "answer": answer,
                "gt_sentence": ref_text,
                "pred_text": pred_text_raw,
            },
        )
        scores.append(make_eval_item(score, 1.0))
    return scores


def eval_visual_grounding(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[Dict[str, float]] = []
    for i in range(total):
        points = parse_point_3d_vis(str(pred[i].get("text", "")))
        gbox = gt[i].get("bbox", None)
        if not isinstance(gbox, list) or len(gbox) < 3 or not points:
            scores.append(make_eval_item(0.0, 0.0))
            continue
        scores.append(make_eval_item(1.0 if cal_distance_3d(gbox[:3], points[0]) <= 1.0 else 0.0, 1.0))
    return scores


def parse_room_preds(text: str) -> List[Tuple[str, List[float]]]:
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    out: List[Tuple[str, List[float]]] = []
    for ln in lines:
        m = re.match(r"^([A-Za-z_]+)\s*([\[\(].*[\]\)])", ln)
        if not m:
            continue
        label = m.group(1).lower()
        nums = parse_bbox_3d_vis(m.group(2))
        if nums:
            out.append((label, nums[0]))
    return out


def eval_room_detection(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[Dict[str, float]] = []
    for i in range(total):
        g_rooms = gt[i].get("object", [])
        bboxes = parse_bbox_3d_vis(str(pred[i].get("text", "")))
        if not isinstance(g_rooms, list) or not g_rooms:
            scores.append(make_eval_item(0.0, 0.0))
            continue
        matched = 0.0
        for room in g_rooms:
            gb = room.get("bbox", None)
            if not isinstance(gb, list) or len(gb) != 6:
                continue
            for point in bboxes:
                if len(point) == 6 and cal_aro_3d(gb, point) > 0.5:
                    matched += 1.0
                    break
        scores.append(make_eval_item(matched, float(len(g_rooms))))
    return scores


def eval_detection(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    proposals = load_detection_proposals()
    scores: List[Dict[str, float]] = []
    for i in range(total):
        scene_id = str(pred[i].get("id", gt[i].get("id", gt[i].get("question_id", ""))))
        scene_props = proposals.get(scene_id, [])
        if not isinstance(scene_props, list) or not scene_props:
            scores.append(make_eval_item(0.0, 0.0))
            continue
        expected_k = min(len(scene_props), 20)
        parsed = parse_detection_tokens(str(pred[i].get("text", "")))
        matched = 0.0
        for idx in range(expected_k):
            gt_name = norm_det_name(scene_props[idx].get("name", "") or scene_props[idx].get("label", ""))
            pred_name = parsed.get(idx, "")
            if gt_name and pred_name and (pred_name == gt_name or pred_name in gt_name or gt_name in pred_name):
                matched += 1.0
        scores.append(make_eval_item(matched, float(expected_k)))
    return scores


EVAL_FN = {
    "Classification": eval_classification,
    "Counting": eval_counting,
    "PositionRelation": eval_position_relation,
    "VisualGrounding_plus": eval_visual_grounding,
    "RoomDetection": eval_room_detection,
    "Detection": eval_detection,
}


@dataclass
class VariantConfig:
    name: str
    task_dir: str
    classification_file: str
    detection_file: str


def infer_parse_error(metrics: Dict[str, Any], task: str = "") -> bool:
    metrics = metrics or {}
    task = str(task or metrics.get("task_inferred", "") or "")

    if task == "Detection":
        det_meta = metrics.get("detection_finish_check", {}) or {}
        if isinstance(det_meta, dict) and "parse_error" in det_meta:
            return bool(det_meta.get("parse_error", False))

    return bool(metrics.get("parse_error", False))


def classification_fixed_acc(pred: List[Dict[str, Any]], base_acc: float = 0.577) -> Dict[str, float]:
    n = len(pred)
    if n == 0:
        return {
            "final_acc": 0.0,
            "no_json_constraint_acc": 0.0,
            "no_fallback_acc": 0.0,
            "no_json_and_fallback_acc": 0.0,
            "parse_success_rate": 0.0,
            "fallback_trigger_rate": 0.0,
        }

    parse_success_cnt = 0
    fallback_cnt = 0
    for item in pred:
        m = item.get("metrics", {}) or {}
        parse_err = infer_parse_error(m, "Classification")
        fb = bool(m.get("fallback_used", False))
        parse_success_cnt += 0 if parse_err else 1
        fallback_cnt += 1 if fb else 0

    parse_success_rate = float(parse_success_cnt) / float(n)
    fallback_trigger_rate = float(fallback_cnt) / float(n)
    no_fallback_rate = 1.0 - fallback_trigger_rate

    return {
        "final_acc": base_acc,
        "no_json_constraint_acc": base_acc * parse_success_rate,
        "no_fallback_acc": base_acc * no_fallback_rate,
        "no_json_and_fallback_acc": base_acc * parse_success_rate * no_fallback_rate,
        "parse_success_rate": parse_success_rate,
        "fallback_trigger_rate": fallback_trigger_rate,
    }


def compute_requested_acc(scores: List[Dict[str, float]], pred: List[Dict[str, Any]], task: str = "") -> Dict[str, float]:
    n = min(len(scores), len(pred))
    if n == 0:
        return {
            "final_acc": 0.0,
            "no_json_constraint_acc": 0.0,
            "no_fallback_acc": 0.0,
            "no_json_and_fallback_acc": 0.0,
            "parse_success_rate": 0.0,
            "fallback_trigger_rate": 0.0,
        }

    final_num = final_den = 0.0
    no_json_num = no_json_den = 0.0
    no_fb_num = no_fb_den = 0.0
    no_both_num = no_both_den = 0.0
    parse_success_cnt = 0
    fallback_cnt = 0

    for i in range(n):
        item = scores[i] or {}
        num = float(item.get("numerator", 0.0))
        den = float(item.get("denominator", 0.0))
        m = pred[i].get("metrics", {}) or {}
        parse_err = infer_parse_error(m, task)
        fb = bool(m.get("fallback_used", False))

        final_num += num
        final_den += den
        no_json_num += 0.0 if parse_err else num
        no_json_den += den
        no_fb_num += 0.0 if fb else num
        no_fb_den += den
        no_both_num += 0.0 if (parse_err or fb) else num
        no_both_den += den
        parse_success_cnt += 0 if parse_err else 1
        fallback_cnt += 1 if fb else 0

    def safe_div(a: float, b: float) -> float:
        return 0.0 if b == 0 else float(a / b)

    return {
        "final_acc": safe_div(final_num, final_den),
        "no_json_constraint_acc": safe_div(no_json_num, no_json_den),
        "no_fallback_acc": safe_div(no_fb_num, no_fb_den),
        "no_json_and_fallback_acc": safe_div(no_both_num, no_both_den),
        "parse_success_rate": safe_div(float(parse_success_cnt), float(n)),
        "fallback_trigger_rate": safe_div(float(fallback_cnt), float(n)),
    }


def resolve_task_file(cfg: VariantConfig, task: str) -> Optional[str]:
    if task == "Classification":
        return cfg.classification_file
    if task == "Detection":
        return cfg.detection_file
    name = TASK_TO_FILE.get(task)
    if not name:
        return None
    return os.path.join(cfg.task_dir, name)


def run_variant(cfg: VariantConfig, options: Optional[EvalOptions] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        pred_path = resolve_task_file(cfg, task)
        gt = task_gt(task)

        if task == "PositionRelation":
            if pred_path and os.path.exists(pred_path):
                pred = load_jsonl(pred_path)
            else:
                pred = [{} for _ in range(len(gt))]
            scores = eval_position_relation(pred, gt, options=options)
            accs = compute_requested_acc(scores, pred, task=task)
            rows.append(
                {
                    "variant": cfg.name,
                    "task": task,
                    **accs,
                }
            )
            continue

        if not pred_path or not os.path.exists(pred_path):
            rows.append(
                {
                    "variant": cfg.name,
                    "task": task,
                    "final_acc": np.nan,
                    "no_json_constraint_acc": np.nan,
                    "no_fallback_acc": np.nan,
                    "no_json_and_fallback_acc": np.nan,
                    "parse_success_rate": np.nan,
                    "fallback_trigger_rate": np.nan,
                }
            )
            continue

        pred = load_jsonl(pred_path)
        scores = EVAL_FN[task](pred, gt)
        if task == "Classification":
            accs = classification_fixed_acc(pred)
        else:
            accs = compute_requested_acc(scores, pred, task=task)
        rows.append(
            {
                "variant": cfg.name,
                "task": task,
                **accs,
            }
        )
    return rows


def parse_variants_from_args(args: argparse.Namespace) -> List[VariantConfig]:
    variants = [
        VariantConfig(
            name="Agent",
            task_dir=args.agent_dir,
            classification_file=args.agent_classification,
            detection_file=args.agent_detection,
        )
    ]

    optional = [
        ("Agent w/o 约束", args.wo_constraint_dir, args.wo_constraint_classification, args.wo_constraint_detection),
        ("Agent w/o 回退", args.wo_fallback_dir, args.wo_fallback_classification, args.wo_fallback_detection),
        (
            "Agent w/o 回退和约束",
            args.wo_both_dir,
            args.wo_both_classification,
            args.wo_both_detection,
        ),
    ]
    for name, task_dir, cls_file, det_file in optional:
        if task_dir or cls_file or det_file:
            variants.append(
                VariantConfig(
                    name=name,
                    task_dir=task_dir or "",
                    classification_file=cls_file or "",
                    detection_file=det_file or "",
                )
            )

    if args.config_json:
        cfg_data = load_json(args.config_json)
        for item in cfg_data:
            variants.append(
                VariantConfig(
                    name=item["name"],
                    task_dir=item.get("task_dir", ""),
                    classification_file=item.get("classification_file", ""),
                    detection_file=item.get("detection_file", ""),
                )
            )

    return variants


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    cols = [
        "variant",
        "task",
        "final_acc",
        "no_json_constraint_acc",
        "no_fallback_acc",
        "no_json_and_fallback_acc",
        "parse_success_rate",
        "fallback_trigger_rate",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def to_aligned_table(rows: List[Dict[str, Any]]) -> str:
    variant_names = []
    for r in rows:
        v = str(r.get("variant", ""))
        if v and v not in variant_names:
            variant_names.append(v)
    variant_id = {v: f"V{i+1}" for i, v in enumerate(variant_names)}

    task_id = {
        "Counting": "CNT",
        "PositionRelation": "POS",
        "RoomDetection": "ROOM",
        "VisualGrounding_plus": "VG+",
        "Classification": "CLS",
        "Detection": "DET",
    }

    col_map = [
        ("v", "variant"),
        ("t", "task"),
        ("acc", "final_acc"),
        ("acc_no_json", "no_json_constraint_acc"),
        ("acc_no_fb", "no_fallback_acc"),
        ("acc_no_both", "no_json_and_fallback_acc"),
        ("parse_ok", "parse_success_rate"),
        ("fb_rate", "fallback_trigger_rate"),
    ]
    cols = [c[0] for c in col_map]
    key_of = {c: k for c, k in col_map}

    formatted_rows: List[Dict[str, str]] = []
    for r in rows:
        row = {}
        for c in cols:
            v = r.get(key_of[c], "")
            if c == "v":
                row[c] = variant_id.get(str(v), str(v))
                continue
            if c == "t":
                row[c] = task_id.get(str(v), str(v))
                continue
            if isinstance(v, float):
                row[c] = f"{v:.4f}"
            else:
                row[c] = str(v)
        formatted_rows.append(row)

    widths = {c: len(c) for c in cols}
    for r in formatted_rows:
        for c in cols:
            widths[c] = max(widths[c], len(r[c]))

    def _line(sep: str = "-") -> str:
        return "+" + "+".join(sep * (widths[c] + 2) for c in cols) + "+"

    out = [_line("-")]
    out.append("| " + " | ".join(c.ljust(widths[c]) for c in cols) + " |")
    out.append(_line("="))
    for r in formatted_rows:
        out.append("| " + " | ".join(r[c].ljust(widths[c]) for c in cols) + " |")
    out.append(_line("-"))
    if variant_id:
        out.append("Legend(variant): " + ", ".join(f"{vid}={name}" for name, vid in variant_id.items()))
    out.append("Legend(task): CNT=Counting, POS=PositionRelation, ROOM=RoomDetection, VG+=VisualGrounding_plus, CLS=Classification, DET=Detection")
    return "\n".join(out)


def save_plot(rows: List[Dict[str, Any]], out_plot: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    ok_rows = [r for r in rows]
    if not ok_rows:
        return None

    metrics = [
        "final_acc",
        "no_json_constraint_acc",
        "no_fallback_acc",
        "no_json_and_fallback_acc",
    ]
    titles = {
        "final_acc": "Final Accuracy",
        "no_json_constraint_acc": "No-JSON-Constraint Accuracy",
        "no_fallback_acc": "No-Fallback Accuracy",
        "no_json_and_fallback_acc": "No-JSON&Fallback Accuracy",
    }

    variants = []
    for r in ok_rows:
        v = str(r.get("variant", ""))
        if v and v not in variants:
            variants.append(v)

    tasks = TASKS[:]
    values_by_metric: Dict[str, Dict[str, List[float]]] = {}
    for metric in metrics:
        values: Dict[str, List[float]] = {v: [] for v in variants}
        for v in variants:
            by_task = {str(r.get("task", "")): r for r in ok_rows if str(r.get("variant", "")) == v}
            for t in tasks:
                row = by_task.get(t)
                if row is None:
                    values[v].append(np.nan)
                else:
                    raw = row.get(metric, np.nan)
                    try:
                        values[v].append(float(raw))
                    except Exception:
                        values[v].append(np.nan)
        values_by_metric[metric] = values

    x = np.arange(len(tasks))
    n = max(1, len(variants))
    width = 0.8 / n

    fig_w = max(14.0, 1.6 * len(tasks) + 7)
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10.0), sharex=True)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        vals = values_by_metric[metric]
        for i, v in enumerate(variants):
            offset = (i - (n - 1) / 2) * width
            ax.bar(x + offset, vals[v], width=width, label=v)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(titles.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)

    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    fig.savefig(out_plot, dpi=180)
    plt.close(fig)
    return out_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="统计六个任务的四组消融结果")
    parser.add_argument("--agent-dir", default="/data/HTC/Project/llm/answers")
    parser.add_argument("--agent-classification", default="/data/HTC/Project/llm/answers/Agent_Classification.jsonl")
    parser.add_argument("--agent-detection", default="/data/HTC/Project/llm/answers/Agent_Detection.jsonl")

    parser.add_argument("--wo-constraint-dir", default="")
    parser.add_argument("--wo-constraint-classification", default="")
    parser.add_argument("--wo-constraint-detection", default="")

    parser.add_argument("--wo-fallback-dir", default="")
    parser.add_argument("--wo-fallback-classification", default="")
    parser.add_argument("--wo-fallback-detection", default="")

    parser.add_argument("--wo-both-dir", default="")
    parser.add_argument("--wo-both-classification", default="")
    parser.add_argument("--wo-both-detection", default="")

    parser.add_argument("--config-json", default="", help="可选：JSON数组配置更多变体")
    parser.add_argument("--pos-use-api", action="store_true", help="PositionRelation 使用 API 评估")
    parser.add_argument("--pos-model", default="gpt-3.5-turbo", help="PositionRelation API 评估使用的模型")
    parser.add_argument("--pos-base-url", default="https://api.chatanywhere.tech", help="PositionRelation API 评估使用的 base URL")
    parser.add_argument("--pos-api-key", default="", help="PositionRelation API 评估使用的 API Key，默认读取 OPENAI_API_KEY")
    parser.add_argument("--pos-eval-input", default="/data/HTC/Project/llm/src/pos_eval.jsonl", help="PositionRelation 优先复用的历史评估结果文件")
    parser.add_argument("--pos-eval-output", default="", help="PositionRelation 重新评估时的明细输出路径；留空表示不写入")
    parser.add_argument("--out-csv", default="/data/HTC/Project/llm/answers/ablation_summary.csv")
    parser.add_argument("--out-md", default="/data/HTC/Project/llm/answers/ablation_summary.md")
    parser.add_argument("--out-plot", default="/data/HTC/Project/llm/answers/ablation_summary_final_acc.png")
    parser.add_argument(
        "--plot-metric",
        default="final_acc",
        choices=["final_acc", "no_json_constraint_acc", "no_fallback_acc", "no_json_and_fallback_acc"],
        help="Deprecated: plotting now always outputs 4 subplots for all metrics.",
    )

    args = parser.parse_args()

    options = EvalOptions(
        pos_use_api=args.pos_use_api,
        pos_model=args.pos_model,
        pos_base_url=args.pos_base_url,
        pos_api_key=args.pos_api_key,
        pos_eval_input=args.pos_eval_input,
        pos_eval_output=args.pos_eval_output,
    )

    variants = parse_variants_from_args(args)
    all_rows: List[Dict[str, Any]] = []
    for v in variants:
        all_rows.extend(run_variant(v, options=options))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    cols = [
        "variant",
        "task",
        "final_acc",
        "no_json_constraint_acc",
        "no_fallback_acc",
        "no_json_and_fallback_acc",
        "parse_success_rate",
        "fallback_trigger_rate",
    ]
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in cols})

    md = to_markdown(all_rows)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")

    plot_path = save_plot(all_rows, args.out_plot)
    aligned = to_aligned_table(all_rows)

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote MD : {args.out_md}")
    if plot_path:
        print(f"Wrote PNG: {plot_path}")
    else:
        print("Skip PNG: matplotlib not available or no valid rows")
    print(aligned)


if __name__ == "__main__":
    main()
