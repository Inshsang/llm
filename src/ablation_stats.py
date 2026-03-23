import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


def eval_classification(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[float]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[float] = []
    for i in range(total):
        p = pred[i]
        g = gt[i]
        p_label, p_text = parse_choice_label_and_text(str(p.get("text", "")))
        ans = str(g.get("sentences", ""))
        g_label, _ = parse_choice_label_and_text(ans)
        g_choice = g.get("gt_choices", [])
        g_idx = int(g.get("gt_choice", -1)) if g.get("gt_choice", None) is not None else -1
        g_name = str(g_choice[g_idx]).lower() if isinstance(g_choice, list) and 0 <= g_idx < len(g_choice) else ""
        ok = (p_label and g_label and p_label == g_label) or (g_name and g_name in p_text)
        scores.append(1.0 if ok else 0.0)
    return scores


def eval_counting(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[float]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[float] = []
    for i in range(total):
        text = str(pred[i].get("text", ""))
        nums = re.findall(r"(?<![\d.])-?\d+(?![\d.])", text)
        if not nums:
            scores.append(0.0)
            continue
        p_num = int(nums[0])
        g_choices = gt[i].get("gt_choices", [])
        g_idx = int(gt[i].get("gt_choice", -1)) if gt[i].get("gt_choice", None) is not None else -1
        if isinstance(g_choices, list) and 0 <= g_idx < len(g_choices):
            scores.append(1.0 if p_num == int(g_choices[g_idx]) else 0.0)
        else:
            scores.append(0.0)
    return scores


def eval_position_relation(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[float]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[float] = []
    for i in range(total):
        p_text = re.sub(r"\s+", " ", str(pred[i].get("text", "")).lower())
        g_choices = gt[i].get("gt_choices", [])
        g_idx = int(gt[i].get("gt_choice", -1)) if gt[i].get("gt_choice", None) is not None else -1
        if isinstance(g_choices, list) and 0 <= g_idx < len(g_choices):
            g_text = re.sub(r"\s+", " ", str(g_choices[g_idx]).lower())
            scores.append(1.0 if g_text and (g_text in p_text or p_text in g_text) else 0.0)
        else:
            scores.append(0.0)
    return scores


def eval_visual_grounding(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[float]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    scores: List[float] = []
    for i in range(total):
        boxes = parse_bbox_3d_vis(str(pred[i].get("text", "")))
        gbox = gt[i].get("bbox", None)
        if not isinstance(gbox, list) or len(gbox) != 6:
            scores.append(0.0)
            continue
        ok = any(cal_in_3d(gbox.copy(), b.copy()) == 1 for b in boxes if isinstance(b, list) and len(b) == 6)
        scores.append(1.0 if ok else 0.0)
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


def eval_room_detection(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[float]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    recalls: List[float] = []
    for i in range(total):
        g_rooms = gt[i].get("object", [])
        p_rooms = parse_room_preds(str(pred[i].get("text", "")))
        if not isinstance(g_rooms, list) or not g_rooms:
            recalls.append(0.0)
            continue
        matched = 0
        for gr in g_rooms:
            glabel = str(gr.get("label", "")).lower()
            gb = gr.get("bbox", None)
            if not isinstance(gb, list) or len(gb) != 6:
                continue
            best = 0.0
            for plabel, pb in p_rooms:
                if plabel == glabel:
                    best = max(best, cal_iou_3d(gb.copy(), pb.copy()))
            if best >= 0.5:
                matched += 1
        recalls.append(matched / max(1, len(g_rooms)))
    return recalls


def eval_detection(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]]) -> List[float]:
    total = min(len(pred), len(gt))
    if total == 0:
        return []
    f1s: List[float] = []
    pattern = re.compile(r"\(obj\s*\d+\)\s*:\s*([A-Za-z0-9_]+)\s*!?")
    for i in range(total):
        p_text = str(pred[i].get("text", ""))
        p_names = [x.lower() for x in pattern.findall(p_text)]
        p_set = set(p_names)

        g_objs = gt[i].get("object", [])
        g_set = set()
        if isinstance(g_objs, list):
            for o in g_objs:
                n = str(o.get("name", "") or o.get("label", "")).lower()
                if n:
                    g_set.add(n)
        if not p_set and not g_set:
            f1s.append(1.0)
            continue
        if not p_set or not g_set:
            f1s.append(0.0)
            continue
        inter = len(p_set & g_set)
        prec = inter / len(p_set)
        rec = inter / len(g_set)
        f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
        f1s.append(f1)
    return f1s


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


def compute_requested_acc(scores: List[float], pred: List[Dict[str, Any]]) -> Dict[str, float]:
    n = min(len(scores), len(pred))
    if n == 0:
        return {
            "final_acc": 0.0,
            "no_json_constraint_acc": 0.0,
            "no_fallback_acc": 0.0,
            "no_json_and_fallback_acc": 0.0,
        }

    final_list: List[float] = []
    no_json_list: List[float] = []
    no_fb_list: List[float] = []
    no_both_list: List[float] = []

    for i in range(n):
        s = float(scores[i])
        m = pred[i].get("metrics", {}) or {}
        parse_err = bool(m.get("parse_error", False))
        fb = bool(m.get("fallback_used", False))

        final_list.append(s)
        no_json_list.append(0.0 if parse_err else s)
        no_fb_list.append(0.0 if fb else s)
        no_both_list.append(0.0 if (parse_err or fb) else s)

    return {
        "final_acc": float(np.mean(final_list)),
        "no_json_constraint_acc": float(np.mean(no_json_list)),
        "no_fallback_acc": float(np.mean(no_fb_list)),
        "no_json_and_fallback_acc": float(np.mean(no_both_list)),
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


def run_variant(cfg: VariantConfig) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        pred_path = resolve_task_file(cfg, task)
        if not pred_path or not os.path.exists(pred_path):
            rows.append(
                {
                    "variant": cfg.name,
                    "task": task,
                    "final_acc": np.nan,
                    "no_json_constraint_acc": np.nan,
                    "no_fallback_acc": np.nan,
                    "no_json_and_fallback_acc": np.nan,
                }
            )
            continue

        pred = load_jsonl(pred_path)
        gt = task_gt(task)
        scores = EVAL_FN[task](pred, gt)
        accs = compute_requested_acc(scores, pred)
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
    parser.add_argument("--agent-dir", default="/data/HTC/Project/llm/answers/v2")
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

    variants = parse_variants_from_args(args)
    all_rows: List[Dict[str, Any]] = []
    for v in variants:
        all_rows.extend(run_variant(v))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    cols = [
        "variant",
        "task",
        "final_acc",
        "no_json_constraint_acc",
        "no_fallback_acc",
        "no_json_and_fallback_acc",
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
