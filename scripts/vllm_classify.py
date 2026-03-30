#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# 默认使用当前脚本所在目录，避免重命名目录后导入失败
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    LABELS,
    NEED_LLM_FILE,
    VLLM_MAX_OUTPUT_TOKENS_PER_ITEM,
    VLLM_MODEL,
    VLLM_RESULTS_FILE,
    VLLM_RESULTS_JSONL,
    VLLM_RETURN_REASON,
    VLLM_TEXT_MAX_CHARS,
)
from utils import ensure_exists, load_done_ids_from_jsonl, safe_str, truncate_text
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

REJECT_LABEL = "拒绝"
ALL_OUTPUT_LABELS = LABELS + [REJECT_LABEL]

# 指向服务器本地模型目录；
VLLM_MODEL = os.getenv("VLLM_MODEL", VLLM_MODEL)
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
VLLM_MAX_MODEL_LEN = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
VLLM_DTYPE = os.getenv("VLLM_DTYPE", "auto")
VLLM_TRUST_REMOTE_CODE = os.getenv("VLLM_TRUST_REMOTE_CODE", "1") not in {"0", "false", "False"}
VLLM_BATCH_SIZE = int(os.getenv("VLLM_BATCH_SIZE", "128"))

SCHEMA = {
    "type": "object",
    "properties": {
        "is_target": {"type": "boolean"},
        "label": {"type": "string", "enum": ALL_OUTPUT_LABELS},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["is_target", "label", "confidence", "reason"],
}

SYSTEM_PROMPT = (
    "你是一个非常保守的 Unsplash 图片元数据审核器。"
    "输入不是图片本身，而是图片的文本元数据。"
    "当前你只能参考 ai_description 来做判断。"
    "你的首要任务不是强行五分类，而是先判断它是否真的可以稳定归入目标五大类。"
    "只要元数据证据不足、主体不明确、多个大类同等合理、或明显不像真实摄影照片主体，就必须拒绝。"
    "你必须只输出符合 JSON Schema 的 JSON，不得输出任何额外文字。"
)

LOCAL_REJECT_TERMS = {
    "illustration", "painting", "drawing", "poster", "logo", "graphic", "diagram",
    "map", "screenshot", "menu", "brochure", "abstract", "pattern", "texture",
    "render", "rendering", "cgi", "3d render", "wallpaper",
}

CATEGORY_GUIDE = """
目标五类定义（按“主视觉主体”判定）：
1. 城市、建筑：城市室外、建筑外观、街景、桥梁、地标、校园外景、天际线；主体是 building environment，而不是室内空间、人物或单个物体。
2. 室内：室内空间本身是主体，如房间、客厅、卧室、厨房、走廊、办公室、图书馆、酒店房间；如果只是桌面小物或餐食，不算室内。
3. 自然：自然风景、山水、森林、海洋、河流、天空、沙漠、花草、动物；主体是自然环境或自然生物。
4. 静物：食物、饮品、产品、器具、桌面物品、书本、相机、手机、手表、车辆特写等 object-centric 画面；主体是物体，不是空间，不是人。
5. 人像：人或人群是主要主体，包含单人、多人、半身、全身、特写、街拍人像；只要“人明显是主体”，优先归到人像。

必须拒绝的情况：
- 元数据无法支持“主视觉主体”判断
- 同时像多个大类，且没有明显主类
- 明显像插画、海报、Logo、UI 截图、图表、文档、抽象纹理、渲染图
- 看起来可能是普通记录照，但主体并不稳定落在上述五类之一

严格要求：
- 不要因为出现城市名、国家名、地标名，就自动判“城市、建筑”
- 不要因为出现 person / people / woman / man，就自动判“人像”；只有“人是主体”才算
- 不要因为出现 room / hotel / indoor，就自动判“室内”；如果更像桌面物品、餐食、产品，则应判“静物”
- 不要因为出现 tree / flower / animal 单词，就自动判“自然”；必须看起来主体确实是自然环境或自然生物
- 证据不够就拒绝，不要硬判
""".strip()


def short_text(s: str, max_chars: int = VLLM_TEXT_MAX_CHARS) -> str:
    return truncate_text(safe_str(s).strip(), max_chars)


def has_local_reject_hint(text: str) -> list[str]:
    low = safe_str(text).lower()
    hits = [x for x in LOCAL_REJECT_TERMS if x in low]
    return sorted(set(hits))


def build_prompt(row: dict) -> str:
    text_for_cls = short_text(row.get("text_for_cls", ""))
    reason_instruction = "reason 用一句极简中文说明主依据；若拒绝，说明拒绝原因。" if VLLM_RETURN_REASON else "reason 固定输出空字符串。"

    return f"""[System]
{SYSTEM_PROMPT}

[Task]
请只依据下面这条图片元数据中的 ai_description 来判断，
它是否可以稳定归入以下五类之一：

{CATEGORY_GUIDE}

输出 JSON 字段：
- reason: {reason_instruction}
- is_target: true / false
- label: 若 is_target=true，则必须是五类之一；若 is_target=false，则输出“{REJECT_LABEL}”
- confidence: 0 到 1 之间，表示你对“属于目标五类且标签正确”的信心；保守打分

[Metadata]
{text_for_cls}

[Output]
请只输出 JSON：
""".strip()


def parse_response_obj(obj: dict) -> dict:
    is_target = bool(obj.get("is_target", False))
    label = safe_str(obj.get("label", "")).strip()
    confidence = float(obj.get("confidence", 0.0))
    reason = safe_str(obj.get("reason", "")).strip()

    if label not in ALL_OUTPUT_LABELS:
        raise ValueError(f"非法 label: {label}")

    confidence = max(0.0, min(1.0, confidence))

    if (not is_target) or label == REJECT_LABEL:
        return {
            "vllm_label": None,
            "vllm_confidence": round(confidence, 3),
            "vllm_reason": reason if VLLM_RETURN_REASON else "",
            "vllm_in_scope": False,
            "vllm_reject_reason": reason or "llm_reject",
        }

    return {
        "vllm_label": label,
        "vllm_confidence": round(confidence, 3),
        "vllm_reason": reason if VLLM_RETURN_REASON else "",
        "vllm_in_scope": True,
        "vllm_reject_reason": "",
    }


def safe_parse_json_text(text: str) -> dict:
    text = safe_str(text).strip()
    if not text:
        raise ValueError("empty model response")
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError(f"cannot parse json from response: {text[:300]}")


def local_fast_reject(row: dict) -> dict | None:
    photo_id = safe_str(row.get("photo_id"))
    text_for_cls = short_text(row.get("text_for_cls", ""))
    hits = has_local_reject_hint(text_for_cls)

    if not text_for_cls or len(text_for_cls) < 10:
        return {
            "photo_id": photo_id,
            "vllm_label": None,
            "vllm_confidence": 0.0,
            "vllm_reason": "",
            "vllm_raw_response": "",
            "vllm_ok": True,
            "vllm_error": "",
            "vllm_in_scope": False,
            "vllm_reject_reason": "metadata_too_short",
            "vllm_called": False,
            "vllm_source_hint": "",
            "vllm_finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    if hits:
        return {
            "photo_id": photo_id,
            "vllm_label": None,
            "vllm_confidence": 0.0,
            "vllm_reason": "",
            "vllm_raw_response": "",
            "vllm_ok": True,
            "vllm_error": "",
            "vllm_in_scope": False,
            "vllm_reject_reason": f"local_reject_terms={','.join(hits[:6])}",
            "vllm_called": False,
            "vllm_source_hint": "",
            "vllm_finished_at": datetime.now().isoformat(timespec="seconds"),
        }

    return None


def build_llm() -> LLM:
    print(f"[INFO] 使用本地 vLLM 离线推理")
    print(f"[INFO] VLLM_MODEL = {VLLM_MODEL}")
    print(f"[INFO] tensor_parallel_size = {VLLM_TENSOR_PARALLEL_SIZE}")
    print(f"[INFO] gpu_memory_utilization = {VLLM_GPU_MEMORY_UTILIZATION}")
    print(f"[INFO] max_model_len = {VLLM_MAX_MODEL_LEN}")
    print(f"[INFO] dtype = {VLLM_DTYPE}")

    llm = LLM(
        model=VLLM_MODEL,
        tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        max_model_len=VLLM_MAX_MODEL_LEN,
        dtype=VLLM_DTYPE,
        trust_remote_code=VLLM_TRUST_REMOTE_CODE,
        generation_config="vllm",
    )
    return llm


def run_batch_inference(llm: LLM, rows: list[dict]) -> list[dict]:
    prompts = [build_prompt(r) for r in rows]
    structured = StructuredOutputsParams(json=SCHEMA)
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=VLLM_MAX_OUTPUT_TOKENS_PER_ITEM,
        structured_outputs=structured,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    results = []
    for row, out in zip(rows, outputs):
        photo_id = safe_str(row.get("photo_id"))
        text = ""
        try:
            text = out.outputs[0].text if out.outputs else ""
            obj = safe_parse_json_text(text)
            parsed = parse_response_obj(obj)
            results.append(
                {
                    "photo_id": photo_id,
                    "vllm_label": parsed["vllm_label"],
                    "vllm_confidence": parsed["vllm_confidence"],
                    "vllm_reason": parsed["vllm_reason"],
                    "vllm_raw_response": text,
                    "vllm_ok": True,
                    "vllm_error": "",
                    "vllm_in_scope": parsed["vllm_in_scope"],
                    "vllm_reject_reason": parsed["vllm_reject_reason"],
                    "vllm_called": True,
                    "vllm_source_hint": "",
                    "vllm_finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
        except Exception as e:
            results.append(
                {
                    "photo_id": photo_id,
                    "vllm_label": None,
                    "vllm_confidence": 0.0,
                    "vllm_reason": "",
                    "vllm_raw_response": text,
                    "vllm_ok": False,
                    "vllm_error": str(e),
                    "vllm_in_scope": False,
                    "vllm_reject_reason": "vllm_parse_failed",
                    "vllm_called": True,
                    "vllm_source_hint": "",
                    "vllm_finished_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
    return results


EMPTY_RESULT_COLUMNS = [
    "photo_id",
    "vllm_label",
    "vllm_confidence",
    "vllm_reason",
    "vllm_raw_response",
    "vllm_ok",
    "vllm_error",
    "vllm_in_scope",
    "vllm_reject_reason",
    "vllm_called",
    "vllm_source_hint",
    "vllm_finished_at",
]


def load_jsonl_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["_line_no"] = line_no
                records.append(obj)
            except Exception:
                continue

    if not records:
        return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    out_df = pd.DataFrame(records)
    if "photo_id" not in out_df.columns:
        return pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    out_df = out_df.sort_values("_line_no").drop_duplicates(subset=["photo_id"], keep="last")
    out_df = out_df.drop(columns=["_line_no"])
    return out_df


def append_jsonl(records: list[dict], path: Path):
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def main() -> None:
    ensure_exists(NEED_LLM_FILE, "请先运行 prepare_llm.py")
    VLLM_RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(NEED_LLM_FILE)
    if len(df) == 0:
        print("[INFO] 没有需要送给 vLLM 的样本。")
        pd.DataFrame(columns=EMPTY_RESULT_COLUMNS).to_parquet(VLLM_RESULTS_FILE, index=False)
        return

    done_ids = load_done_ids_from_jsonl(VLLM_RESULTS_JSONL)
    todo_df = df[~df["photo_id"].astype(str).isin(done_ids)].copy()

    print(f"[INFO] 待审核总数：{len(df)}")
    print(f"[INFO] 已完成（断点续跑）：{len(done_ids)}")
    print(f"[INFO] 本轮待处理：{len(todo_df)}")

    if len(todo_df) == 0:
        out_df = load_jsonl_results(VLLM_RESULTS_JSONL)
        if len(out_df) == 0:
            out_df = pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)
        out_df.to_parquet(VLLM_RESULTS_FILE, index=False)
        print("[INFO] 无需新增推理。")
        return

    keep_cols = [c for c in ["photo_id", "text_for_cls"] if c in todo_df.columns]
    all_rows = todo_df[keep_cols].to_dict(orient="records")

    llm = build_llm()

    final_results = []
    infer_rows = []

    for row in all_rows:
        local_reject = local_fast_reject(row)
        if local_reject is not None:
            final_results.append(local_reject)
        else:
            infer_rows.append(row)

    if final_results:
        append_jsonl(final_results, VLLM_RESULTS_JSONL)

    if infer_rows:
        print(f"[INFO] 进入本地 vLLM 推理的样本数：{len(infer_rows)}")
        for start in tqdm(range(0, len(infer_rows), VLLM_BATCH_SIZE), desc="vllm_local_ai_description"):
            batch_rows = infer_rows[start:start + VLLM_BATCH_SIZE]
            batch_results = run_batch_inference(llm, batch_rows)
            append_jsonl(batch_results, VLLM_RESULTS_JSONL)

    out_df = load_jsonl_results(VLLM_RESULTS_JSONL)
    if len(out_df) == 0:
        out_df = pd.DataFrame(columns=EMPTY_RESULT_COLUMNS)

    for col in EMPTY_RESULT_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = None
    out_df = out_df[EMPTY_RESULT_COLUMNS]
    out_df.to_parquet(VLLM_RESULTS_FILE, index=False)

    ok_n = int(out_df["vllm_ok"].fillna(False).sum()) if len(out_df) else 0
    in_scope_n = int(out_df["vllm_in_scope"].fillna(False).sum()) if len(out_df) else 0
    reject_n = len(out_df) - in_scope_n

    print(f"[OK] 本地 vLLM 结果文件（兼容旧字段名）：{VLLM_RESULTS_FILE}")
    print(f"[INFO] 调用成功条数：{ok_n} / {len(out_df)}")
    print(f"[INFO] 通过五类审核：{in_scope_n}")
    print(f"[INFO] 拒绝/排除：{reject_n}")


if __name__ == "__main__":
    main()
