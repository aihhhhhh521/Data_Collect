#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from config import (
    MANIFEST_FILE,
    PARQUET_BATCH_SIZE,
    PRECLASSIFIED_FILE,
    NEED_LLM_FILE,
    TEXT_MAX_CHARS,
    WORK_DIR,
)
from utils import ensure_exists, safe_str, truncate_text


RULE_REJECT_FILE = WORK_DIR / "rule_rejected.parquet"


def build_text_for_cls(row: pd.Series) -> str:
    ai_desc = safe_str(row.get("ai_description")).strip()
    if not ai_desc:
        return ""
    return truncate_text(f"ai_description: {ai_desc}", TEXT_MAX_CHARS)


def normalize_batch_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = [
        "photo_width",
        "photo_height",
        "stats_views",
        "stats_downloads",
        "rule_top1_score",
        "rule_top2_score",
        "rule_margin",
        "rule_positive_strong_hits",
        "rule_positive_weak_hits",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = [
        "photo_aspect_ratio",
        "exif_iso",
        "photo_location_latitude",
        "photo_location_longitude",
        "ai_primary_landmark_latitude",
        "ai_primary_landmark_longitude",
        "ai_primary_landmark_confidence",
        "category_confidence",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    bool_cols = ["photo_featured", "needs_llm", "rule_gate_pass"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype("boolean")

    string_cols = [
        "photo_id", "photo_url", "photo_image_url", "photo_description",
        "photographer_username", "photographer_first_name", "photographer_last_name",
        "exif_camera_make", "exif_camera_model", "exif_aperture_value",
        "exif_focal_length", "exif_exposure_time", "photo_location_name",
        "photo_location_country", "photo_location_city", "ai_description",
        "ai_primary_landmark_name", "blur_hash", "text_for_cls", "rule_top1_label",
        "rule_scores_json", "rule_candidate_labels_json", "rule_matched_terms_json",
        "rule_reject_reason", "category", "category_source",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if "photo_submitted_at" in df.columns:
        df["photo_submitted_at"] = pd.to_datetime(df["photo_submitted_at"], errors="coerce")

    return df


def enrich_batch(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text_for_cls"] = out.apply(build_text_for_cls, axis=1)

    out["rule_top1_label"] = pd.NA
    out["rule_top1_score"] = pd.NA
    out["rule_top2_score"] = pd.NA
    out["rule_margin"] = pd.NA
    out["rule_scores_json"] = pd.NA
    out["rule_candidate_labels_json"] = pd.NA
    out["rule_matched_terms_json"] = pd.NA
    out["rule_positive_strong_hits"] = pd.NA
    out["rule_positive_weak_hits"] = pd.NA
    out["rule_gate_pass"] = pd.NA
    out["rule_reject_reason"] = pd.NA

    out["needs_llm"] = True
    out["category"] = pd.NA
    out["category_confidence"] = pd.NA
    out["category_source"] = pd.NA
    return normalize_batch_dtypes(out)


def main() -> None:
    ensure_exists(MANIFEST_FILE, "请先运行 build_manifest.py")

    for path in [PRECLASSIFIED_FILE, NEED_LLM_FILE, RULE_REJECT_FILE]:
        if path.exists():
            print(f"[WARN] 检测到旧文件，将覆盖写入：{path}")
            path.unlink()

    print(f"[INFO] ai-only 输入文件：{MANIFEST_FILE}")

    parquet_file = pq.ParquetFile(MANIFEST_FILE)
    writer_all = None
    writer_llm = None
    fixed_schema = None
    total_rows = 0

    for batch in tqdm(parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE), desc="prepare_ai_only_need_llm"):
        df = batch.to_pandas()
        out = enrich_batch(df)
        total_rows += len(out)

        table = pa.Table.from_pandas(out, preserve_index=False)
        if fixed_schema is None:
            fixed_schema = table.schema
            writer_all = pq.ParquetWriter(PRECLASSIFIED_FILE, fixed_schema, compression="zstd")
            writer_llm = pq.ParquetWriter(NEED_LLM_FILE, fixed_schema, compression="zstd")
        else:
            table = table.cast(fixed_schema)

        writer_all.write_table(table)
        writer_llm.write_table(table)

    if writer_all is not None:
        writer_all.close()
    if writer_llm is not None:
        writer_llm.close()

    pd.DataFrame().to_parquet(RULE_REJECT_FILE, index=False)

    print(f"[OK] ai-only 预分类底表：{PRECLASSIFIED_FILE}")
    print(f"[OK] 待送 vLLM 文件：{NEED_LLM_FILE}")
    print(f"[OK] rule 直接拒绝文件（空）：{RULE_REJECT_FILE}")
    print(f"[INFO] 待送 vLLM 数量：{total_rows}")


if __name__ == "__main__":
    main()
