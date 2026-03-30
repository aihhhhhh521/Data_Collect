#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def safe_str(x) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "<na>", "none"}:
        return ""
    return s.strip()


def ensure_exists(path: Path, msg: str | None = None):
    if not path.exists():
        raise FileNotFoundError(msg or f"文件不存在: {path}")


def load_config_paths():
    try:
        from config import (
            VLLM_RESULTS_FILE,
            CLASSIFIED_FILE,
            NEED_REVIEW_FILE,
            STATS_FILE,
        )
        return {
            "input_file": Path(VLLM_RESULTS_FILE),
            "classified_file": Path(CLASSIFIED_FILE),
            "need_review_file": Path(NEED_REVIEW_FILE),
            "stats_file": Path(STATS_FILE),
        }
    except Exception:
        work_dir = Path("work")
        work_dir.mkdir(parents=True, exist_ok=True)
        return {
            "input_file": work_dir / "vllm_results.parquet",
            "classified_file": work_dir / "classified.parquet",
            "need_review_file": work_dir / "need_review.parquet",
            "stats_file": work_dir / "category_stats.csv",
        }


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "photo_id",
        "vllm_label",
        "vllm_reason",
        "vllm_raw_response",
        "vllm_error",
        "vllm_reject_reason",
        "vllm_source_hint",
        "vllm_finished_at",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["vllm_ok", "vllm_in_scope", "vllm_called"]:
        if col not in df.columns:
            df[col] = pd.NA

    if "vllm_confidence" not in df.columns:
        df["vllm_confidence"] = pd.NA

    df["photo_id"] = df["photo_id"].map(safe_str)
    df["vllm_label"] = df["vllm_label"].map(safe_str)
    df["vllm_reason"] = df["vllm_reason"].map(safe_str)
    df["vllm_raw_response"] = df["vllm_raw_response"].map(safe_str)
    df["vllm_error"] = df["vllm_error"].map(safe_str)
    df["vllm_reject_reason"] = df["vllm_reject_reason"].map(safe_str)
    df["vllm_source_hint"] = df["vllm_source_hint"].map(safe_str)

    df["vllm_confidence"] = pd.to_numeric(df["vllm_confidence"], errors="coerce")
    df["vllm_ok"] = df["vllm_ok"].astype("boolean")
    df["vllm_in_scope"] = df["vllm_in_scope"].astype("boolean")
    df["vllm_called"] = df["vllm_called"].astype("boolean")

    df = df[df["photo_id"] != ""].copy()

    if "vllm_finished_at" in df.columns:
        try:
            df["_sort_ts"] = pd.to_datetime(df["vllm_finished_at"], errors="coerce")
            df = df.sort_values(["photo_id", "_sort_ts"], kind="mergesort")
            df = df.drop_duplicates(subset=["photo_id"], keep="last")
            df = df.drop(columns=["_sort_ts"])
        except Exception:
            df = df.drop_duplicates(subset=["photo_id"], keep="last")
    else:
        df = df.drop_duplicates(subset=["photo_id"], keep="last")

    df["category"] = df["vllm_label"].replace("", pd.NA)
    df["category_confidence"] = df["vllm_confidence"]
    df["category_source"] = "vllm"

    return df


def split_results(df: pd.DataFrame, min_confidence: float | None):
    df = df.copy()

    has_label = df["vllm_label"].fillna("").astype(str).str.strip() != ""
    ok_mask = df["vllm_ok"].fillna(False)

    if "vllm_in_scope" in df.columns:
        in_scope_mask = df["vllm_in_scope"].fillna(False)
    else:
        in_scope_mask = has_label

    accept_mask = has_label & ok_mask & in_scope_mask

    if min_confidence is not None:
        accept_mask = accept_mask & (df["vllm_confidence"].fillna(0) >= min_confidence)

    classified_df = df[accept_mask].copy()
    review_df = df[~accept_mask].copy()

    classified_df["review_flag"] = False
    review_df["review_flag"] = True
    review_df["reject_reason"] = ""

    review_df.loc[
        (review_df["reject_reason"] == "") & (review_df["vllm_label"].fillna("").astype(str).str.strip() == ""),
        "reject_reason"
    ] = "empty_label"
    review_df.loc[
        (review_df["reject_reason"] == "") & (~review_df["vllm_ok"].fillna(False)),
        "reject_reason"
    ] = "vllm_failed"
    if "vllm_in_scope" in review_df.columns:
        review_df.loc[
            (review_df["reject_reason"] == "") & (~review_df["vllm_in_scope"].fillna(False)),
            "reject_reason"
        ] = "out_of_scope_or_rejected"
    if min_confidence is not None:
        review_df.loc[
            (review_df["reject_reason"] == "") & (review_df["vllm_confidence"].fillna(0) < min_confidence),
            "reject_reason"
        ] = "low_confidence"

    review_df.loc[review_df["reject_reason"] == "", "reject_reason"] = "other"

    return classified_df, review_df


def build_stats(all_df: pd.DataFrame, classified_df: pd.DataFrame, review_df: pd.DataFrame):
    stats_all = (
        all_df.groupby(["category", "category_source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
    )

    stats_classified = (
        classified_df.groupby(["category", "category_source"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
    )

    stats_review = (
        review_df.groupby(["reject_reason"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "reject_reason"], ascending=[False, True])
    )

    return stats_all, stats_classified, stats_review


def main():
    defaults = load_config_paths()

    parser = argparse.ArgumentParser(description="清洗整理 vllm_results.parquet，并可选按阈值过滤")
    parser.add_argument("--input", type=Path, default=defaults["input_file"], help="输入 parquet")
    parser.add_argument("--classified", type=Path, default=defaults["classified_file"], help="最终保留结果")
    parser.add_argument("--review", type=Path, default=defaults["need_review_file"], help="待复核/被过滤结果")
    parser.add_argument("--stats", type=Path, default=defaults["stats_file"], help="统计 csv")
    parser.add_argument("--min-confidence", type=float, default=None, help="可选：最小置信度阈值，例如 0.6")
    args = parser.parse_args()

    input_file = args.input
    classified_file = args.classified
    review_file = args.review
    stats_file = args.stats

    classified_all_file = classified_file.with_name("classified_all.parquet")
    rejected_file = classified_file.with_name("rejected_or_low_conf.parquet")
    stats_all_file = stats_file.with_name("category_stats_all.csv")
    stats_review_file = stats_file.with_name("category_review_stats.csv")

    ensure_exists(input_file, f"找不到输入文件：{input_file}")

    print(f"[INFO] 读取输入：{input_file}")
    df = pd.read_parquet(input_file)
    print(f"[INFO] 原始行数：{len(df)}")

    df = normalize_df(df)
    print(f"[INFO] 规范化、去重后行数：{len(df)}")

    classified_df, review_df = split_results(df, args.min_confidence)

    classified_file.parent.mkdir(parents=True, exist_ok=True)
    review_file.parent.mkdir(parents=True, exist_ok=True)

    classified_df.to_parquet(classified_file, index=False)
    review_df.to_parquet(review_file, index=False)
    review_df.to_parquet(rejected_file, index=False)
    df.to_parquet(classified_all_file, index=False)

    stats_all, stats_classified, stats_review = build_stats(df, classified_df, review_df)
    stats_all.to_csv(stats_all_file, index=False, encoding="utf-8-sig")
    stats_classified.to_csv(stats_file, index=False, encoding="utf-8-sig")
    stats_review.to_csv(stats_review_file, index=False, encoding="utf-8-sig")

    print(f"[OK] 全量清洗结果：{classified_all_file}")
    print(f"[OK] 最终保留结果：{classified_file}")
    print(f"[OK] 待复核结果：{review_file}")
    print(f"[OK] 复核/拒绝结果：{rejected_file}")
    print(f"[OK] 全量统计：{stats_all_file}")
    print(f"[OK] 最终统计：{stats_file}")
    print(f"[OK] 复核原因统计：{stats_review_file}")

    print()
    print(f"[SUMMARY] 原始总数: {len(df)}")
    print(f"[SUMMARY] 最终保留: {len(classified_df)}")
    print(f"[SUMMARY] 待复核/剔除: {len(review_df)}")
    if args.min_confidence is not None:
        print(f"[SUMMARY] 启用置信度阈值: {args.min_confidence:.3f}")
    else:
        print("[SUMMARY] 未启用置信度阈值")

    if len(stats_classified) > 0:
        print("\\n最终分类分布：")
        print(stats_classified.to_string(index=False))

    if len(stats_review) > 0:
        print("\\n待复核/剔除原因分布：")
        print(stats_review.to_string(index=False))


if __name__ == "__main__":
    main()
