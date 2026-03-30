# Unsplash Data Collect Pipeline

这是一个最小可运行的数据处理流水线，核心流程如下：

1. `scripts/build_manifest.py`：读取 `DATA_ROOT/photos.csv*`，筛选 featured 且有 `ai_description` 的记录，输出 `manifest.parquet`。
2. `scripts/prepare_llm.py`：构建待分类字段，生成 `preclassified.parquet` 与 `need_llm.parquet`。
3. `scripts/vllm_classify.py`：用本地 vLLM 对 `need_llm.parquet` 做五类分类/拒绝判断，输出 `vllm_results.jsonl/parquet`。
4. `scripts/clean_results.py`：清洗推理结果，导出最终分类、待复核和统计文件。

> 你可以按需修改 `scripts/` 下脚本与配置；本仓库 README 只保留核心说明。