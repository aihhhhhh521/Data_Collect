# scripts 目录说明

## 核心文件
- `config.py`：统一路径与参数（支持环境变量覆盖）。
- `utils.py`：通用工具函数。
- `build_manifest.py`：构建 `manifest.parquet`。
- `prepare_llm.py`：准备待 LLM 分类数据。
- `vllm_classify.py`：本地 vLLM 推理。
- `clean_results.py`：结果清洗与统计。

## 推荐执行顺序
```bash
python scripts/build_manifest.py
python scripts/prepare_llm.py
python scripts/vllm_classify.py
python scripts/clean_results.py
```

## 常用环境变量
- `DATA_ROOT`：原始 `photos.csv*` 所在目录。
- `WORK_DIR`：中间产物与结果目录。
- `VLLM_MODEL`：本地模型目录。
- `PROJECT_ROOT`：仅在 `vllm_classify.py` 需要覆盖导入路径时使用。