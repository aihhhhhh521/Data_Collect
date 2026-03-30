from pathlib import Path

# ========= 基本路径 =========
# 改成你自己的 Unsplash 数据集目录；也可通过环境变量 DATA_ROOT 覆盖
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data")).resolve()
WORK_DIR = Path(os.getenv("WORK_DIR", PROJECT_ROOT / "work")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ========= Local_Model 配置 =========
VLLM_MODEL = str(Path(os.getenv("VLLM_MODEL", PROJECT_ROOT.parent / "models" / "Qwen3.5-4B")).resolve())
VLLM_TEXT_MAX_CHARS = 600
VLLM_MAX_OUTPUT_TOKENS_PER_ITEM = 48

# 速度优先时建议关掉
VLLM_RETURN_REASON = False

# ========= 数据处理配置 =========
PARQUET_BATCH_SIZE = 20000
TEXT_MAX_CHARS = 1800

# ========= 五类标签 =========
LABELS = ["城市、建筑", "室内", "自然", "静物", "人像"]

# ========= 文件名 =========
MANIFEST_FILE = WORK_DIR / "manifest.parquet"
PRECLASSIFIED_FILE = WORK_DIR / "preclassified.parquet"
NEED_LLM_FILE = WORK_DIR / "need_llm.parquet"
VLLM_RESULTS_JSONL = WORK_DIR / "vllm_results.jsonl"
VLLM_RESULTS_FILE = WORK_DIR / "vllm_results.parquet"
CLASSIFIED_FILE = WORK_DIR / "classified.parquet"
NEED_REVIEW_FILE = WORK_DIR / "need_review.parquet"
STATS_FILE = WORK_DIR / "category_stats.csv"
