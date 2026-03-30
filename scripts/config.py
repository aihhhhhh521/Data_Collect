from pathlib import Path

# ========= 基本路径 =========
# 改成你自己的 Unsplash 数据集目录
DATA_ROOT = Path(r"Your_Dataset_Catalog").resolve()
WORK_DIR = "./work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# ========= Local_Model 配置 =========
VLLM_BASE_URL = "http://localhost:11434/api"
VLLM_MODEL = "./models/Qwen3.5-4B"   # 可改成你本机已经 pull 好的模型
VLLM_KEEP_ALIVE = "30m"
VLLM_READ_BATCH_SIZE = 10000
VLLM_RECORDS_PER_REQUEST = 16
VLLM_MAX_IN_FLIGHT_BATCHES = 16
VLLM_WRITE_BUFFER_SIZE = 1000
VLLM_TEXT_MAX_CHARS = 600
VLLM_MAX_OUTPUT_TOKENS_PER_ITEM = 48
VLLM_OPTIONS = {}

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
