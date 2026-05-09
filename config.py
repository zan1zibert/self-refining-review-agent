import os
from dotenv import load_dotenv

load_dotenv()

# Model configuration
MODEL_PROVIDER = "anthropic"  # "openai" or "anthropic"
MODEL_NAME = "claude-3-5-sonnet-20240620"  # or "gpt-4o", etc.

# Experiment settings
DEFAULT_ITERATIONS = 12
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 4000

# Paths
INITIAL_SPEC = "review_agent.md"
DATA_DIR = "data"
RAW_DIR = f"{DATA_DIR}/raw"
PLOTS_DIR = f"{DATA_DIR}/plots"
METRICS_FILE = f"{DATA_DIR}/metrics.csv"

# Create directories
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)