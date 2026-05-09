import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Model configuration
MODEL_NAME = "claude-opus-4-7"  # or "gpt-4o", etc.

# Experiment settings
ITERATIONS = 12
MAX_TOKENS = 4000
TOP_P = 0.95

# Paths
ROOT = Path(__file__).parent
INITIAL_SPEC = ROOT / "review_agent.md"
FEEDBACK_PROMPT = ROOT / "prompts/refine_prompt.txt"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PLOTS_DIR = DATA_DIR / "plots"
METRICS_FILE = DATA_DIR / "metrics.csv"

# Create directories
def ensure_dirs() -> None:
  os.makedirs(RAW_DIR, exist_ok=True)
  os.makedirs(PLOTS_DIR, exist_ok=True)