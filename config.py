from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Model configuration
MODEL_NAME = "claude-sonnet-4-6"

# Experiment settings
ITERATIONS = 50
MAX_TOKENS = 12000

# Paths
ROOT = Path(__file__).parent
INITIAL_SPEC = ROOT / "review_agent.md"
FEEDBACK_PROMPT = ROOT / "prompts/be_concise_prompt.txt"
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"


def new_run_dir() -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    run_dir = RUNS_DIR / run_id
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    _update_latest_symlink(run_dir)
    return run_dir


def _update_latest_symlink(run_dir: Path) -> None:
    latest = RUNS_DIR / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)
    except OSError:
        pass
