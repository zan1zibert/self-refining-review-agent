import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import anthropic
import matplotlib.pyplot as plt
import pandas as pd
from anthropic import Anthropic
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

from config import (
    FEEDBACK_PROMPT,
    INITIAL_SPEC,
    ITERATIONS,
    MAX_TOKENS,
    MODEL_NAME,
    new_run_dir,
)

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = Anthropic()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def save_version(raw_dir: Path, version: int, content: str, output_tokens: int) -> None:
    path = raw_dir / f"review_agent_v{version:02d}.md"
    path.write_text(content, encoding="utf-8")
    print(f"✓ Saved version {version} ({output_tokens} tokens)")


def get_token_count(spec: str) -> int:
    token_count = client.messages.count_tokens(
        messages=[{"role": "user", "content": spec}],
        model=MODEL_NAME,
    )
    return token_count.input_tokens


def self_refine_iteration(current_spec: str, feedback_prompt: str) -> Tuple[str, int, str]:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=current_spec,
        messages=[{"role": "user", "content": feedback_prompt}],
    )
    new_spec = response.content[0].text.strip()
    return new_spec, response.usage.output_tokens, response.stop_reason


def write_metadata(run_dir: Path, started_at: str, finished_at: str, status: str) -> None:
    metadata = {
        "run_id": run_dir.name,
        "model": MODEL_NAME,
        "max_tokens": MAX_TOKENS,
        "iterations": ITERATIONS,
        "seed_spec_path": str(INITIAL_SPEC.relative_to(INITIAL_SPEC.parent.parent)),
        "seed_spec_sha256": sha256_of(INITIAL_SPEC),
        "feedback_prompt_path": str(FEEDBACK_PROMPT.relative_to(FEEDBACK_PROMPT.parent.parent)),
        "feedback_prompt_sha256": sha256_of(FEEDBACK_PROMPT),
        "anthropic_sdk_version": anthropic.__version__,
        "git": git_info(),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    print("🚀 Starting Self-Refining Review Agent Experiment\n")

    run_dir = new_run_dir()
    raw_dir = run_dir / "raw"
    plots_dir = run_dir / "plots"
    metrics_file = run_dir / "metrics.csv"
    print(f"📁 Run directory: {run_dir}\n")

    started_at = datetime.now(timezone.utc).isoformat()
    write_metadata(run_dir, started_at, finished_at="", status="running")

    feedback_prompt = FEEDBACK_PROMPT.read_text(encoding="utf-8")
    current_spec = INITIAL_SPEC.read_text(encoding="utf-8")

    initial_token_count = get_token_count(current_spec)
    save_version(raw_dir, 0, current_spec, initial_token_count)

    fieldnames = ["iteration", "tokens", "stop_reason", "timestamp"]
    with metrics_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        status = "completed"
        try:
            for i in range(1, ITERATIONS + 1):
                print(f"\n--- Iteration {i} ---")
                new_spec, output_tokens, stop_reason = self_refine_iteration(
                    current_spec, feedback_prompt
                )
                save_version(raw_dir, i, new_spec, output_tokens)

                # Metrics
                # embedding = embedder.encode([new_spec])[0]
                # prev_embedding = embedder.encode([current_spec])[0]
                # similarity = cosine_similarity([embedding], [prev_embedding])[0][0]

                writer.writerow({
                    "iteration": i,
                    "tokens": output_tokens,
                    "stop_reason": stop_reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                f.flush()

                current_spec = new_spec
                print(f"--- Iteration {i} stop_reason={stop_reason} ---")

                if stop_reason != "end_turn":
                    print(f"Stop reason {stop_reason!r} is not end_turn — ending experiment")
                    status = f"halted:{stop_reason}"
                    break

                time.sleep(1.5)
        except Exception as e:
            status = f"error:{type(e).__name__}"
            write_metadata(run_dir, started_at, datetime.now(timezone.utc).isoformat(), status)
            raise

    finished_at = datetime.now(timezone.utc).isoformat()
    write_metadata(run_dir, started_at, finished_at, status)
    print(f"\n✅ Metrics → {metrics_file}")

    df = pd.read_csv(metrics_file)
    if not df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df["iteration"], df["tokens"], marker="o", linewidth=2.5)
        plt.title("Token Count Evolution – Self-Refining Review Agent")
        plt.xlabel("Iteration")
        plt.ylabel("Output tokens")
        plt.grid(True)
        plot_path = plots_dir / "token_evolution.png"
        plt.savefig(plot_path, dpi=200)
        print(f"📊 Plot → {plot_path}")


if __name__ == "__main__":
    main()
