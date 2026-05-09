import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
from anthropic import Anthropic
from typing import Tuple
from config import (
  MODEL_NAME,
  MAX_TOKENS,
  ITERATIONS,
  INITIAL_SPEC,
  FEEDBACK_PROMPT,
  RAW_DIR,
  PLOTS_DIR,
  METRICS_FILE,
  ensure_dirs
)

# embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = Anthropic()

def load_current_spec() -> str:
    with open(INITIAL_SPEC, "r", encoding="utf-8") as f:
        return f.read()

def load_feedback_prompt() -> str:
    with open(FEEDBACK_PROMPT, "r", encoding="utf-8") as f:
      return f.read()

def save_version(version: int, content: str, output_tokens):
    path = f"{RAW_DIR}/review_agent_v{version:02d}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Saved version {version} ({output_tokens} tokens)")
    
def get_token_count(current_spec: str) -> int:
  token_count = client.messages.count_tokens(
    messages=[{
      "role": "user",
      "content": current_spec
    }]
  )
  
  return token_count.input_tokens
  

def self_refine_iteration(current_spec: str, feedback_prompt: str) -> Tuple[str, int, str]:
    # The entire review_agent.md becomes the SYSTEM PROMPT
    system_prompt = current_spec

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": feedback_prompt}]
    )

    new_spec = response.content[0].text.strip()
    output_tokens = response.usage.output_tokens
    stop_reason = response.stop_reason
    
    return new_spec, output_tokens, stop_reason

def main():
    print("🚀 Starting Self-Refining Review Agent Experiment (Claude as full agent)\n")
    
    ensure_dirs()
    
    feedback_prompt = load_feedback_prompt()
    
    current_spec = load_current_spec()
    
    initial_token_count = get_token_count(current_spec)
    
    save_version(0, current_spec, initial_token_count)
    
    records = []
    
    for i in range(1, ITERATIONS + 1):
        print(f"\n--- Iteration {i} ---")
        
        print("Calling Claude with full spec as system prompt...")
        new_spec, output_tokens, stop_reason = self_refine_iteration(current_spec, feedback_prompt)
        
        # Save the new version
        save_version(i , new_spec, output_tokens)
        
        # Metrics
        # embedding = embedder.encode([new_spec])[0]
        # prev_embedding = embedder.encode([current_spec])[0]
        # similarity = cosine_similarity([embedding], [prev_embedding])[0][0]
        
        records.append({
            "iteration": i,
            "tokens": output_tokens,
            # "similarity_to_prev": round(similarity, 4),
            "timestamp": datetime.now()
        })
        
        current_spec = new_spec
        
        print(f"\n--- Iteration {i} Finished with stop reason: {stop_reason} ---")
        time.sleep(1.5)  # Rate limit safety
    
    # Save metrics
    df = pd.DataFrame(records)
    df.to_csv(METRICS_FILE, index=False)
    print(f"\n✅ Experiment finished! Metrics saved → {METRICS_FILE}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["tokens"], marker='o', linewidth=2.5)
    plt.title("Token Count Evolution – Self-Refining Review Agent")
    plt.xlabel("Iteration")
    plt.ylabel("Tokens")
    plt.grid(True)
    plt.savefig(f"{PLOTS_DIR}/token_evolution.png", dpi=200)
    print(f"📊 Plot saved → {PLOTS_DIR}/token_evolution.png")

if __name__ == "__main__":
    main()
