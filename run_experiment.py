import os
import time
import pandas as pd
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from openai import OpenAI
from anthropic import Anthropic

# Load embedding model for similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def read_spec(version: int) -> str:
    path = f"{RAW_DIR}/review_agent_v{version:02d}.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def save_spec(version: int, content: str):
    path = f"{RAW_DIR}/review_agent_v{version:02d}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ Saved version {version}")

def get_feedback(current_spec: str, previous_spec: str = "") -> str:
    with open("prompts/feedback_prompt.txt", "r") as f:
        template = f.read()
    
    prompt = template.format(
        current_spec=current_spec,
        previous_spec=previous_spec or "No previous version."
    )
    
    if MODEL_PROVIDER == "anthropic":
        client = Anthropic()
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=2000,
            temperature=DEFAULT_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    else:
        # OpenAI fallback
        client = OpenAI()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=DEFAULT_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

def refine_spec(current_spec: str, feedback: str) -> str:
    with open("prompts/refine_prompt.txt", "r") as f:
        template = f.read()
    
    prompt = template.format(current_spec=current_spec, feedback=feedback)
    
    if MODEL_PROVIDER == "anthropic":
        client = Anthropic()
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    else:
        client = OpenAI()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=DEFAULT_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

def main():
    print("🚀 Starting Self-Refining Code Review Agent Experiment\n")
    
    # Load initial spec
    with open(INITIAL_SPEC, "r", encoding="utf-8") as f:
        current = f.read()
    
    save_spec(1, current)
    
    records = []
    
    for i in range(1, DEFAULT_ITERATIONS + 1):
        print(f"\n--- Iteration {i} ---")
        
        prev = read_spec(i-1) if i > 1 else ""
        
        # 1. Get feedback
        print("Generating feedback...")
        feedback = get_feedback(current, prev)
        
        # 2. Refine
        print("Refining specification...")
        new_spec = refine_spec(current, feedback)
        
        # 3. Save and measure
        save_spec(i + 1, new_spec)
        
        # Metrics
        tokens = count_tokens(new_spec)
        embedding = embedder.encode([new_spec])[0]
        prev_embedding = embedder.encode([current])[0] if current else None
        similarity = cosine_similarity([embedding], [prev_embedding])[0][0] if prev_embedding is not None else 1.0
        
        records.append({
            "iteration": i + 1,
            "tokens": tokens,
            "similarity_to_prev": round(similarity, 4),
            "timestamp": datetime.now()
        })
        
        current = new_spec
        time.sleep(1)  # Be nice to the API
    
    # Save metrics
    df = pd.DataFrame(records)
    df.to_csv(METRICS_FILE, index=False)
    print(f"\n✅ Experiment completed! Metrics saved to {METRICS_FILE}")
    
    # Quick plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["iteration"], df["tokens"], marker='o', linewidth=2)
    plt.title("Token Count Evolution - Self-Refining Review Agent")
    plt.xlabel("Iteration")
    plt.ylabel("Token Count")
    plt.grid(True)
    plt.savefig(f"{PLOTS_DIR}/token_evolution.png")
    print(f"📊 Token evolution plot saved!")

if __name__ == "__main__":
    main()