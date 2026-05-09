# Self-Refining Code Review Agent

**I let a code review agent iteratively rewrite its own `.md` instruction file using Self-Refine loops. Here's what happened to token count, structure, and quality.**

This repo contains the full experiment: initial spec, orchestration script, raw versions, metrics, and plots.

## Key Findings (Coming in the Substack post)
- [Token evolution chart]
- Clear oscillation / convergence / divergence depending on temperature
- Final evolved review spec (steal this!)

## Repository Structure
- `review_agent.md` → Initial version
- `evolved_review_agent.md` → Best final version
- `run_experiment.py` → Main orchestrator
- `data/raw/` → All intermediate versions
- `data/plots/` → Generated charts

## Quick Start

```bash
git clone https://github.com/YOURUSERNAME/self-refining-review-agent.git
cd self-refining-review-agent
pip install -r requirements.txt
cp .env.example .env          # add your API keys
python run_experiment.py --iterations 12 --temperature 0.7

