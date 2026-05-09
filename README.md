# Self-Refining Code Review Agent

An experiment: feed Claude a code-review-agent specification as a system prompt, ask it to rewrite that spec, then feed the rewrite back in as the next system prompt. Iterate. Measure what happens to length and content over time.

The goal isn't to build a great review agent — it's to study the dynamics of self-refinement: does the spec converge, oscillate, or drift?

## What we found so far

Across multiple runs we've seen:

- **True fixed points exist.** Once the spec settles, Claude rewrites it to byte-identical output for many iterations in a row.
- **There can be multiple attractors.** A run can sit on one fixed point for 15+ iterations, escape, briefly drift, then settle on a *different* fixed point.
- **Token count alone hides state.** Two different specs can have identical token counts. Embedding similarity is needed to see content changes the token-count plot misses.
- **The prompt wording matters a lot.** Without a "be concise" instruction, length grows monotonically until it hits `MAX_TOKENS`. With it, length stabilises in the 450–550 token range.

See `data/runs/<run_id>/plots/` for per-run charts.

## Repository layout

```
review_agent.md             # initial spec — the seed
prompts/
  refine_prompt.txt         # default user message: "produce an improved version"
  be_concise_prompt.txt     # variant: same task + "be concise and succinct"
config.py                   # model, iteration count, paths
run_experiment.py           # main orchestrator
diff_run.py                 # post-hoc analysis: detect plateaus, diff attractors
data/
  runs/
    <UTC-timestamp>/
      raw/review_agent_vNN.md   # spec at every iteration
      metrics.csv               # per-iteration tokens + similarities
      metadata.json             # model, hashes, git SHA, timing
      plots/
        token_evolution.png
        similarity.png
    latest -> <most recent run>
```

Each run is fully self-contained — you can compare runs without overwriting.

## Setup

Requires Python 3.11+ and an Anthropic API key.

```bash
git clone <this repo>
cd self-refining-review-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit, set ANTHROPIC_API_KEY
```

## Running an experiment

```bash
python run_experiment.py
```

Tunables live in `config.py`:

- `MODEL_NAME` — Anthropic model ID
- `ITERATIONS` — number of refine cycles
- `MAX_TOKENS` — output budget per iteration
- `FEEDBACK_PROMPT` — which user-message file to use (`prompts/refine_prompt.txt` or `prompts/be_concise_prompt.txt`)

Output lands in `data/runs/<timestamp>/`. The symlink `data/runs/latest` always points to the most recent run.

## Analysing a run

```bash
python diff_run.py                    # uses data/runs/latest
python diff_run.py <run_id>           # specific run
python diff_run.py <run_id> 11 28     # diff two specific iterations
```

The default invocation prints a plateau summary (which iteration ranges share identical content) and shows unified diffs between successive distinct attractors — the cleanest way to see what actually changed when the spec jumped from one fixed point to another.

## Reproducibility

`metadata.json` in each run captures everything that determines the output:

- model ID and SDK version
- iteration count and `MAX_TOKENS`
- SHA-256 of the seed spec and feedback prompt
- git SHA and dirty-tree flag
- start / finish timestamps (UTC)
- final status (`completed`, `halted:<reason>`, `error:<type>`)

If results look weird, check it.

## Caveats

- **Generation is non-deterministic.** A single trajectory is one sample path. To distinguish drift from noise, run several trajectories with the same seed and compare.
- **The embedder (`all-MiniLM-L6-v2`) truncates around 256 tokens.** Specs that exceed that have their tails ignored when computing similarity. For larger specs, swap in a long-context embedder.
- **The seed spec is intentionally thin.** Iteration 1 is closer to "write a spec from scratch" than "edit". If you care about pure refinement dynamics, start with a substantive seed.
