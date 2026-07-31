# Agent Evaluation

All reported metrics must be generated from real game traces or deterministic
scenario execution. Do not enter estimated improvements as measured results.

## Run

```bash
python -m evals.run \
  --trace-dir logs/traces \
  --output reports/evaluation/baseline.json \
  --experiment-label structured-tools \
  --metadata '{"commit":"<git-sha>","model":"<model-id>"}'
```

Optional cost estimation accepts the provider's actual prices at experiment
time:

```bash
python -m evals.run \
  --input-price <price-per-million-input-tokens> \
  --output-price <price-per-million-output-tokens>
```

Zero is the default so the repository never embeds stale pricing.

## Metrics

- complete-game rate;
- first Tool Call legal rate;
- repair success rate after an invalid first call;
- deterministic fallback rate;
- LLM error count;
- P50/P95 model latency;
- input and output tokens;
- optional cost from explicitly supplied prices;
- final scores and winner counts.

Tool attempts are joined by the real `action_id` recorded in the trace.
Reports retain the exact game IDs, explicit pricing inputs, experiment label,
and caller-supplied metadata needed to reproduce a comparison.

## Intelligence score with LLM Judge

LLM Judge is opt-in because it makes additional provider calls. The normal
evaluation path remains offline and consumes no model tokens.

```bash
ANTHROPIC_API_KEY=... python -m evals.run \
  --trace-dir logs/traces \
  --game-id <32-character-game-id> \
  --judge-model anthropic:claude-sonnet-4-6 \
  --judge-max-actions 30 \
  --output reports/evaluation/judged.json
```

`--judge-max-actions` is a global cap across the batch and must be between
1 and 200. The report records the Judge provider/model and its input/output
tokens separately from gameplay usage.

For each `action_id`, the Judge sees only events available before the Decision
event. Targeted events and private chats belonging to other players are
excluded. Replay text is marked as untrusted data to reduce prompt-injection
risk.

The Judge returns:

- `decision_score` (1–10): state/rule understanding, evidence use,
  uncertainty, and risk/reward reasoning;
- `action_score` (1–10): consistency with the decision, tool legality,
  timing, information value, retries, and fallbacks;
- `intelligence_score` (0–100): equal-weight conversion of the two scores;
- confidence and a short user-safe reason.

Scores are reported per action, per Agent, per game, and across the batch.
Invalid Judge output is `unscored`; the evaluator never invents a replacement
score.

The same feature is available through MCP `start_evaluation` with
`judge_model` and `judge_max_actions`.

## Deterministic scenarios

Scenarios under `evals/scenarios/` cover:

- dynamic Tool Schema boundaries;
- unexpected arguments;
- replay ID validation and path traversal;
- secret redaction;
- prompt-injection detection;
- memory truth verification.

They consume no model tokens and run in CI.

## Ablation comparison

Run the same game/scenario population for baseline and candidate variants,
then compare the resulting reports:

```bash
python -m evals.compare \
  --baseline reports/evaluation/baseline.json \
  --candidate reports/evaluation/candidate.json \
  --output reports/evaluation/comparison.json
```

Recommended experiment labels:

- `free-text-actions` vs `structured-tools`;
- `memory-disabled` vs `structured-memory`;
- `single-stage` vs `decision-action-two-stage`.

Only claim an improvement when both reports come from comparable real runs.
The comparison tool reports descriptive deltas and does not claim statistical
significance.

## Results

Populate this section only after running a controlled experiment:

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| Complete-game rate | pending | pending | pending |
| First-call legal rate | pending | pending | pending |
| Repair success rate | pending | pending | pending |
| Fallback rate | pending | pending | pending |
| P95 latency | pending | pending | pending |
| Token use/game | pending | pending | pending |
