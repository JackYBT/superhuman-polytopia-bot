# Reproducible local checkpoint matchups

`python -m polytopia_bot.rl.matchup` records a fixed two-player matchup on CPU.
Each map seed is played twice, with the candidate in each seat. It uses the
existing `search_arena` game loop and paired-map bootstrap; there is no training,
cloud launcher, automatic retry, or result-based stopping.

The existing `experiments/curio/arena.py search` remains useful for a single
aggregate result. This wrapper adds checkpoint/source receipts, incremental game
records, an execution deadline and exact roster validation before reporting a score.

## Run a small reproduction

From a Git checkout, with Python 3.11+ and Rust installed:

```sh
uv venv
source .venv/bin/activate
uv pip install -e '.[test]' 'maturin>=1.7,<2'
(cd engine-rs && maturin develop --release)
mkdir -p runs
cat > runs/hero-champion.json <<'JSON'
{
  "candidate": "weights/hero-1v1/final.npz",
  "opponent": "weights/champion/final.npz",
  "maps": 2,
  "seed_base": 916380,
  "size": 14,
  "map_mix": "classic",
  "salt": 916406,
  "lanes": 1
}
JSON
python -m polytopia_bot.rl.matchup --cfg runs/hero-champion.json \
  --output runs/hero-champion-smoke --processes 2 --timeout 600
```

These four games check the workflow, not model strength. For a comparison, choose
the number of maps and unused seeds **before** seeing outcomes. Keep all settings
and checkpoint bytes fixed. Use a new output directory for every run; an existing
directory is never overwritten. Relative checkpoint paths are resolved from the
working directory, as in the Curio runner. Rebuild the Rust extension when changing
engine source; recording both hashes does not prove that one was built from the other.

The config uses the Curio search format (`maps`, `seed_base`, optional `name`, plus
`search_arena` settings). Keys starting with `_` are comments; other unknown keys
are rejected. Defaults are size 14, one lane, salt 0 and 96 actions per planned
turn chunk. The arena can replan longer turns; that chunk limit is not a game cap.
The runner checks checkpoint and metadata hashes and accepts two-seat,
observation-version-1 checkpoints whose declared board size matches the arena.
This is the observation path supported by `search_arena`, not every released model.

For a searching candidate, set `lanes`, `depth`, `replies`, `rollout_temperature`
and `determinize` explicitly. For example Champion supports its released value
head with `lanes=16`, `depth=1`, `replies=64`, `rollout_temperature=1.0` and
`determinize=true`. Hero's released raw policy has no trained search value head;
do not interpret untrained value-head search as its intended playing strength.
The existing `value`, `opponent_search` and `opponent_value` settings are supported.
Search diagnostics in the output cover the **candidate only**, even if both search.

## Read the evidence

| File | Meaning |
| --- | --- |
| `protocol.json` | Fixed config, seeds, candidate perspective, checkpoint/metadata hashes, source hashes, installed Rust binary, versions and rules-table digest |
| `games.json` | Atomically replaced after each accepted game, in completion order; `seat` and `reward` always belong to the candidate |
| `status.json` | Running, complete or failed, with accepted-game count and any error |
| `complete.json` | Written only after every expected seed/seat is present once, inputs are rechecked and summary succeeds; includes hashes of protocol and games |

The timeout covers pool startup, compilation and game collection. Result-saving
callbacks are cooperative: the deadline is checked again after each callback
returns. Preflight, final hashing and cleanup can add time. An exception, timeout or keyboard interrupt
leaves partial evidence and a failed status, with no complete score. A hard kill
or power loss can leave the last status marked running; the absence of
`complete.json` means the run is incomplete. There is no automatic resume.
The complete receipt is written before the final status refresh and is authoritative
if that refresh is interrupted. Check its protocol/game hashes when auditing a run.

`win_rate = wins / games`; `score = (wins + 0.5 * draws) / games`.
`draws` counts zero rewards; `truncations` counts explicit harness-cap flags and is
reported separately. The shared arena previously labeled every draw a truncation;
consumers displaying W–L–D should now use `draws`. The 95% intervals resample map
pairs together (4,000 bootstrap samples, RNG seed 0), not individual games. They
describe these maps and frozen checkpoints, not training-seed uncertainty. Per-seat
summaries expose first/second-seat effects. Arrival order does not change the summary.

## Scope and attribution

These are frozen-simulator results. The protocol records the actual rules target
and round cap; neither a ruleset label nor a matching table digest establishes
current-retail parity. At public base `b00d634`, the target is `polytopia-2.17.3`
and the 14×14 harness cap is 200 rounds. `classic` uses that checkout's Rust map
generator. `proposed-v1` includes custom research presets such as Jungle,
Highlands and Ridge; familiar map names also do not prove official generation.
Do not compare results across engine/map revisions as if conditions were identical.

Search inherits the existing belief model and true-state fallback behavior. The
summary reports candidate `fallbacks`, `incomplete_lanes` and override rate.
Zero fallbacks alone does not establish an information-leak-free comparison.
Neither this wrapper nor its validation changes or certifies engine/search fidelity.

The game loop, search and bootstrap are Kevin Z's public Apache-2.0 code
(`search_arena.py`, introduced in `85557b5`); the config convention comes from his
Curio launcher, publicly released in `dcc0d97`. JackYBT contributed the durable
wrapper and reporting checks after frozen Champion/Hero comparisons. See the
repository's `LICENSE` and `THIRD_PARTY_NOTICES.md`. This contribution adds only
code, synthetic tests and documentation; it adds no checkpoints, private research,
commercial game assets or decompiled source.

Validation:

```sh
python -m pytest tests/test_search_arena.py tests/test_matchup.py tests/test_search.py -q
```
