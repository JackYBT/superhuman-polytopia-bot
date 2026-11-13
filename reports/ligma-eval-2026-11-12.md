# LIGMA-EVAL: Kevin's site bot (Ligma) vs the real PolytopAI bot, measured exactly like T3

**What is measured (META REVIEW-LIGMA 18:56 PT):** the *served Ligma weights and hierarchical chooser, adapted to the
frozen T3 public chain*. This is not a demonstrated behavioural replica of the website. The reference-pick parity
(§4) holds conditional on our inputs: the reference re-encodes our recorded public view and our recorded catalogue.
It does not validate the site's masks, observation or WASM. Known differences from the site:
- `detect` reflects public information only;
- diplomacy 18-22 is dropped (site mode 1 may offer it);
- mask normalisers and features can differ even where the kept action tuples match.

Results off 14x14 measure transfer.

Owner: Claude session T6-EVAL. Requested by Jack via infra 2026-11-12 18:1x PT ("I want Kevin's own bot measured
the same way"). Harness: `harness/game.py` **unchanged at SHA256 cf745477…c3d** (META scoped PASS 14:27 PT, fog
closure). Ligma enters only through `harness/ligma.py` (loader + chooser swapped in; the game loop, referee, public
chain, one-shot top-1 commit, refusal = INCOMPLETE and metrics are T3's). Status: **FINAL (report-only measurement)**. Headline: **Ligma
wins 13.9% (95% seed-cluster CI 11.3-16.4), 120/865 attempts**, vs PolytopAI policy-only. Spend: $4.38 of the $10 cap
(stop $8).

## 1. What Ligma is, and the exact pins (read-only; nothing written to Kevin's repo)

| Item | Pin |
|---|---|
| Served weights | `deploy-d14/web/public/bot/ligma.bin` (live site = external-website @ 76595018), SHA256 `63080388…2a44`, 7,889,900 B, PTB1 |
| Served metadata | `ligma.json`: source checkpoint `distributed/ov-h0-1107/final.npz` SHA256 `c9afc05d…bcd2` (Kevin's Modal volume `polytopia-structured-rl`, not readable from our workspace), stage update-5933, ruleset polytopia-2.17.3, simulator `70174d2b…48bc2`, 1,972,387 params |
| Architecture | normalized-v2 conv net, width 128, 4 residual blocks, 25 leaves, observation version 1, `no_resign` |
| Our checkpoint | `checkpoints/ligma/final.npz` SHA256 `32b3a84d…4da1` + `final.json`, rebuilt by `ligma_import.py` from the served bin. **Round trip: Kevin's own exporter (`scripts/export_browser_weights.py`) re-exports our npz to bytes identical to the served `ligma.bin`** |
| Inference code (reference) | kevinz917/polytopia-bot @ `0812f66a72217d786368485d59b13fb5c1220565` `src/polytopia_bot` — the commit Kevin's `scripts/verify_browser_models.py` pins as the JAX reference the site's WASM is checked against; extracted read-only with `git archive` to `ref/ligma-0812f66a` |
| Site chooser | engine-wasm (`ux/t3-engine-gen` a6685999, live wasm `d5525b0c…ffde`) `Policy::act_mode`: the site has no `trained` block for Ligma, so it plays **mode 1 = hierarchical argmax** (`Draw::Greedy` = `network.sample(deterministic=True)`: argmax family, then source, then target, first maximum), candidates from `e.legal(public_state(s, player))`, resignation masked, **action layout 2** ("the site keeps action layout 2") |
| Training | 2.17.3 rules, 14x14 only, no-peace rules, random map types. The site plays it on 14x14 only |

## 2. Hidden information

- **Site (as served):** candidates come from the mover's public state (`public::public_state`; GEN audit d06602ae).
  Features come from engine-rs `kernels::observe(true state, player)` with the player's fog. One feature is not
  clearly public: **`detect`**, which flags an own unit adjacent to an invisible foreign unit (Cloak; `hidden_from`)
  and is computed from the true state. Ligma's v1 encoder reads it as a board channel
  (0812f66a `observation.py:174`). Whether retail Polytopia shows this to the player is not established here.
  Reported, not tested.
- **Here:** Ligma never sees the referee state. Its input is the T3 chain's public reconstruction (converter → T3
  adapter `public.project` → `T3LiveBuilder` → pinned obs6 native view), encoded at version 1. Invisible enemy units
  are not in that reconstruction, so `detect` can only reflect public information: it is 0 in all 899 recorded Mac
  decisions. Candidates are the same public catalogue T3 uses.

## 3. Adapter (`harness/ligma.py`, `ligma_import.py`)

- Loader: `polytopia_bot.rl.runner.load_checkpoint` (sha + ruleset check) on the 14x14 template. The conv trunk and
  heads do not depend on board size, and `parameter_embedding` is read only through its first 128 rows. So the same
  weights run on 11-20 (`network.grow` docstring). **Ligma never trained off 14x14; per-size results off 14 measure
  transfer.**
- Encoder: the pinned T3 encoder (`visibility-v2/runtime/.../observation.py`) at `obs_version=1`.
- Catalogue: the T3 chain's public catalogue, restricted to layout 2. Recover-family targets 2-8 (FreezeArea,
  BreakIce, Canal, ClearFlood, ClearAlgae, Explode, Swarm), which the site's layout 2 lacks, are dropped. Resign is
  masked. Diplomacy kind-3 params 18-22 are dropped as in T3's chain. The site does not drop them in mode 1, but
  Ligma trained under no-peace rules. Layouts are append-only, so every kept entry has the same logits as on the site.
- Chooser: hierarchical argmax (not T3's joint top-1). `game.OurPlayer.decide` commits that one pick once; an
  unmappable/refused pick ends the game INCOMPLETE (unchanged).
- Top-10 diagnostics: the rest of the served catalogue by joint probability (diagnostic only).
- Modal: `modal_panel.play` calls `ligma.install()` when `spec['actor'] == 'ligma'`; identity now also hashes
  `/t6/checkpoints/**`.
- Panel `panels/ligma-v1.json`: the 1,000 t3-base-v2 games (same size/seed/map/opponent tribe/seat per game_id),
  spec hash `1acbfb64…9097`. `launch.py --interleave` spawns seed index by seed index across sizes, so any stop trims
  evenly. `--seeds-per-size K` is the even pre-launch trim.

## 4. Qualification (step 2, $0; Mac, `qual-ligma/`)

10 games from the frozen panel (2 per size, 6 map types represented, both seats; same game_ids as t3-base-v2),
played twice by `ligma_local.py` on the Mac with the live T3 interpreter/env (`.venv-gen`, 1 thread):

| Check | Result |
|---|---|
| Weights | rebuilt npz re-exports byte-identically to the served `ligma.bin` (Kevin's exporter, unchanged) |
| Reference picks (conditional on our recorded view + catalogue) | **4,826 / 4,826 decisions identical** to Kevin's pinned inference code (0812f66a: its `encode_view` v1, its layout-2 masks, its `network.sample(deterministic=True)`), all 5 sizes; encoded features max abs difference **0.0** (`ligma_parity.py`, separate process, only 0812f66a's package importable) |
| Determinism | 10/10 games bit-identical across two runs (`move_sha256`) |
| Completion | 10/10 complete, 0 INCOMPLETE, 0 crashes; 2/10 Ligma wins (not a result: n=10) |
| Legal-set fidelity | 4,826 decisions: 12 differ (20 ours-only entries), all 12 FOG-explained, 0 unexplained |
| Speed (Mac) | 34-352 s/game, mean 118 s (T3 on Modal: 5.4 min/game mean) |
| Linux rebuild parity | **gate 0 of the paid run** (`gate0.py`, exits non-zero unless every condition holds): exactly the 10 frozen game_ids, each complete/valid/actor ligma, each bound to this checkout's harness/checkpoint hashes, and Linux `move_sha256` = Mac 10/10 (both Mac runs agree). `launch.py` refuses any other Ligma launch without a GATE0-PASS.json that certifies the current checkout. **PASS 19:3x PT**: 10/10 move hashes + exact identity, certificate `qual-ligma/gate0/GATE0-PASS.json`, $0.08 |
| Not done | the site's WASM was not run on our positions; WASM ↔ 0812f66a parity is Kevin's own claim (`verify_browser_models.py`, result file not in our copy) |

### Paid-run guards (META REVIEW-LIGMA 18:52/18:56/18:59 PT, fixed at $0; `test_preflight.py` 39/39 PASS)

- **One bound app identity.** `T6_APP=claude-t6-eval-ligma`, `T6_MAX_CONTAINERS=50` and `T6_TIMEOUT_S=1800` drive
  the deploy and are baked into the image env; launch.py and ops.py read them too.
- **Deployed read-back with exact identity.**
  - Before spawning, launch.py calls the deployed `probe()`.
  - It refuses unless app, container limit and timeout match, and the deployed identity holds exactly the expected
    keys and values: every `harness/*.py`, every `checkpoints/**` (npz and metadata), the deployed Modal entry
    source `modal_panel.py` itself, and `live-pins.json`. Omissions, substitutions and extra files are refused.
  - The read-back is stored in `runs/<run>/binding.json`.
- **Entry guards in `play`.** A Ligma spec, or any spec on a non-default app, must carry `app` equal to the running
  app and a finite `not_after_unix`, or the call raises before playing. A call starting after its deadline plays
  nothing.
- **Enforced spend envelope, independent of the supervisor (`envelope.py`).**
  - Worst case of a launch = min(containers, calls) x (W + timeout + 5 min) x $0.085/h.
  - launch.py refuses unless prior + $0.5 build + $0.02 per probe + this worst case < $8.
  - prior = max(billed for the app, every earlier launch's recorded worst case). Earlier launches are assumed to
    have spent their full worst case even if billing hasn't posted.
  - W, timeout and container count must be finite and positive; NaN, infinity and 0 are refused.
  - Plan: gate 0 = 10 calls, W 15 min, worst case $0.71. Main = 1,000 calls at 50 containers, W ≈ 55 min (launch
    prints the maximum allowed window), total ≤ $8 standalone of the supervisor.
  - Interleaved spawning trims every size evenly if calls expire.
- **Fail-closed supervisor.**
  - Polled container-hours are never reset within one supervisor process (the counter did reset 56.778 → 0 when
    the supervisor was restarted at 21:35 PT after the tool's 2 h background limit killed it; the reset is kept in
    `qual-ligma/main/ledger.jsonl`, so that counter is not restart-monotone evidence; the static envelope bound
    $7.6233 < $8 held independently and billing posted $4.38): projection = max(billed, polled container-h x $0.085 + $0.5) plus a
    lag term for live tasks.
  - Task-telemetry or app-list failure twice stops the app (from the very first poll), as does billing or volume
    failure twice, and any unexpected error.
  - The run stops at its deadline.
  - Stop is retried 5 times and must verify 0 containers, otherwise it logs `stop_failed` and exits non-zero.
- **Scoped Ligma install.** `ligma.installed()` restores T3's loader and chooser in `finally` and refuses a nested
  or leaked swap. A Ligma → T3 → Ligma(exception) → T3 sequence is tested. Unknown actors are rejected.
- **Strict gate 0 (`gate0.py`).** Exactly the 10 frozen game_ids, each complete/valid/actor ligma, each with the
  exact identity above, and Linux `move_sha256` = Mac 10/10. Otherwise it exits non-zero, and launch.py refuses the
  main panel without its PASS file for the current checkout. `ligma_parity.py` now also exits non-zero on zero
  decisions, any mismatch, or a non-zero feature difference.
- **Recheck.** The smoke game replays with the same `move_sha256` (8f4902f2…).

## 5. Results (step 3)

**Run.** Gate 0 (Linux parity) PASSed 10/10 at 19:28 PT: complete, exact identity, `move_sha256` = Mac. Evidence is
in `qual-ligma/gate0`; it billed $0.08. The main panel launched at 19:35 PT on the same checkout. The 990 non-gate
games plus the 10 gate rows make up the panel. Spawning was interleaved by seed index. Settings: 30 containers,
30-min game timeout, start window W = 115 min. The launch-enforced envelope was $7.62, the same total as META's
reviewed C50/W55 plan. I changed it because at Modal's 4.1 min/game, C50/W55 could start only ~2/3 of the panel;
META and infra were told at launch.

The app stopped at 21:38 PT with 0 containers verified. Posted billing for app `claude-t6-eval-ligma` is
**$4.384816**; ledger row `t6-eval-ligma-panel` is settled.

**Coverage (1,000 frozen games):**
- 863 result rows.
- 2 games hit the 30-min timeout (18-continents 7018011-s1 and 7018032-s1). They count as attempts, not won.
- 135 never started before the window closed (137 games lack a result row = these 135 + the 2 started timeouts).
  133 of them are the tail, seed indices ≥ 86/87 in every size (26-28
  per size): an even trim.
- The other 2, 14-lakes 7014029-s1 and 18-lakes 7018030-s1, returned "expired" despite early queue positions. Their
  inputs likely restarted after the deadline, e.g. after a container preemption.
- Missing games are missing, not losses.

**Coverage by cut (actual; `qual-ligma/main/coverage.md`):**

| size | panel | result rows | timeout (started) | never started (expired) |
|---|---|---|---|---|
| 11 | 200 | 172 | 0 | 28 |
| 14 | 200 | 172 | 0 | 28 |
| 16 | 200 | 174 | 0 | 26 |
| 18 | 200 | 171 | 2 | 27 |
| 20 | 200 | 174 | 0 | 26 |

| map | panel | result rows | timeout (started) | never started (expired) |
|---|---|---|---|---|
| archipelago | 42 | 38 | 0 | 4 |
| continents | 236 | 202 | 2 | 32 |
| dryland | 210 | 174 | 0 | 36 |
| lakes | 396 | 339 | 0 | 57 |
| pangea | 44 | 44 | 0 | 0 |
| waterworld | 72 | 66 | 0 | 6 |

| seat | panel | result rows | timeout (started) | never started (expired) |
|---|---|---|---|---|
| 1 | 500 | 430 | 2 | 68 |
| 2 | 500 | 433 | 0 | 67 |

**Outcomes:** 831 complete (none capped), 31 INCOMPLETE (top-1 not mappable: 29 fog-explained, 2 unexplained; not
won, as the fixed harness requires), 1 crash (16-lakes 7016079-s1: the public-chain converter raised `Unsupported:
own passenger shape outside engine rules`; invalid, not won), and the 2 timeouts.

**Primary statistic (attempt denominator: incomplete/crash/timeout = not won): 13.9% (11.3-16.4), 120/865, 434
seed clusters.** As played (831 valid): 14.4% (11.8-17.1); design effect 1.25.

| Size | Ligma (attempt denominator) |
|---|---|
| 11 | 28.5% (21.0-36.0) 49/172 |
| 14 (Ligma's training size) | 18.0% (11.4-24.7) 31/172 |
| 16 | 14.4% (8.8-19.9) 25/174 |
| 18 | 4.6% (1.6-7.7) 8/173 |
| 20 | 4.0% (1.1-6.9) 7/174 |

Wins: 83 by capital capture, 37 by elimination. Losses: 583 by capital, 128 by elimination.

**Legal-set fidelity:** 392,967 Ligma decisions, of which 24,253 differ from the referee (822 fog, 23,431
unexplained).
- Most unexplained entries are referee moves that Ligma's served action layout 2 lacks by design: BreakIce (type 17;
  1,875 in the kept examples) and ClearTileEffect (102; 1,721). The site's catalogue lacks them too. They restrict
  Ligma's options; they cannot make a game fail.
- The remainder are the public chain's known mapping gaps, also seen in T3's runs (T3 baseline: 8,027 unexplained
  decisions): build param 48 vs referee type 3 (1,414 / 651), and some move/kind-8 entries.
- Sensitivity, excluding every game with any unexplained difference: 14.4% (11.5-17.3), 99/689.

## 6. Side by side with the T3 report-only baseline

**Caveat:** T3's numbers come from the pre-fix fog-skip harness 19ac863a (28 assisted games, counted as not won here)
over 800 of the 1,000 games; 11/14 seeds 50-99 were cancelled. They are descriptive context, **not a clean paired
strength comparison**. Ligma's row is the served weights and chooser adapted to the T3 public chain (see the top of
this file), not a website replica. Off-14 sizes measure transfer. Both columns count incomplete/assisted/invalid rows as not won, with
95% seed-cluster CIs; except where a row says "attempt estimand", Ligma cells are over result rows only (size 18 and
seat 1 each exclude the 2 timeouts) (`runs/ligma-main/compare.md`, `ligma_compare.py`).

| Cut | Ligma (**result rows only**, 863; the 2 started timeouts are not in these rows; 95% seed-cluster CI) | T3 report-only (pre-fix, assisted = not won) |
|---|---|---|
| **All, attempt estimand (headline; rows + 2 timeouts)** | **13.9% (11.3-16.4) 120/865** | 28.6% (25.2-32.0) 229/800 |
| All, result rows only | 13.9% (11.3-16.5) 120/863 | 28.6% (25.2-32.0) 229/800 |
| Same 718 game_ids, result rows in both | 11.7% (9.1-14.3) 84/718 | 29.0% (25.3-32.6) 208/718 |
| Same 720 game_ids, attempt estimand (adds the 2 Ligma timeouts; descriptive, no CI computed) | 11.7% 84/720 | 29.2% 210/720 |
| size 11 | 28.5% (21.0-36.0) 49/172 | 50.0% (39.2-60.8) 50/100 |
| size 14 | 18.0% (11.4-24.7) 31/172 | 33.0% (22.3-43.7) 33/100 |
| size 16 | 14.4% (8.8-19.9) 25/174 | 31.5% (24.3-38.7) 63/200 |
| size 18 | 4.7% (1.6-7.8) 8/171 | 20.0% (14.3-25.7) 40/200 |
| size 20 | 4.0% (1.1-6.9) 7/174 | 21.5% (15.9-27.1) 43/200 |
| map archipelago | 10.5% (1.1-19.9) 4/38 | 37.5% (18.9-56.1) 15/40 |
| map continents | 3.5% (0.6-6.3) 7/202 | 16.8% (12.1-21.5) 38/226 |
| map dryland | 25.3% (18.0-32.6) 44/174 | 47.0% (39.0-54.9) 78/166 |
| map lakes | 11.5% (7.9-15.1) 39/339 | 22.8% (17.6-27.9) 66/290 |
| map pangea | 36.4% (19.1-53.6) 16/44 | 55.6% (38.0-73.1) 20/36 |
| map waterworld | 15.2% (6.1-24.2) 10/66 | 28.6% (11.2-45.9) 12/42 |
| seat 1 | 14.7% (11.3-18.0) 63/430 | 34.0% (29.4-38.6) 136/400 |
| seat 2 | 13.2% (10.0-16.4) 57/433 | 23.2% (19.1-27.4) 93/400 |

The same-game_ids rows are the closest like-for-like cut: 718 result rows in both, Ligma 11.7% vs T3 29.0%;
with the 2 Ligma timeouts as attempts, 84/720 = 11.7% vs 210/720 = 29.2%. Still not a clean paired comparison (T3 is pre-fix).

**Behaviour** (mean per game over valid rows; per-turn values are means over own turn ends):

| Side (mean per game) | Roads | Recover | Units <=3 of enemy capital | Units <=2 of own capital | Unspent stars | Own turns |
|---|---|---|---|---|---|---|
| Ligma | 0.68 | 2.80 | 0.60 | 2.72 | 2.9 | 23.2 |
| PolytopAI vs Ligma | 17.46 | 18.67 | 1.71 | 2.44 | 2.3 | 22.9 |
| T3 | 0.49 | 7.84 | 0.83 | 2.39 | 4.3 | 25.4 |
| PolytopAI vs T3 | 19.51 | 25.81 | 1.38 | 2.60 | 2.2 | 25.2 |

Ligma, like T3, builds almost no roads and uses Recover far less than PolytopAI. It keeps fewer units near the enemy
capital (0.60 vs T3 0.83 vs PolytopAI 1.71), keeps slightly more at home (2.72), and leaves fewer stars unspent than
T3 (2.9 vs 4.3).

## 7. Re-running T3 on the fixed harness (NOT run; needs infra's go)

The T3 baseline in §6 ran on the pre-fix harness 19ac863a (fog skip; 28 assisted games) and covers 800 of the 1,000
panel games (11/14 seeds 50-99 cancelled). A clean T3 row on cf745477 over the same 1,000 games costs **about $6-6.5
and ~1 h**. That figure is measured T3 container time per size (11: 1.5, 14: 2.5, 16: 4.3, 18: 5.8, 20: 9.5 min/game)
x 200 games per size = 79 container-h, at the measured ~$0.076/container-h, plus a redeploy. Not run. Under the same
enforced envelope (worst case < $8), a complete 1,000-game T3 panel needs ~79 container-h. At most ~75 container-h
fit in the window, so expect ~90-95% coverage with an even trim, or a second funded launch for the rest.
