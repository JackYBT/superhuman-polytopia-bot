"""Record a fixed, paired local matchup using search_arena; see docs/paired-matchups.md."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT_KEYS = ('candidate', 'opponent', 'value', 'opponent_value')
ARENA_KEYS = set(CHECKPOINT_KEYS) | {
    'size', 'map_mix', 'salt', 'lanes', 'depth', 'temperature', 'rollout_temperature',
    'max_turn_actions', 'root', 'margin', 'prior_coef', 'after_end', 'replies',
    'determinize', 'parallel', 'workers', 'opponent_search',
}
SEARCH_KEYS = ARENA_KEYS - set(CHECKPOINT_KEYS) - {'size', 'map_mix', 'salt', 'opponent_search', 'max_turn_actions'}
ENVIRONMENT_KEYS = ('JAX_PLATFORMS', 'XLA_FLAGS', 'EXTRA_XLA_FLAGS', 'OMP_NUM_THREADS',
                    'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_config(path: Path) -> tuple[dict, list[int]]:
    """Accept the existing Curio search config; relative checkpoint paths use cwd."""
    from polytopia_bot.ruleset import BOARD_SIZES

    cfg = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('arena config must be an object')
    cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
    maps, seed = cfg.pop('maps'), cfg.pop('seed_base')
    cfg.pop('name', None)
    if type(maps) is not int or type(seed) is not int or maps < 1 or seed < 0 or seed + maps >= 2 ** 31:
        raise ValueError('require positive maps and nonnegative seeds below 2**31 - 1')
    if unknown := cfg.keys() - ARENA_KEYS:
        raise ValueError(f'unknown arena settings: {sorted(unknown)}')
    cfg = {'size': 14, 'lanes': 1, 'salt': 0, 'max_turn_actions': 96, **cfg}
    if cfg['size'] not in BOARD_SIZES:
        raise ValueError(f'unsupported board size: {cfg["size"]}')
    opponent_search = cfg.get('opponent_search')
    if opponent_search is not None:
        if not isinstance(opponent_search, dict) or opponent_search.keys() - SEARCH_KEYS:
            raise ValueError('invalid opponent_search settings')
        if opponent_search and 'lanes' not in opponent_search:
            raise ValueError('opponent_search requires lanes')
    for key in CHECKPOINT_KEYS:
        if key in ('candidate', 'opponent') or cfg.get(key):
            cfg[key] = str(Path(cfg[key]).resolve(strict=True))
    return cfg, list(range(seed, seed + maps))


def checkpoint_receipts(cfg: dict) -> dict:
    from polytopia_bot.ruleset import compatible

    receipts = {}
    for role in CHECKPOINT_KEYS:
        if not cfg.get(role):
            continue
        path = Path(cfg[role])
        meta_path = path.with_suffix('.json')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        actual = digest(path)
        if actual != meta.get('sha256') or not compatible(meta):
            raise ValueError(f'{role} checkpoint integrity/rules mismatch: {path}')
        config = meta['config']
        if config.get('players', 2) != 2 or config.get('obs_version', 1) != 1 or config['size'] != cfg['size']:
            raise ValueError(f'{role} requires a two-seat, observation-v1 checkpoint at arena size')
        receipts[role] = dict(sha256=actual, metadata_sha256=digest(meta_path),
                              training_ruleset=meta.get('ruleset'),
                              training_rules_digest=meta.get('simulator_sha256'))
    return receipts


def provenance(size: int) -> dict:
    """Hash current checkout sources and installed binary separately; no parity claim."""
    import polytopia_engine_rs
    from polytopia_bot.ruleset import RULESET_ID, rules_digest, training_round_limit

    package = ROOT / 'src/polytopia_bot'
    source_paths = list(package.rglob('*.py')) + list(package.rglob('*.json'))
    source_paths += list((ROOT / 'engine-rs/src').rglob('*.rs'))
    source_paths += [ROOT / p for p in ('pyproject.toml', 'engine-rs/pyproject.toml',
                                       'engine-rs/Cargo.toml', 'engine-rs/Cargo.lock')]
    module_path = Path(polytopia_engine_rs.__file__).resolve()
    binaries = [p for p in [module_path, *module_path.parent.iterdir()]
                if p.is_file() and any(p.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES)]
    if not binaries:
        raise RuntimeError('cannot identify installed polytopia_engine_rs extension')
    return dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(source_paths)},
        rust_binary_sha256={p.name: digest(p) for p in sorted(set(binaries))},
        python=sys.version, platform=platform.platform(),
        packages={name: metadata.version(name) for name in ('jax', 'jaxlib', 'numpy', 'zstandard')},
        ruleset=RULESET_ID, rules_table_digest=rules_digest(), round_limit=training_round_limit(size),
    )


def validate_row(row: dict, expected: set[tuple[int, int]], seen: set[tuple[int, int]]) -> tuple[int, int]:
    """The candidate owns both reward and seat; never silently flip either."""
    key = (row.get('seed'), row.get('seat'))
    if any(type(v) is not int for v in key) or key not in expected or key in seen:
        raise ValueError(f'unexpected or duplicate game: {key}')
    reward = row.get('reward')
    if type(reward) not in (int, float) or reward not in (-1, 0, 1):
        raise ValueError(f'invalid reward for {key}: {reward}')
    if type(row.get('truncated')) is not bool or (row['truncated'] and reward != 0):
        raise ValueError(f'invalid truncation for {key}')
    return key


def paired_summary(rows: list[dict], seeds: list[int]) -> dict:
    from .search_arena import summarize

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('require distinct map seeds')
    expected = {(seed, seat) for seed in seeds for seat in (1, 2)}
    seen = set()
    for row in rows:
        seen.add(validate_row(row, expected, seen))
    if seen != expected:
        raise ValueError(f'incomplete paired roster: {len(seen)}/{len(expected)} games')
    # Pool arrival order must not change bootstrap resampling or its intervals.
    rows = sorted(rows, key=lambda row: (row['seed'], row['seat']))
    result = summarize(rows)
    result['perspective'] = 'candidate'
    result['by_seat'] = {str(seat): summarize([r for r in rows if r['seat'] == seat]) for seat in (1, 2)}
    return result


def evaluate(cfg: dict, seeds: list[int], output: Path, *, processes: int, timeout: float) -> dict:
    """Save partial evidence on failure; only a complete fixed roster gets a score."""
    if processes < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('processes and timeout must be positive and finite')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('require distinct map seeds')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    from . import search_arena

    weights = checkpoint_receipts(cfg)
    source = provenance(cfg['size'])
    items = [(seed, seat) for seed in seeds for seat in (1, 2)]
    expected, seen, rows = set(items), set(), []
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(), arena_config=cfg,
                    map_seeds=seeds, seats=[1, 2], games=len(items), processes=min(processes, len(items)),
                    timeout_seconds=timeout, checkpoints=weights, provenance=source,
                    environment_before_pool={key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
                    perspective='candidate (reward, seat, summary and search diagnostics)',
                    scoring='win=1, draw=0.5, loss=0; truncations are draws',
                    scope='Frozen simulator comparison; current retail rules and map parity are not established',
                    stopping='Fixed roster; fail on error/deadline; no retries or result-based stopping')
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'games.json', rows)
    started = time.monotonic()
    status = dict(status='running', completed_games=0, total_games=len(items))
    write_json(output / 'status.json', status)

    def save(row: dict) -> None:
        key = validate_row(row, expected, seen)
        # Serialize before accepting, so malformed/nonfinite worker output fails closed.
        write_json(output / 'games.json', [*rows, row])
        seen.add(key)
        rows.append(row)
        status.update(completed_games=len(rows), elapsed_seconds=time.monotonic() - started)
        write_json(output / 'status.json', status)

    try:
        search_arena.run(cfg, items, processes=processes, timeout=timeout, on_result=save)
        if checkpoint_receipts(cfg) != weights or provenance(cfg['size']) != source:
            raise ValueError('checkpoint or source changed during evaluation')
        summary = paired_summary(rows, seeds)
        result = dict(summary=summary, elapsed_seconds=time.monotonic() - started,
                      worker_environment={key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
                      games_sha256=digest(output / 'games.json'), protocol_sha256=digest(output / 'protocol.json'))
        write_json(output / 'complete.json', result)
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}',
                      elapsed_seconds=time.monotonic() - started)
        write_json(output / 'status.json', status)
        raise
    # The complete receipt is authoritative if a crash prevents the status refresh.
    status.update(status='complete', elapsed_seconds=result['elapsed_seconds'])
    write_json(output / 'status.json', status)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory; never overwritten')
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=10800, help='pool wall-time limit in seconds')
    args = parser.parse_args()
    cfg, seeds = load_config(args.cfg)
    result = evaluate(cfg, seeds, args.output.resolve(), processes=args.processes, timeout=args.timeout)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
