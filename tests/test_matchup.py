"""Fixed-roster and evidence contracts, including failure without a misleading score."""
import json
from pathlib import Path

import pytest

from polytopia_bot.rl import matchup, search_arena
from polytopia_bot.ruleset import RULESET_ID


def game(seed: int, seat: int, reward: int) -> dict:
    return dict(seed=seed, seat=seat, reward=reward, truncated=False, search_turns=1,
                units={}, search_seconds=0., sub_actions=1, lane_steps=1, overrides=0,
                plans=1, incomplete_lanes=0, turns=1, seconds=0., compile_seconds=0.)


@pytest.fixture
def cfg(tmp_path):
    checkpoint = tmp_path / 'model.npz'
    checkpoint.write_bytes(b'synthetic checkpoint; never loaded by a game engine')
    matchup.write_json(checkpoint.with_suffix('.json'), dict(
        sha256=matchup.digest(checkpoint), ruleset=RULESET_ID, config={'size': 14}))
    return dict(candidate=str(checkpoint), opponent=str(checkpoint), lanes=1, size=14)


def test_config_uses_curio_format_and_rejects_typos(tmp_path, cfg) -> None:
    path = tmp_path / 'cfg.json'
    value = dict(cfg, maps=2, seed_base=40, name='example', _note='comment')
    matchup.write_json(path, value)
    loaded, seeds = matchup.load_config(path)
    assert seeds == [40, 41] and loaded['candidate'] == cfg['candidate']
    assert loaded['salt'] == 0 and loaded['max_turn_actions'] == 96
    matchup.write_json(path, {**value, 'lane': 16})
    with pytest.raises(ValueError, match='unknown arena settings'):
        matchup.load_config(path)


def test_metadata_integrity_and_supported_input(cfg) -> None:
    assert matchup.checkpoint_receipts(cfg)['candidate']['sha256'] == matchup.digest(Path(cfg['candidate']))
    meta_path = Path(cfg['candidate']).with_suffix('.json')
    meta = json.loads(meta_path.read_text())
    meta['config']['players'] = 4
    matchup.write_json(meta_path, meta)
    with pytest.raises(ValueError, match='two-seat'):
        matchup.checkpoint_receipts(cfg)
    meta['config']['players'] = 2
    meta['sha256'] = 'wrong'
    matchup.write_json(meta_path, meta)
    with pytest.raises(ValueError, match='integrity/rules mismatch'):
        matchup.checkpoint_receipts(cfg)


def test_paired_score_has_explicit_candidate_perspective_and_stable_order() -> None:
    rows = [game(1, 1, 1), game(1, 2, 0), game(2, 1, 1), game(2, 2, -1)]
    result = matchup.paired_summary(rows, [1, 2])
    assert result == matchup.paired_summary(list(reversed(rows)), [2, 1])
    assert result['perspective'] == 'candidate'
    assert result['win_rate'] == .5 and result['score'] == .625
    assert result['by_seat']['1']['wins'] == 2 and result['by_seat']['2']['wins'] == 0


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'unexpected', 'fractional', 'nan', 'cap_win'])
def test_invalid_roster_never_gets_a_score(change) -> None:
    rows = [game(1, 1, 1), game(1, 2, -1)]
    if change == 'missing':
        rows.pop()
    elif change == 'duplicate':
        rows[1] = rows[0]
    elif change == 'unexpected':
        rows[1]['seed'] = 2
    elif change in ('fractional', 'nan'):
        rows[1]['reward'] = .5 if change == 'fractional' else float('nan')
    else:
        rows[0]['truncated'] = True
    with pytest.raises(ValueError):
        matchup.paired_summary(rows, [1])


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(matchup, 'provenance', lambda size: {'source': 'fixed'})


def test_complete_evidence_and_refuse_overwrite(tmp_path, cfg, evidence, monkeypatch) -> None:
    rows = [game(10, 2, -1), game(10, 1, 1)]

    def run(config, items, *, on_result, **kwargs):
        assert items == [(10, 1), (10, 2)]
        for row in rows:
            on_result(row)
        return rows

    monkeypatch.setattr(search_arena, 'run', run)
    output = tmp_path / 'result'
    result = matchup.evaluate(cfg, [10], output, processes=1, timeout=10)
    assert result['summary']['score'] == .5
    assert result['games_sha256'] == matchup.digest(output / 'games.json')
    assert result['protocol_sha256'] == matchup.digest(output / 'protocol.json')
    assert json.loads((output / 'games.json').read_text()) == rows
    assert json.loads((output / 'complete.json').read_text()) == result
    with pytest.raises(FileExistsError):
        matchup.evaluate(cfg, [10], output, processes=1, timeout=10)


@pytest.mark.parametrize('failure', ['timeout', 'duplicate', 'missing', 'weights_changed', 'metadata_changed', 'source_changed'])
def test_failure_preserves_partial_evidence_without_complete(tmp_path, cfg, evidence, monkeypatch, failure) -> None:
    first = game(10, 1, 1)

    def run(config, items, *, on_result, **kwargs):
        on_result(first)
        if failure == 'timeout':
            raise TimeoutError('test deadline')
        if failure == 'duplicate':
            on_result(first)
        if failure == 'weights_changed':
            Path(cfg['candidate']).write_bytes(b'changed')
        if failure == 'metadata_changed':
            path = Path(cfg['candidate']).with_suffix('.json')
            meta = json.loads(path.read_text())
            matchup.write_json(path, {**meta, 'stage': 'changed'})
        if failure == 'source_changed':
            monkeypatch.setattr(matchup, 'provenance', lambda size: {'source': 'changed'})
        return [first]

    monkeypatch.setattr(search_arena, 'run', run)
    output = tmp_path / 'result'
    with pytest.raises((ValueError, TimeoutError)):
        matchup.evaluate(cfg, [10], output, processes=1, timeout=10)
    assert json.loads((output / 'games.json').read_text()) == [first]
    assert json.loads((output / 'status.json').read_text())['status'] == 'failed'
    assert not (output / 'complete.json').exists()


@pytest.mark.parametrize('failed_file', ['complete.json', 'status.json'])
def test_completion_marker_precedes_complete_status(tmp_path, cfg, evidence, monkeypatch, failed_file) -> None:
    def run(config, items, *, on_result, **kwargs):
        for seed, seat in items:
            on_result(game(seed, seat, 0))

    original = matchup.write_json

    def write(path: Path, value: dict | list) -> None:
        if path.name == failed_file and (failed_file == 'complete.json' or value.get('status') == 'complete'):
            raise OSError('simulated final write failure')
        original(path, value)

    monkeypatch.setattr(search_arena, 'run', run)
    monkeypatch.setattr(matchup, 'write_json', write)
    output = tmp_path / 'result'
    with pytest.raises(OSError, match='final write failure'):
        matchup.evaluate(cfg, [10], output, processes=1, timeout=10)
    if failed_file == 'complete.json':
        assert json.loads((output / 'status.json').read_text())['status'] == 'failed'
        assert not (output / 'complete.json').exists()
    else:
        # The final receipt proves completion even if the status refresh failed.
        assert json.loads((output / 'status.json').read_text())['status'] == 'running'
        assert json.loads((output / 'complete.json').read_text())['summary']['games'] == 2
