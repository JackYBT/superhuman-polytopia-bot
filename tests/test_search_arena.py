"""Outcome reporting and bounded execution, without loading model weights."""
import multiprocessing as mp

import pytest

from polytopia_bot.rl import search_arena


def row(seed: int, seat: int, reward: int, *, truncated: bool = False) -> dict:
    return dict(seed=seed, seat=seat, reward=reward, truncated=truncated, search_turns=1,
                units={}, search_seconds=0., sub_actions=1, lane_steps=1, overrides=0,
                plans=1, incomplete_lanes=0, turns=1, seconds=0., compile_seconds=0., fallbacks=0)


def test_draws_are_not_all_truncations() -> None:
    rows = [row(1, 1, 1), row(1, 2, -1), row(2, 1, 0), row(2, 2, 0, truncated=True)]
    rows[0]['fallbacks'] = 2
    result = search_arena.summarize(rows)
    assert (result['wins'], result['losses'], result['draws'], result['truncations']) == (1, 1, 2, 1)
    assert result['win_rate'] == .25
    assert result['score'] == .5
    # Every map scores .5; bootstrap must keep the two seats together.
    assert result['score_ci95'] == [.5, .5]
    assert result['fallbacks'] == 2


@pytest.fixture
def pool(monkeypatch):
    """Exercise orchestration and cleanup, with controllable worker results."""
    class Pool:
        def __init__(self):
            self.closed = False
            self.values = iter([{'seed': 1}, {'seed': 2}])
            self.timeouts = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

        def imap_unordered(self, fn, items):
            return self

        def next(self, timeout):
            self.timeouts.append(timeout)
            value = next(self.values)
            if isinstance(value, Exception):
                raise value
            return value

    instance = Pool()
    context = mp.get_context('spawn')
    monkeypatch.setattr(context, 'Pool', lambda *a, **kw: instance)
    monkeypatch.setattr(mp, 'get_context', lambda _: context)
    return instance


def test_run_preserves_default_and_streams_rows(pool) -> None:
    received = []
    rows = search_arena.run({}, [(1, 1), (1, 2)], processes=1, on_result=received.append)
    assert rows == received == [{'seed': 1}, {'seed': 2}]
    assert pool.closed and pool.timeouts == [None, None]


@pytest.mark.parametrize('error', [mp.TimeoutError(), RuntimeError('worker failed')])
def test_worker_failure_preserves_partial_callback_and_closes_pool(pool, error) -> None:
    pool.values = iter([{'seed': 1}, error])
    received = []
    expected = TimeoutError if isinstance(error, mp.TimeoutError) else RuntimeError
    with pytest.raises(expected):
        search_arena.run({}, [(1, 1), (1, 2)], processes=1, timeout=10, on_result=received.append)
    assert received == [{'seed': 1}] and pool.closed
    assert all(0 < t <= 10 for t in pool.timeouts)


def test_callback_failure_closes_pool(pool) -> None:
    def reject(row: dict) -> None:
        raise ValueError('duplicate game')

    with pytest.raises(ValueError, match='duplicate game'):
        search_arena.run({}, [(1, 1)], processes=1, on_result=reject)
    assert pool.closed


def test_expired_deadline_closes_pool(pool, monkeypatch) -> None:
    times = iter([0., 2.])
    monkeypatch.setattr(search_arena.time, 'monotonic', lambda: next(times))
    with pytest.raises(TimeoutError, match='wall-time'):
        search_arena.run({}, [(1, 1)], processes=1, timeout=1)
    assert pool.closed and not pool.timeouts
