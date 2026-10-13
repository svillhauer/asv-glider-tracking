import numpy as np

from glidertrack.metrics import LOST_THRESHOLD, episode_metrics, summarize
from glidertrack.rollout import Episode


def make_episode(offset):
    T, N = 10, 2
    gliders = np.zeros((T, N, 2))
    return Episode(rewards=np.ones(T), asv=np.zeros((T, 2)), heading=np.zeros(T), gliders=gliders,
                   estimates=gliders + [offset, 0.0], covariances=np.tile(np.eye(2) * 1e-4, (T, N, 1, 1)),
                   depth=np.zeros(T), in_range=np.ones((T, N), bool), comm_range=0.4, max_depth=0.12)


def test_episode_metrics():
    m = episode_metrics(make_episode(0.01))
    assert m["reward_per_step"] == 1.0
    assert m["time_in_range"] == 1.0
    assert np.isclose(m["true_error_pct"], 1.0)
    assert not m["lost"]


def test_lost_when_error_exceeds_threshold():
    assert episode_metrics(make_episode(LOST_THRESHOLD + 0.01))["lost"]


def test_summarize_counts_lost_episodes():
    rows = [episode_metrics(make_episode(o)) for o in (0.01, 0.02, 0.5)]
    s = summarize(rows)
    assert s["lost"] == 1 and s["episodes"] == 3
    assert np.isclose(s["true_error_pct"], 2.0)
