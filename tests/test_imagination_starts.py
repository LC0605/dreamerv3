import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np


sys.path.insert(0, str(Path(__file__).parents[1] / "dreamerv3"))
from dreamerv3.agent import last_valid_indices, take_time  # noqa: E402


def padded_mask(lengths, width=301):
    steps = np.arange(width)[None, :]
    return steps < np.asarray(lengths)[:, None]


def test_last_valid_starts_for_short_success_episode():
    indices = np.asarray(last_valid_indices(jnp.asarray(padded_mask([75])), 16))
    np.testing.assert_array_equal(indices[0], np.arange(59, 75))


def test_last_valid_starts_for_collision_episode():
    indices = np.asarray(last_valid_indices(jnp.asarray(padded_mask([20])), 16))
    np.testing.assert_array_equal(indices[0], np.arange(4, 20))


def test_last_valid_starts_for_full_timeout_episode():
    indices = np.asarray(last_valid_indices(jnp.asarray(padded_mask([301])), 16))
    np.testing.assert_array_equal(indices[0], np.arange(285, 301))


def test_mixed_length_batch_never_gathers_padding():
    lengths = np.asarray([75, 20, 301])
    mask = jnp.asarray(padded_mask(lengths))
    indices = last_valid_indices(mask, 16)
    gathered_mask = np.asarray(jnp.take_along_axis(mask, indices, axis=1))
    assert gathered_mask.all()

    values = np.broadcast_to(np.arange(301)[None, :, None], (3, 301, 2))
    gathered = np.asarray(take_time({"value": jnp.asarray(values)}, indices)["value"])
    for row, length in enumerate(lengths):
        np.testing.assert_array_equal(gathered[row, :, 0], np.arange(length - 16, length))


def test_episode_shorter_than_imag_last_repeats_valid_not_padding():
    mask = jnp.asarray(padded_mask([5], width=20))
    indices = last_valid_indices(mask, 16)
    gathered = np.asarray(jnp.take_along_axis(mask, indices, axis=1))
    assert gathered.all()
    assert set(np.asarray(indices)[0]) <= set(range(5))
