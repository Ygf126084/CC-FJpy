"""
Example: shallow-surface array cross-correlation.

示例：浅地表线性台阵互相关计算。

This example generates a synthetic shallow-surface array dataset, computes
frequency-domain cross-correlations with ``ccfj.CC``, then converts the result
to the time domain for display.
"""

import matplotlib.pyplot as plt
import numpy as np

import ccfj


def build_shallow_surface_dataset(
    nsta=12,
    duration=60.0,
    fs=250.0,
    spacing=5.0,
    apparent_velocity=300.0,
    random_seed=2026,
):
    """
    Build a simple synthetic shallow-surface dataset for cross-correlation.

    Parameters
    ----------
    nsta : int
        Number of receivers in the linear array.
    duration : float
        Record length in seconds.
    fs : float
        Sample rate in Hz.
    spacing : float
        Receiver spacing in meters.
    apparent_velocity : float
        Apparent propagation velocity in m/s.
    random_seed : int
        Seed for reproducible random noise.
    """

    npts = int(duration * fs)
    time = np.arange(npts, dtype=np.float32) / fs
    offsets = np.arange(nsta, dtype=np.float32) * spacing

    rng = np.random.default_rng(random_seed)
    shared_wavefield = rng.standard_normal(npts).astype(np.float32)
    coherent_burst = np.sin(2 * np.pi * 18.0 * time) * np.exp(-((time - 20.0) / 3.0) ** 2)
    coherent_burst += 0.7 * np.sin(2 * np.pi * 32.0 * time) * np.exp(-((time - 38.0) / 2.5) ** 2)
    coherent_burst = coherent_burst.astype(np.float32)

    data = np.zeros((nsta, npts), dtype=np.float32)
    for i, offset in enumerate(offsets):
        delay = int(round(offset / apparent_velocity * fs))
        shifted = np.roll(shared_wavefield, delay)
        if delay > 0:
            shifted[:delay] = 0.0
        station_noise = 0.35 * rng.standard_normal(npts).astype(np.float32)
        data[i] = shifted + coherent_burst + station_noise

    return data, offsets, fs


def compute_shallow_surface_cc(data, fs, overlaprate=0.5, n_threads=4):
    """
    Compute cross-correlations for the full receiver array.
    """

    nsta, npts = data.shape
    fftlen = 2048
    nf = fftlen // 2 + 1

    pairs = ccfj.GetStationPairs(nsta)
    startend = np.tile(np.array([0, npts], dtype=np.int32), nsta)

    ccfs = ccfj.CC(
        npts,
        nsta,
        nf,
        fftlen,
        pairs,
        startend,
        data.reshape(-1),
        overlaprate=overlaprate,
        nThreads=n_threads,
        fstride=1,
        ifonebit=1,
        ifspecwhittenning=0,
    )

    cc_time = np.fft.irfft(ccfs, n=fftlen, axis=1)
    cc_time = np.roll(cc_time, fftlen // 2, axis=1)
    lag = (np.arange(fftlen) - fftlen // 2) / fs
    return pairs.reshape(-1, 2), lag, cc_time


def plot_results(data, offsets, fs, pairs, lag, cc_time):
    """
    Plot raw records and cross-correlations for the nearest-neighbor pairs.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    time = np.arange(data.shape[1]) / fs
    for i in range(data.shape[0]):
        trace = data[i] / np.max(np.abs(data[i]))
        axes[0].plot(time, trace + offsets[i], linewidth=0.8, color="black")
    axes[0].set_title("Synthetic shallow-surface records")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Offset (m)")

    pair_offsets = offsets[pairs[:, 1]] - offsets[pairs[:, 0]]
    neighbor_mask = pair_offsets == pair_offsets.min()
    for idx, pair in enumerate(pairs[neighbor_mask]):
        trace = cc_time[neighbor_mask][idx]
        trace = trace / np.max(np.abs(trace))
        axes[1].plot(lag, trace + offsets[pair[0]], linewidth=0.8)
    axes[1].set_xlim(-1.5, 1.5)
    axes[1].set_title("Nearest-neighbor cross-correlations")
    axes[1].set_xlabel("Lag time (s)")
    axes[1].set_ylabel("Reference offset (m)")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    data, offsets, fs = build_shallow_surface_dataset()
    pairs, lag, cc_time = compute_shallow_surface_cc(data, fs)
    plot_results(data, offsets, fs, pairs, lag, cc_time)
