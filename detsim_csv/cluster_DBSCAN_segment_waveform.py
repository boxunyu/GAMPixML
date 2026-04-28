#!/usr/bin/env python3
"""
Cluster waveform-based GAMPix pixel hits and match them to waveform-based tile hits.

This version is intended for the newer .h5 format with top-level datasets
    meta, pixels, tiles
where pixels/tiles contain a 20-bin waveform instead of scalar `hit charge` and
`hit t` fields.

Main outputs per pixel cluster:
  - pixel/tile charge centroids in x and y
  - pixel/tile transverse widths in x and y, using charge-weighted RMS
  - pixel/tile longitudinal time widths, using both charge-weighted RMS and an
    optional Gaussian fit to the summed waveform
  - number of triggered pixel/tile channels
  - pixel/tile spans in x and y
  - pixel charge, tile charge, pixel/tile charge ratio
  - optional dominant segment and drift length, if a truth file and detector yaml
    are available

Notes:
  1. The time coordinate is built as
         sample_time = trig_t + tick_index * TICK_SIZE
     Set TICK_SIZE to your real readout tick spacing if you want physical units.
     If you leave TICK_SIZE = 1.0, all time widths are in tick units.
  2. DBSCAN is run event-by-event to avoid merging hits from different events.
  3. The code clusters only pixel waveform samples. Tile waveform samples are
     then matched to each pixel cluster in the same event using spatial and time
     windows.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from collections import Counter, defaultdict
import warnings

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# Optional dependencies. The script still runs without them.
try:
    from scipy.optimize import curve_fit
except Exception:
    curve_fit = None

try:
    from gampixpy import config as gampix_config
    import torch
except Exception:
    gampix_config = None
    torch = None


# ============================================================
# User parameters
# ============================================================
HIT_FILE = "../detsim_sample/gampixpy_fullgeoanatruth-vd-reduced_g4_00_2Mhz_segmentlabel_lowtrig_5mmpitch.h5"
OUTPUT_CSV = "../detsim_sample/clusters_dbscan_segment_lowtrig_waveform_features.csv"

# Optional truth/drift information. Set either to None to skip drift_length.
TRUTH_FILE = "../g4_cv1_sample/fullgeoanatruth-vd-reduced_g4_00.h5"
DETECTOR_YAML = "/home/yboxun/NeutrinoGAMPix/detsim_prediction/depth/far_detector_vd.yaml"

# If you only want a quick test, set one or more of these limits.
# MAX_PIXEL_SAMPLES_TOTAL is usually the most useful cap, because one pixel
# channel can expand into up to 20 time samples before DBSCAN.
MAX_EVENTS = 10
MAX_PIXEL_CHANNELS = None
MAX_PIXEL_SAMPLES_TOTAL = 50_000
MAX_PIXEL_SAMPLES_PER_EVENT = 5_000
SAMPLE_LIMIT_STRATEGY = "charge"  # "charge" or "first"

# Waveform handling.
N_TICKS = 20
TICK_SIZE = 1.0            # time unit per tick; use physical tick spacing if known
MIN_SAMPLE_CHARGE = 0.0    # keep waveform samples with q > this value

# DBSCAN on normalized coordinates:
#   x_norm = x / PIXEL_PITCH
#   y_norm = y / PIXEL_PITCH
#   t_norm = time / TIME_SCALE
# With 5 mm pitch, eps ~ 1.5--2.0 connects neighboring pixels close in time.
PIXEL_PITCH = 5.0
TIME_SCALE = 1.0
DBSCAN_EPS = 1.75
DBSCAN_MIN_SAMPLES = 5

# Tile matching around each pixel cluster. Units are the original x/y/time units.
# Because tile centers are coarser than pixel centers, use a spatial pad larger
# than one pixel pitch. Tune TILE_MATCH_PAD_XY to the actual half-size of a tile.
TILE_MATCH_PAD_XY = 10.0
TILE_MATCH_PAD_T = 2.0

# Skip tiny clusters after DBSCAN.
MIN_CLUSTER_PIXEL_SAMPLES = 3
MIN_CLUSTER_PIXEL_CHARGE = 0.0

# Progress printing.
PRINT_EVERY_EVENTS = 100
PRINT_EVERY_CLUSTERS = 500


# ============================================================
# Small numerical helpers
# ============================================================
def safe_weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    wsum = np.sum(weights)
    if len(values) == 0 or wsum <= 0:
        return np.nan
    return float(np.sum(values * weights) / wsum)


def safe_weighted_rms(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = safe_weighted_mean(values, weights)
    wsum = np.sum(weights)
    if not np.isfinite(mean) or wsum <= 0:
        return np.nan
    var = np.sum(weights * (values - mean) ** 2) / wsum
    return float(np.sqrt(max(var, 0.0)))


def safe_span(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    return float(np.max(values) - np.min(values))


def gaussian_with_baseline(t, amp, mu, sigma, baseline):
    sigma = np.maximum(sigma, 1.0e-12)
    return baseline + amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def aggregate_waveform_by_time(times, charges):
    """Return sorted unique time bins and total charge in each bin."""
    times = np.asarray(times, dtype=float)
    charges = np.asarray(charges, dtype=float)
    if len(times) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    # Avoid exact-float fragmentation when trig_t is stored as float32.
    rounded = np.round(times, decimals=6)
    unique_t, inv = np.unique(rounded, return_inverse=True)
    y = np.zeros(len(unique_t), dtype=float)
    np.add.at(y, inv, charges)
    order = np.argsort(unique_t)
    return unique_t[order], y[order]


def waveform_time_features(times, charges):
    """
    Moment and Gaussian-fit features for a longitudinal charge distribution.

    Returns a dictionary with:
      charge, mean, width_moment, gauss_amp, gauss_mean, gauss_sigma,
      gauss_baseline, gauss_success
    """
    times = np.asarray(times, dtype=float)
    charges = np.asarray(charges, dtype=float)
    mask = np.isfinite(times) & np.isfinite(charges) & (charges > 0)
    times = times[mask]
    charges = charges[mask]

    total_q = float(np.sum(charges)) if len(charges) else 0.0
    mean = safe_weighted_mean(times, charges)
    width = safe_weighted_rms(times, charges)

    out = {
        "charge": total_q,
        "mean": mean,
        "width_moment": width,
        "gauss_amp": np.nan,
        "gauss_mean": np.nan,
        "gauss_sigma": np.nan,
        "gauss_baseline": np.nan,
        "gauss_success": False,
    }

    # Need at least a few time bins to fit four parameters.
    tbin, ybin = aggregate_waveform_by_time(times, charges)
    if curve_fit is None or len(tbin) < 4 or np.sum(ybin) <= 0:
        return out

    baseline0 = float(np.min(ybin))
    amp0 = float(np.max(ybin) - baseline0)
    if amp0 <= 0:
        return out
    mu0 = safe_weighted_mean(tbin, np.maximum(ybin - baseline0, 0.0))
    sigma0 = safe_weighted_rms(tbin, np.maximum(ybin - baseline0, 0.0))
    if not np.isfinite(mu0):
        mu0 = float(tbin[np.argmax(ybin)])
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = max(TICK_SIZE, 1.0)

    lower = [0.0, float(np.min(tbin) - 2 * TICK_SIZE), 1.0e-6, -np.inf]
    upper = [np.inf, float(np.max(tbin) + 2 * TICK_SIZE), np.inf, np.inf]

    try:
        popt, _ = curve_fit(
            gaussian_with_baseline,
            tbin,
            ybin,
            p0=[amp0, mu0, sigma0, baseline0],
            bounds=(lower, upper),
            maxfev=10000,
        )
        amp, mu, sigma, baseline = popt
        if np.isfinite(sigma) and sigma > 0:
            out.update({
                "gauss_amp": float(amp),
                "gauss_mean": float(mu),
                "gauss_sigma": float(abs(sigma)),
                "gauss_baseline": float(baseline),
                "gauss_success": True,
            })
    except Exception:
        pass

    return out


# ============================================================
# HDF5 waveform expansion helpers
# ============================================================
def event_slices(event_ids):
    """Yield (event_id, start, stop) for an array sorted by event id."""
    if len(event_ids) == 0:
        return
    starts = np.r_[0, np.nonzero(event_ids[1:] != event_ids[:-1])[0] + 1]
    stops = np.r_[starts[1:], len(event_ids)]
    for start, stop in zip(starts, stops):
        yield int(event_ids[start]), int(start), int(stop)


def expand_waveform_channels(x, y, trig_t, waveform, charge_threshold=0.0):
    """
    Expand channel-level 20-bin waveforms into sample-level arrays.

    Returns row index, tick index, x, y, sample time, and sample charge.
    The row index refers to the local event slice, not the full HDF5 dataset.
    """
    waveform = np.asarray(waveform, dtype=float)
    keep = np.isfinite(waveform) & (waveform > charge_threshold)
    row_idx, tick_idx = np.nonzero(keep)

    if len(row_idx) == 0:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f

    q = waveform[row_idx, tick_idx].astype(float)
    sx = np.asarray(x, dtype=float)[row_idx]
    sy = np.asarray(y, dtype=float)[row_idx]
    st = np.asarray(trig_t, dtype=float)[row_idx] + tick_idx.astype(float) * TICK_SIZE
    return row_idx.astype(int), tick_idx.astype(int), sx, sy, st, q



def limit_expanded_samples(row_idx, tick_idx, x, y, t, q, max_samples, strategy="charge"):
    """Limit expanded waveform samples before DBSCAN to keep runtime/memory manageable."""
    if max_samples is None or len(q) <= max_samples:
        return row_idx, tick_idx, x, y, t, q
    max_samples = int(max_samples)
    if max_samples <= 0:
        empty_i = np.array([], dtype=int)
        empty_f = np.array([], dtype=float)
        return empty_i, empty_i, empty_f, empty_f, empty_f, empty_f
    if strategy == "charge":
        # Keep the largest charge samples. argpartition is faster than a full sort.
        keep = np.argpartition(q, -max_samples)[-max_samples:]
        # Sort selected samples for deterministic clustering/debugging.
        keep = keep[np.lexsort((x[keep], y[keep], t[keep]))]
    elif strategy == "first":
        keep = np.arange(max_samples)
    else:
        raise ValueError(f"Unknown SAMPLE_LIMIT_STRATEGY={strategy!r}; use 'charge' or 'first'")
    return row_idx[keep], tick_idx[keep], x[keep], y[keep], t[keep], q[keep]

def accumulate_segment_weights(labels, attribution, row_idx, tick_idx, charges):
    """
    Accumulate segment weights using per-tick attribution fractions.

    labels:      shape (n_channels, 3)
    attribution: shape (n_channels, 20, 3)
    row_idx/tick_idx/charges are sample-level arrays.
    """
    seg_w = defaultdict(float)
    if len(row_idx) == 0:
        return seg_w

    labs = labels[row_idx]                 # (n_samples, 3)
    fracs = attribution[row_idx, tick_idx]  # (n_samples, 3)
    charges = np.asarray(charges, dtype=float)

    for k in range(labs.shape[1]):
        lab_k = labs[:, k].astype(int)
        frac_k = fracs[:, k].astype(float)
        mask = (lab_k != -9999) & np.isfinite(frac_k) & (frac_k > 0) & np.isfinite(charges)
        if not np.any(mask):
            continue
        contrib = frac_k[mask] * charges[mask]
        for sid, w in zip(lab_k[mask], contrib):
            seg_w[int(sid)] += float(w)
    return seg_w


def dominant_segment_and_drift(seg_w, segment_drift_map):
    if not seg_w:
        return -9999, np.nan
    dominant = max(seg_w.items(), key=lambda kv: kv[1])[0]
    if not segment_drift_map:
        return int(dominant), np.nan
    num = 0.0
    den = 0.0
    for sid, w in seg_w.items():
        drift = segment_drift_map.get(int(sid), np.nan)
        if np.isfinite(drift) and w > 0:
            num += w * drift
            den += w
    drift_length = float(num / den) if den > 0 else np.nan
    return int(dominant), drift_length


# ============================================================
# Optional truth drift mapping
# ============================================================
def load_segment_drift_map(truth_file, detector_yaml):
    if truth_file is None or detector_yaml is None:
        return {}
    if not os.path.exists(truth_file):
        warnings.warn(f"Truth file not found: {truth_file}; drift_length will be NaN")
        return {}
    if gampix_config is None:
        warnings.warn("gampixpy/torch import failed; drift_length will be NaN")
        return {}

    detector_config = gampix_config.DetectorConfig(detector_yaml)
    anode_center = detector_config["drift_volumes"]["volume_0"]["anode_center"]
    drift_axis = detector_config["drift_volumes"]["volume_0"]["drift_axis"]
    if torch is not None and torch.is_tensor(anode_center):
        anode_center = anode_center.detach().cpu().numpy()
    if torch is not None and torch.is_tensor(drift_axis):
        drift_axis = drift_axis.detach().cpu().numpy()
    anode_center = np.asarray(anode_center, dtype=float)
    drift_axis = np.asarray(drift_axis, dtype=float)

    with h5py.File(truth_file, "r") as f:
        seg = f["segments"]
        s_ids = seg["segment_id"][:].astype(int)
        seg_mid_x = (seg["x_start"][:] + seg["x_end"][:]) / 2.0
        seg_mid_y = (seg["y_start"][:] + seg["y_end"][:]) / 2.0
        seg_mid_z = (seg["z_start"][:] + seg["z_end"][:]) / 2.0
        seg_pos = np.stack([seg_mid_x, seg_mid_y, seg_mid_z], axis=1).astype(float)

    # Drift = (segment midpoint - anode center) dot (-drift_axis)
    drift_values = np.dot(seg_pos - anode_center, -drift_axis)
    return dict(zip(s_ids, drift_values.astype(float)))


# ============================================================
# Feature extraction for one cluster
# ============================================================
def build_feature_row(
    global_cluster_id,
    event_id,
    pixel_cluster_mask,
    pixel_row_idx,
    pixel_tick_idx,
    pixel_xs,
    pixel_ys,
    pixel_ts,
    pixel_qs,
    pixel_channel_labels,
    pixel_channel_attr,
    tile_row_idx,
    tile_tick_idx,
    tile_xs,
    tile_ys,
    tile_ts,
    tile_qs,
    tile_channel_global_rows,
    segment_drift_map,
):
    # Pixel samples in this cluster.
    pr = pixel_row_idx[pixel_cluster_mask]
    ptick = pixel_tick_idx[pixel_cluster_mask]
    px = pixel_xs[pixel_cluster_mask]
    py = pixel_ys[pixel_cluster_mask]
    ptime = pixel_ts[pixel_cluster_mask]
    pq = pixel_qs[pixel_cluster_mask]

    pixel_charge = float(np.sum(pq))
    if len(pq) < MIN_CLUSTER_PIXEL_SAMPLES or pixel_charge <= MIN_CLUSTER_PIXEL_CHARGE:
        return None

    # Match tile waveform samples in the same event.
    min_x, max_x = float(np.min(px)), float(np.max(px))
    min_y, max_y = float(np.min(py)), float(np.max(py))
    min_t, max_t = float(np.min(ptime)), float(np.max(ptime))

    tile_mask = (
        (tile_xs >= min_x - TILE_MATCH_PAD_XY) &
        (tile_xs <= max_x + TILE_MATCH_PAD_XY) &
        (tile_ys >= min_y - TILE_MATCH_PAD_XY) &
        (tile_ys <= max_y + TILE_MATCH_PAD_XY) &
        (tile_ts >= min_t - TILE_MATCH_PAD_T) &
        (tile_ts <= max_t + TILE_MATCH_PAD_T)
    )

    tx = tile_xs[tile_mask]
    ty = tile_ys[tile_mask]
    ttime = tile_ts[tile_mask]
    tq = tile_qs[tile_mask]
    tr = tile_row_idx[tile_mask]

    tile_charge = float(np.sum(tq)) if len(tq) else 0.0
    charge_ratio = pixel_charge / tile_charge if tile_charge > 0 else np.nan

    pix_time = waveform_time_features(ptime, pq)
    tile_time = waveform_time_features(ttime, tq)

    seg_w = accumulate_segment_weights(pixel_channel_labels, pixel_channel_attr, pr, ptick, pq)
    dominant_segment_id, drift_length = dominant_segment_and_drift(seg_w, segment_drift_map)

    n_pixel_triggered = int(len(np.unique(pr)))
    n_tile_triggered = int(len(np.unique(tr))) if len(tr) else 0

    # Global row ids make it easier to debug matching outside this script.
    # tile_channel_global_rows is the array of full-file tile row ids for this event.
    matched_tile_global_rows = tile_channel_global_rows[np.unique(tr)] if len(tr) else np.array([], dtype=int)

    row = {
        "cluster_id": int(global_cluster_id),
        "event_id": int(event_id),

        "n_pixel_triggered": n_pixel_triggered,
        "n_tile_triggered": n_tile_triggered,
        "n_pixel_time_samples": int(len(pq)),
        "n_tile_time_samples": int(len(tq)),

        "pixel_charge": pixel_charge,
        "tile_charge": tile_charge,
        "charge_ratio_pixel_over_tile": float(charge_ratio) if np.isfinite(charge_ratio) else np.nan,

        "pixel_centroid_x": safe_weighted_mean(px, pq),
        "pixel_centroid_y": safe_weighted_mean(py, pq),
        "tile_centroid_x": safe_weighted_mean(tx, tq),
        "tile_centroid_y": safe_weighted_mean(ty, tq),

        "pixel_width_x": safe_weighted_rms(px, pq),
        "pixel_width_y": safe_weighted_rms(py, pq),
        "tile_width_x": safe_weighted_rms(tx, tq),
        "tile_width_y": safe_weighted_rms(ty, tq),

        "pixel_span_x": safe_span(px),
        "pixel_span_y": safe_span(py),
        "tile_span_x": safe_span(tx),
        "tile_span_y": safe_span(ty),

        "pixel_time_mean": pix_time["mean"],
        "pixel_time_width_moment": pix_time["width_moment"],
        "pixel_time_gauss_mean": pix_time["gauss_mean"],
        "pixel_time_gauss_sigma": pix_time["gauss_sigma"],
        "pixel_time_gauss_amp": pix_time["gauss_amp"],
        "pixel_time_gauss_baseline": pix_time["gauss_baseline"],
        "pixel_time_gauss_success": bool(pix_time["gauss_success"]),

        "tile_time_mean": tile_time["mean"],
        "tile_time_width_moment": tile_time["width_moment"],
        "tile_time_gauss_mean": tile_time["gauss_mean"],
        "tile_time_gauss_sigma": tile_time["gauss_sigma"],
        "tile_time_gauss_amp": tile_time["gauss_amp"],
        "tile_time_gauss_baseline": tile_time["gauss_baseline"],
        "tile_time_gauss_success": bool(tile_time["gauss_success"]),

        "pixel_time_min": float(min_t),
        "pixel_time_max": float(max_t),
        "tile_time_min": float(np.min(ttime)) if len(ttime) else np.nan,
        "tile_time_max": float(np.max(ttime)) if len(ttime) else np.nan,

        "dominant_segment_id": int(dominant_segment_id),
        "drift_length": float(drift_length) if np.isfinite(drift_length) else np.nan,

        # Debug fields. Comment these out if you want a smaller csv.
        "matched_tile_global_rows": ";".join(map(str, matched_tile_global_rows.tolist())),
    }
    return row


# ============================================================
# Main processing
# ============================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    print("started...")
    print(f"Reading: {HIT_FILE}")
    print(f"Writing: {OUTPUT_CSV}")

    segment_drift_map = load_segment_drift_map(TRUTH_FILE, DETECTOR_YAML)
    if segment_drift_map:
        print(f"Loaded drift map for {len(segment_drift_map)} truth segments")
    else:
        print("No drift map loaded; drift_length will be NaN")

    rows = []
    global_cluster_id = 0
    total_noise_samples = 0
    total_pixel_samples_seen = 0
    total_pixel_samples_clustered = 0
    total_clusters = 0
    total_events_seen = 0

    with h5py.File(HIT_FILE, "r") as f:
        P = f["pixels"]
        T = f["tiles"]

        # The new sample has these fields, according to inspect.txt:
        # pixels: event id, pixel tpc, pixel x/y, trig z/t, waveform, attribution, label
        # tiles:  event id, tile  tpc, tile  x/y, trig z/t, waveform, attribution, label
        p_event = P["event id"][:]
        t_event = T["event id"][:]

        # Sort once by event id. This makes event-by-event processing simple even
        # if the HDF5 rows are not perfectly ordered.
        p_order = np.argsort(p_event, kind="stable")
        t_order = np.argsort(t_event, kind="stable")
        p_event_sorted = p_event[p_order]
        t_event_sorted = t_event[t_order]

        # Build event -> tile slice lookup.
        tile_slices = {
            eid: (start, stop)
            for eid, start, stop in event_slices(t_event_sorted)
        }

        for event_counter, (event_id, ps0, ps1) in enumerate(event_slices(p_event_sorted), start=1):
            if MAX_EVENTS is not None and event_counter > MAX_EVENTS:
                break
            total_events_seen += 1

            p_idx = p_order[ps0:ps1]
            if MAX_PIXEL_CHANNELS is not None:
                # Simple global cap for quick tests. Once enough input channels are
                # consumed, stop processing new events.
                if ps0 >= MAX_PIXEL_CHANNELS:
                    break
                p_idx = p_idx[: max(0, MAX_PIXEL_CHANNELS - ps0)]

            remaining_sample_budget = None
            if MAX_PIXEL_SAMPLES_TOTAL is not None:
                remaining_sample_budget = MAX_PIXEL_SAMPLES_TOTAL - total_pixel_samples_clustered
                if remaining_sample_budget <= 0:
                    print(f"Reached MAX_PIXEL_SAMPLES_TOTAL={MAX_PIXEL_SAMPLES_TOTAL}; stopping.")
                    break

            # Pixel event arrays.
            px_ch = P["pixel x"][p_idx].astype(float)
            py_ch = P["pixel y"][p_idx].astype(float)
            pt_ch = P["trig t"][p_idx].astype(float)
            pwf = P["waveform"][p_idx].astype(float)
            plabel = P["label"][p_idx].astype(int)
            pattr = P["attribution"][p_idx].astype(float)

            pr, ptick, px, py, ptime, pq = expand_waveform_channels(
                px_ch, py_ch, pt_ch, pwf, charge_threshold=MIN_SAMPLE_CHARGE
            )
            total_pixel_samples_seen += len(pq)

            per_event_limit = MAX_PIXEL_SAMPLES_PER_EVENT
            if remaining_sample_budget is not None:
                per_event_limit = min(
                    remaining_sample_budget,
                    per_event_limit if per_event_limit is not None else remaining_sample_budget,
                )

            before_limit = len(pq)
            pr, ptick, px, py, ptime, pq = limit_expanded_samples(
                pr, ptick, px, py, ptime, pq,
                max_samples=per_event_limit,
                strategy=SAMPLE_LIMIT_STRATEGY,
            )
            total_pixel_samples_clustered += len(pq)

            if before_limit > len(pq):
                print(
                    f"Event {event_id}: limited pixel waveform samples "
                    f"from {before_limit} to {len(pq)}"
                )

            if len(pq) < DBSCAN_MIN_SAMPLES:
                continue

            # Tile event arrays. If no tile entry for this event, create empty arrays.
            if event_id in tile_slices:
                ts0, ts1 = tile_slices[event_id]
                t_idx = t_order[ts0:ts1]
                tx_ch = T["tile x"][t_idx].astype(float)
                ty_ch = T["tile y"][t_idx].astype(float)
                tt_ch = T["trig t"][t_idx].astype(float)
                twf = T["waveform"][t_idx].astype(float)
                tr, ttick, tx, ty, ttime, tq = expand_waveform_channels(
                    tx_ch, ty_ch, tt_ch, twf, charge_threshold=MIN_SAMPLE_CHARGE
                )
                tile_global_rows = t_idx
            else:
                tr = ttick = np.array([], dtype=int)
                tx = ty = ttime = tq = np.array([], dtype=float)
                tile_global_rows = np.array([], dtype=int)

            # Cluster pixel waveform samples in this event.
            features = np.column_stack([
                px / PIXEL_PITCH,
                py / PIXEL_PITCH,
                ptime / TIME_SCALE,
            ])
            db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(features)
            db_labels = db.labels_
            cluster_ids = np.unique(db_labels[db_labels >= 0])
            total_noise_samples += int(np.sum(db_labels < 0))

            for cid in cluster_ids:
                cmask = db_labels == cid
                row = build_feature_row(
                    global_cluster_id=global_cluster_id,
                    event_id=event_id,
                    pixel_cluster_mask=cmask,
                    pixel_row_idx=pr,
                    pixel_tick_idx=ptick,
                    pixel_xs=px,
                    pixel_ys=py,
                    pixel_ts=ptime,
                    pixel_qs=pq,
                    pixel_channel_labels=plabel,
                    pixel_channel_attr=pattr,
                    tile_row_idx=tr,
                    tile_tick_idx=ttick,
                    tile_xs=tx,
                    tile_ys=ty,
                    tile_ts=ttime,
                    tile_qs=tq,
                    tile_channel_global_rows=tile_global_rows,
                    segment_drift_map=segment_drift_map,
                )
                if row is None:
                    continue
                rows.append(row)
                global_cluster_id += 1
                total_clusters += 1

                if total_clusters % PRINT_EVERY_CLUSTERS == 0:
                    print(f"Processed {total_clusters} clusters...")

            if event_counter % PRINT_EVERY_EVENTS == 0:
                print(
                    f"Processed {event_counter} events; "
                    f"clusters so far = {total_clusters}; "
                    f"pixel samples clustered so far = {total_pixel_samples_clustered}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("Done.")
    print(f"Events processed: {total_events_seen}")
    print(f"Pixel waveform samples seen before limits: {total_pixel_samples_seen}")
    print(f"Pixel waveform samples actually clustered: {total_pixel_samples_clustered}")
    print(f"DBSCAN noise pixel samples: {total_noise_samples}")
    print(f"Clusters saved: {len(df)}")
    print(f"Output CSV: {OUTPUT_CSV}")

    if curve_fit is None:
        print("WARNING: scipy.optimize.curve_fit was not available; Gaussian fit columns are NaN.")


if __name__ == "__main__":
    main()
