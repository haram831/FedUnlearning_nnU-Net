from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_WEIGHTS = {
    "spacing": 0.35,
    "shape": 0.30,
    "foreground_intensity": 0.20,
    "crop": 0.15,
}

DEFAULT_INTENSITY_STAT_KEYS = (
    "mean",
    "std",
    "median",
    "percentile_00_5",
    "percentile_99_5",
)


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _to_float(value: Any) -> float:
    return float(value)


def _empty_result(metric: str, note: str) -> Dict[str, Any]:
    return {
        "metric": metric,
        "raw_distance": None,
        "normalization_scale": None,
        "normalized_distance": None,
        "notes": [note],
    }


def relative_absolute_difference(a: float, b: float, eps: float = 1e-8) -> Dict[str, Any]:
    raw_distance = abs(_to_float(a) - _to_float(b))
    scale = max(abs(_to_float(a)), abs(_to_float(b)), eps)
    return {
        "metric": "relative_absolute_difference",
        "raw_distance": raw_distance,
        "normalization_scale": scale,
        "normalized_distance": raw_distance / scale,
        "notes": [],
    }


def normalized_l2_distance(
    a: Iterable[float],
    b: Iterable[float],
    scale: Optional[float] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    a_arr = _as_float_array(a).reshape(-1)
    b_arr = _as_float_array(b).reshape(-1)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"Cannot compare vectors with different shapes: {a_arr.shape} vs {b_arr.shape}")

    raw_distance = float(np.linalg.norm(a_arr - b_arr))
    normalization_scale = (
        float(scale)
        if scale is not None
        else max(float(np.linalg.norm(a_arr)), float(np.linalg.norm(b_arr)), eps)
    )
    normalization_scale = max(normalization_scale, eps)
    return {
        "metric": "normalized_l2_distance",
        "raw_distance": raw_distance,
        "normalization_scale": normalization_scale,
        "normalized_distance": raw_distance / normalization_scale,
        "notes": [],
    }


def histogram_wasserstein_distance(
    hist_a: Dict[str, Any],
    hist_b: Dict[str, Any],
    eps: float = 1e-8,
) -> Dict[str, Any]:
    edges_a = _as_float_array(hist_a["bin_edges"]).reshape(-1)
    edges_b = _as_float_array(hist_b["bin_edges"]).reshape(-1)
    if edges_a.shape != edges_b.shape or not np.allclose(edges_a, edges_b, rtol=0.0, atol=0.0):
        raise ValueError("Cannot compare histograms with different bin edges")

    counts_a = _as_float_array(hist_a["counts"]).reshape(-1)
    counts_b = _as_float_array(hist_b["counts"]).reshape(-1)
    if len(edges_a) != len(counts_a) + 1 or len(edges_b) != len(counts_b) + 1:
        raise ValueError("Histogram bin_edges length must be counts length + 1")

    total_a = float(np.sum(counts_a))
    total_b = float(np.sum(counts_b))
    if total_a <= 0 or total_b <= 0:
        return {
            "metric": "histogram_wasserstein_distance",
            "raw_distance": 0.0,
            "normalization_scale": max(float(edges_a[-1] - edges_a[0]), eps),
            "normalized_distance": 0.0,
            "notes": ["At least one histogram has zero total count; distance set to 0."],
        }

    probs_a = counts_a / total_a
    probs_b = counts_b / total_b
    widths = np.diff(edges_a)
    raw_distance = float(np.sum(np.abs(np.cumsum(probs_a - probs_b)) * widths))
    support = max(float(edges_a[-1] - edges_a[0]), eps)
    return {
        "metric": "histogram_wasserstein_distance",
        "raw_distance": raw_distance,
        "normalization_scale": support,
        "normalized_distance": raw_distance / support,
        "notes": [],
    }


def summarize_distribution_features(
    values: Any,
    percentiles: Tuple[float, ...] = (10.0, 50.0, 90.0),
) -> List[float]:
    arr = _as_float_array(values)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        summary = np.percentile(arr, percentiles)
    else:
        summary = np.percentile(arr, percentiles, axis=0)
    return [float(i) for i in np.asarray(summary).reshape(-1)]


def _component_result(
    name: str,
    distance: Optional[float],
    details: Dict[str, Any],
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "component": name,
        "normalized_distance": distance,
        "details": details,
        "notes": notes or [],
    }


def _distance_or_zero(result: Dict[str, Any]) -> float:
    distance = result.get("normalized_distance")
    return 0.0 if distance is None else float(distance)


def _shared_channel_stats(
    original: Dict[str, Any],
    excluded: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    original_channels = {str(k): v for k, v in original.items()}
    excluded_channels = {str(k): v for k, v in excluded.items()}
    shared = sorted(set(original_channels) & set(excluded_channels), key=str)
    missing = []
    for channel in sorted(set(original_channels) - set(excluded_channels), key=str):
        missing.append(f"Channel {channel} missing in excluded fingerprint.")
    for channel in sorted(set(excluded_channels) - set(original_channels), key=str):
        missing.append(f"Channel {channel} missing in original fingerprint.")
    return shared, missing


def _spacing_or_shape_distance(
    original: Dict[str, Any],
    excluded: Dict[str, Any],
    key: str,
    component: str,
) -> Dict[str, Any]:
    if key not in original or key not in excluded:
        return _component_result(
            component,
            None,
            {},
            [f"Missing {key}; component distance skipped."],
        )

    original_summary = summarize_distribution_features(original[key])
    excluded_summary = summarize_distribution_features(excluded[key])
    if not original_summary or not excluded_summary:
        return _component_result(
            component,
            None,
            {
                "original_summary": original_summary,
                "excluded_summary": excluded_summary,
            },
            [f"Empty {key}; component distance skipped."],
        )

    distance = normalized_l2_distance(original_summary, excluded_summary)
    return _component_result(
        component,
        _distance_or_zero(distance),
        {
            "summary_percentiles": [10.0, 50.0, 90.0],
            "original_summary": original_summary,
            "excluded_summary": excluded_summary,
            "distance": distance,
        },
    )


def _foreground_intensity_distance(original: Dict[str, Any], excluded: Dict[str, Any]) -> Dict[str, Any]:
    key = "foreground_intensity_properties_per_channel"
    if key not in original or key not in excluded:
        return _component_result(
            "foreground_intensity",
            None,
            {},
            [f"Missing {key}; component distance skipped."],
        )

    original_channels = {str(k): v for k, v in original[key].items()}
    excluded_channels = {str(k): v for k, v in excluded[key].items()}
    shared_channels, notes = _shared_channel_stats(original[key], excluded[key])
    if not shared_channels:
        return _component_result(
            "foreground_intensity",
            None,
            {"channels": {}},
            notes + ["No shared foreground intensity channels; component distance skipped."],
        )

    channel_details = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for channel in shared_channels:
        original_stats = original_channels[channel]
        excluded_stats = excluded_channels[channel]
        stat_distances = {}
        stat_values = []
        for stat_key in DEFAULT_INTENSITY_STAT_KEYS:
            if stat_key not in original_stats or stat_key not in excluded_stats:
                notes.append(f"Channel {channel} missing stat {stat_key}; stat skipped.")
                continue
            stat_distance = relative_absolute_difference(
                original_stats[stat_key],
                excluded_stats[stat_key],
            )
            stat_distances[stat_key] = stat_distance
            stat_values.append(_distance_or_zero(stat_distance))

        histogram_distance = None
        if "histogram" in original_stats and "histogram" in excluded_stats:
            try:
                histogram_distance = histogram_wasserstein_distance(
                    original_stats["histogram"],
                    excluded_stats["histogram"],
                )
                stat_values.append(_distance_or_zero(histogram_distance))
            except ValueError as exc:
                notes.append(f"Channel {channel} histogram skipped: {exc}")
        else:
            notes.append(f"Channel {channel} histogram missing; histogram distance skipped.")

        channel_distance = float(np.mean(stat_values)) if stat_values else 0.0
        channel_weight = max(
            float(original_stats.get("count", 1.0)),
            float(excluded_stats.get("count", 1.0)),
            1.0,
        )
        weighted_sum += channel_distance * channel_weight
        total_weight += channel_weight
        channel_details[channel] = {
            "normalized_distance": channel_distance,
            "weight": channel_weight,
            "stats": stat_distances,
            "histogram": histogram_distance,
        }

    distance = weighted_sum / total_weight if total_weight > 0 else 0.0
    return _component_result(
        "foreground_intensity",
        float(distance),
        {
            "channels": channel_details,
            "weighting": "max(original_count, excluded_count, 1) per channel",
        },
        notes,
    )


def _crop_distance(original: Dict[str, Any], excluded: Dict[str, Any]) -> Dict[str, Any]:
    details = {}
    notes = []
    distances = []

    key = "median_relative_size_after_cropping"
    if key in original and key in excluded:
        scalar_distance = relative_absolute_difference(original[key], excluded[key])
        details[key] = scalar_distance
        distances.append(_distance_or_zero(scalar_distance))
    else:
        notes.append(f"Missing {key}; scalar crop distance skipped.")

    hist_key = "relative_size_after_cropping_histogram"
    if hist_key in original and hist_key in excluded:
        try:
            histogram_distance = histogram_wasserstein_distance(
                original[hist_key],
                excluded[hist_key],
            )
            details[hist_key] = histogram_distance
            distances.append(_distance_or_zero(histogram_distance))
        except ValueError as exc:
            notes.append(f"{hist_key} skipped: {exc}")
    else:
        notes.append(f"Missing {hist_key}; histogram crop distance skipped.")

    distance = float(np.mean(distances)) if distances else None
    return _component_result("crop", distance, details, notes)


def _normalize_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    merged = dict(DEFAULT_WEIGHTS)
    if weights:
        merged.update({k: float(v) for k, v in weights.items()})
    total = sum(max(float(v), 0.0) for v in merged.values())
    if total <= 0:
        raise ValueError("At least one fingerprint distance weight must be positive")
    return {k: max(float(v), 0.0) / total for k, v in merged.items()}


def _decision_band(overall_distance: float, thresholds: Optional[Dict[str, float]]) -> str:
    if not thresholds or "low" not in thresholds or "high" not in thresholds:
        return "unthresholded"
    low = float(thresholds["low"])
    high = float(thresholds["high"])
    if low > high:
        raise ValueError("Threshold low must be <= high")
    if overall_distance < low:
        return "low"
    if overall_distance < high:
        return "mid"
    return "high"


def compute_fingerprint_distance(
    original: Dict[str, Any],
    excluded: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    normalized_weights = _normalize_weights(weights)
    components = {
        "spacing": _spacing_or_shape_distance(original, excluded, "spacings", "spacing"),
        "shape": _spacing_or_shape_distance(original, excluded, "shapes_after_crop", "shape"),
        "foreground_intensity": _foreground_intensity_distance(original, excluded),
        "crop": _crop_distance(original, excluded),
    }

    contributions = {}
    overall_distance = 0.0
    for component, result in components.items():
        distance = result.get("normalized_distance")
        contribution = normalized_weights[component] * (0.0 if distance is None else float(distance))
        contributions[component] = contribution
        overall_distance += contribution

    top_contributors = [
        {
            "component": component,
            "value": components[component].get("normalized_distance"),
            "weighted_value": contribution,
        }
        for component, contribution in sorted(
            contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    notes = []
    for component in components.values():
        notes.extend(component.get("notes", []))

    return {
        "overall_distance": float(overall_distance),
        "decision_band": _decision_band(overall_distance, thresholds),
        "weights": normalized_weights,
        "thresholds": thresholds,
        "components": components,
        "contributions": contributions,
        "top_contributors": top_contributors,
        "notes": notes,
    }
