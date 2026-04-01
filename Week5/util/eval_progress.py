"""
Utility for computing a weighted AP-change metric during training.

Metric:
    progress = sum_c [ (AP_c(t) - AP_c(t-1)) * w_c ]

Weights w_c give more importance to:
    - Classes with low AP       (hard classes)
    - Classes underrepresented in the dataset (rare classes)

Combined weight (normalised so weights sum to 1):
    w_c  ∝  (1 / (AP_c(t-1) + eps))^alpha  *  (1 / (freq_c + eps))^(1-alpha)
"""

import numpy as np


def compute_class_frequencies(dataset, num_classes: int) -> np.ndarray:
    """
    Compute relative class frequencies directly from the dataset's
    _labels_store — a list of lists of {'label': int, 'label_idx': int}.

    This avoids loading any frames and is essentially instant.
    Labels in _labels_store start at 1, matching the class_dict convention,
    so we map them to 0-indexed by subtracting 1.

    Returns
    -------
    freq : (num_classes,) array of relative frequencies, sums to 1.
           Index 0 corresponds to class 1 (same as the multi-hot label vector).
    """
    counts = np.zeros(num_classes, dtype=np.float64)

    for clip_labels in dataset._labels_store:
        for event in clip_labels:
            class_idx = event['label'] - 1      # labels start at 1 → 0-indexed
            if 0 <= class_idx < num_classes:
                counts[class_idx] += 1

    total = counts.sum()
    if total == 0:
        # Fallback: uniform if somehow no labels found
        return np.ones(num_classes, dtype=np.float64) / num_classes

    freq = counts / total
    return freq                                 # shape: (num_classes,)


def compute_weights(
    prev_ap: np.ndarray,
    class_frequencies: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Compute per-class weights that prioritise rare and low-AP classes.

    Parameters
    ----------
    prev_ap : (num_classes,) AP values from the previous eval step.
    class_frequencies : (num_classes,) relative frequency of each class.
    alpha : balance between AP-based (1.0) and frequency-based (0.0) weighting.
    eps : smoothing term to avoid division by zero.

    Returns
    -------
    w : (num_classes,) normalised weights summing to 1.
    """
    ap_weight   = (1.0 / (prev_ap            + eps)) ** alpha
    freq_weight = (1.0 / (class_frequencies  + eps)) ** (1.0 - alpha)

    w = ap_weight * freq_weight
    w = w / (w.sum() + 1e-12)          # normalise
    return w


def compute_weighted_ap_change(
    current_ap: np.ndarray,
    prev_ap: np.ndarray,
    class_frequencies: np.ndarray,
    alpha: float = 0.5,
    eps: float = 1e-3,
) -> dict:
    """
    Core metric.  Returns a dict with the scalar progress metric and
    supporting diagnostics you can log / store.

    Parameters
    ----------
    current_ap        : (num_classes,) AP at current eval step.
    prev_ap           : (num_classes,) AP at previous eval step.
    class_frequencies : (num_classes,) relative class frequency in dataset.
    alpha             : weight balance (see compute_weights).
    eps               : smoothing term.

    Returns
    -------
    dict with keys:
        'weighted_change'   – the scalar metric  Σ (ΔAP_c * w_c)
        'per_class_change'  – (num_classes,) raw ΔAP per class
        'weights'           – (num_classes,) w_c used
        'mean_ap'           – unweighted mAP at current step
        'delta_mean_ap'     – change in unweighted mAP
    """
    delta_ap = current_ap - prev_ap                         # (num_classes,)
    weights  = compute_weights(prev_ap, class_frequencies, alpha, eps)

    weighted_change = float(np.dot(delta_ap, weights))      # scalar

    return {
        'weighted_change' : weighted_change,
        'per_class_change': delta_ap,
        'weights'         : weights,
        'mean_ap'         : float(np.mean(current_ap)),
        'delta_mean_ap'   : float(np.mean(current_ap) - np.mean(prev_ap)),
    }