#!/usr/bin/env python3
"""
Self-contained tests for eval_progress.py and the training loop logic
introduced in train.py.

No GPU, no real data, no model needed — runs anywhere with just numpy.

Usage:
    python test_eval_progress.py
    python test_eval_progress.py -v      # verbose: print all intermediate values
"""

import sys
import math
import argparse
import numpy as np
import traceback

# ── tiny pretty-printer ──────────────────────────────────────────────────────

VERBOSE = False

def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ── colour helpers (degrade gracefully on Windows) ───────────────────────────

def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"

# ── copy of the functions under test (no import needed) ─────────────────────
# We inline them here so the test file is truly standalone and does not depend
# on the project folder structure being set up correctly on your local machine.

def compute_class_frequencies(dataset, num_classes: int) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for clip_labels in dataset._labels_store:
        for event in clip_labels:
            class_idx = event['label'] - 1
            if 0 <= class_idx < num_classes:
                counts[class_idx] += 1
    total = counts.sum()
    if total == 0:
        return np.ones(num_classes, dtype=np.float64) / num_classes
    return counts / total


def compute_weights(prev_ap, class_frequencies, alpha=0.5, eps=1e-3):
    ap_weight   = (1.0 / (prev_ap            + eps)) ** alpha
    freq_weight = (1.0 / (class_frequencies  + eps)) ** (1.0 - alpha)
    w = ap_weight * freq_weight
    w = w / (w.sum() + 1e-12)
    return w


def compute_weighted_ap_change(current_ap, prev_ap, class_frequencies,
                               alpha=0.5, eps=1e-3):
    delta_ap = current_ap - prev_ap
    weights  = compute_weights(prev_ap, class_frequencies, alpha, eps)
    weighted_change = float(np.dot(delta_ap, weights))
    return {
        'weighted_change' : weighted_change,
        'per_class_change': delta_ap,
        'weights'         : weights,
        'mean_ap'         : float(np.mean(current_ap)),
        'delta_mean_ap'   : float(np.mean(current_ap) - np.mean(prev_ap)),
    }

# ── SoccerNet-like classes (from your AP table) ──────────────────────────────

CLASSES = {
    'PASS': 1, 'DRIVE': 2, 'HEADER': 3, 'HIGH PASS': 4,
    'OUT': 5, 'CROSS': 6, 'THROW IN': 7, 'SHOT': 8,
    'BALL PLAYER BLOCK': 9, 'PLAYER SUCCESSFUL TACKLE': 10,
    'FREE KICK': 11, 'GOAL': 12,
}
NUM_CLASSES = len(CLASSES)

# Approximate AP values from your screenshot (0-indexed, same order as CLASSES)
REAL_AP = np.array([
    0.7515, 0.7141, 0.1091, 0.2184,
    0.1601, 0.1606, 0.0956, 0.1360,
    0.0992, 0.0146, 0.0208, 0.0047,
])

# ── mock dataset ─────────────────────────────────────────────────────────────

class MockDataset:
    """
    Mimics ActionSpotDataset well enough for compute_class_frequencies.
    _labels_store is a list of clip label lists, each entry is
    {'label': int (1-indexed), 'label_idx': int}.
    """
    def __init__(self, labels_store):
        self._labels_store = labels_store

    @classmethod
    def from_counts(cls, counts_per_class: list) -> 'MockDataset':
        """
        Build a fake _labels_store where class i+1 appears counts_per_class[i]
        times, each in its own clip (simplest possible structure).
        """
        store = []
        for class_idx_0, count in enumerate(counts_per_class):
            for _ in range(count):
                store.append([{'label': class_idx_0 + 1, 'label_idx': 0}])
        return cls(store)

# ── test harness ─────────────────────────────────────────────────────────────

results = []   # list of (name, passed, message)

def run(name, fn):
    try:
        fn()
        results.append((name, True, ''))
        print(f"  {green('PASS')}  {name}")
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"  {red('FAIL')}  {name}")
        print(f"         {e}")
        if VERBOSE:
            traceback.print_exc()
    except Exception as e:
        results.append((name, False, f"EXCEPTION: {e}"))
        print(f"  {red('ERR ')}  {name}")
        print(f"         {e}")
        if VERBOSE:
            traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 1 — compute_class_frequencies
# ─────────────────────────────────────────────────────────────────────────────

print(bold("\n── Group 1: compute_class_frequencies ──"))

def test_freq_shape():
    ds = MockDataset.from_counts([10] * NUM_CLASSES)
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    assert freq.shape == (NUM_CLASSES,), \
        f"Expected shape ({NUM_CLASSES},), got {freq.shape}"

def test_freq_sums_to_one():
    ds = MockDataset.from_counts([5, 50, 100, 3, 20, 8, 1, 15, 7, 2, 4, 1])
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    assert abs(freq.sum() - 1.0) < 1e-9, \
        f"Frequencies should sum to 1, got {freq.sum()}"

def test_freq_correct_proportions():
    # Class 1 → 90 events, class 2 → 10 events, rest → 0
    counts = [90, 10] + [0] * (NUM_CLASSES - 2)
    ds = MockDataset.from_counts(counts)
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    vprint(f"    freq[0]={freq[0]:.4f}  freq[1]={freq[1]:.4f}")
    assert abs(freq[0] - 0.9) < 1e-9, f"Expected freq[0]=0.9, got {freq[0]}"
    assert abs(freq[1] - 0.1) < 1e-9, f"Expected freq[1]=0.1, got {freq[1]}"
    assert abs(freq[2:].sum()) < 1e-9, \
        f"Expected freq[2:]=0, got {freq[2:].sum()}"

def test_freq_label_indexing():
    # Labels in _labels_store start at 1; make sure they map to 0-indexed correctly
    store = [
        [{'label': 1, 'label_idx': 0}],   # class 1 → index 0
        [{'label': 1, 'label_idx': 0}],
        [{'label': 12, 'label_idx': 3}],  # class 12 → index 11
    ]
    ds = MockDataset(store)
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    vprint(f"    freq[0]={freq[0]:.4f}  freq[11]={freq[11]:.4f}")
    assert abs(freq[0] - 2/3) < 1e-9, \
        f"Class 1 (idx 0) should have freq 2/3, got {freq[0]}"
    assert abs(freq[11] - 1/3) < 1e-9, \
        f"Class 12 (idx 11) should have freq 1/3, got {freq[11]}"

def test_freq_empty_dataset_fallback():
    # No labels at all → uniform fallback, must not crash
    ds = MockDataset([])
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    expected = 1.0 / NUM_CLASSES
    assert all(abs(f - expected) < 1e-9 for f in freq), \
        f"Expected uniform fallback {expected}, got {freq}"

def test_freq_clips_with_no_events_are_ignored():
    # Clips with empty label lists should contribute 0 counts
    store = [
        [],                                          # no event
        [{'label': 3, 'label_idx': 5}],
        [],
    ]
    ds = MockDataset(store)
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    assert abs(freq[2] - 1.0) < 1e-9, \
        f"Only class 3 (idx 2) present, expected freq=1.0, got {freq[2]}"

def test_freq_multi_event_clip():
    # A clip can have multiple events (possibly different classes)
    store = [
        [{'label': 1, 'label_idx': 0}, {'label': 2, 'label_idx': 5}],
        [{'label': 1, 'label_idx': 2}],
    ]
    ds = MockDataset(store)
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    vprint(f"    freq[0]={freq[0]:.4f}  freq[1]={freq[1]:.4f}")
    assert abs(freq[0] - 2/3) < 1e-9, f"Expected 2/3, got {freq[0]}"
    assert abs(freq[1] - 1/3) < 1e-9, f"Expected 1/3, got {freq[1]}"

def test_freq_out_of_range_label_ignored():
    # A label of 0 or > num_classes should be silently skipped
    store = [
        [{'label': 0,  'label_idx': 0}],   # invalid: 0-1 = -1 → skipped
        [{'label': 13, 'label_idx': 0}],   # invalid: 13-1 = 12 → skipped (≥ NUM_CLASSES)
        [{'label': 1,  'label_idx': 0}],   # valid
    ]
    ds = MockDataset(store)
    freq = compute_class_frequencies(ds, NUM_CLASSES)
    assert abs(freq.sum() - 1.0) < 1e-9, "Should still sum to 1 with bad labels filtered"
    assert abs(freq[0] - 1.0) < 1e-9, "Only class 1 (idx 0) is valid"

run("output shape is (num_classes,)",          test_freq_shape)
run("frequencies sum to 1",                   test_freq_sums_to_one)
run("proportions are correct (90/10 split)",  test_freq_correct_proportions)
run("label 1-indexing maps to 0-indexed",     test_freq_label_indexing)
run("empty dataset → uniform fallback",       test_freq_empty_dataset_fallback)
run("clips with no events are ignored",       test_freq_clips_with_no_events_are_ignored)
run("multi-event clips counted correctly",    test_freq_multi_event_clip)
run("out-of-range labels are silently skipped", test_freq_out_of_range_label_ignored)

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 2 — compute_weights
# ─────────────────────────────────────────────────────────────────────────────

print(bold("\n── Group 2: compute_weights ──"))

UNIFORM_FREQ = np.ones(NUM_CLASSES) / NUM_CLASSES

def test_weights_sum_to_one():
    w = compute_weights(REAL_AP, UNIFORM_FREQ)
    assert abs(w.sum() - 1.0) < 1e-9, f"Weights should sum to 1, got {w.sum()}"

def test_weights_all_positive():
    w = compute_weights(REAL_AP, UNIFORM_FREQ)
    assert (w > 0).all(), f"All weights must be positive, got {w}"

def test_weights_rare_class_gets_more_weight():
    # With alpha=0 (freq-only): rarest class must have highest weight
    freq = np.array([0.5, 0.3, 0.1, 0.1] + [0.0] * (NUM_CLASSES - 4))
    freq = freq / freq.sum()
    ap   = np.ones(NUM_CLASSES) * 0.5         # equal AP → freq drives everything
    w    = compute_weights(ap, freq, alpha=0.0)
    vprint(f"    weights (freq-only): {np.round(w, 4)}")
    # classes 2 and 3 (idx 2,3) are equally rare and should beat classes 0,1
    assert w[2] > w[0], "Rarer class should get higher weight (freq-only)"
    assert w[3] > w[1], "Rarer class should get higher weight (freq-only)"

def test_weights_low_ap_class_gets_more_weight():
    # With alpha=1 (AP-only): lowest-AP class must have highest weight
    freq = UNIFORM_FREQ.copy()
    ap   = REAL_AP.copy()
    w    = compute_weights(ap, freq, alpha=1.0)
    vprint(f"    lowest AP class: {np.argmin(ap)} (GOAL), highest weight: {np.argmax(w)}")
    assert np.argmax(w) == np.argmin(ap), \
        f"Class with lowest AP should have highest weight. " \
        f"lowest AP idx={np.argmin(ap)}, highest weight idx={np.argmax(w)}"

def test_weights_alpha_0_ignores_ap():
    # alpha=0 → weight formula doesn't use AP at all; changing AP shouldn't matter
    freq = UNIFORM_FREQ.copy()
    ap_a = np.ones(NUM_CLASSES) * 0.1
    ap_b = np.ones(NUM_CLASSES) * 0.9
    wa = compute_weights(ap_a, freq, alpha=0.0)
    wb = compute_weights(ap_b, freq, alpha=0.0)
    assert np.allclose(wa, wb, atol=1e-9), \
        "alpha=0 → weights must not depend on AP values"

def test_weights_alpha_1_ignores_freq():
    # alpha=1 → weight formula doesn't use freq at all
    ap   = REAL_AP.copy()
    freq_a = np.ones(NUM_CLASSES) / NUM_CLASSES
    freq_b = np.array([0.9] + [0.1 / (NUM_CLASSES-1)] * (NUM_CLASSES-1))
    wa = compute_weights(ap, freq_a, alpha=1.0)
    wb = compute_weights(ap, freq_b, alpha=1.0)
    assert np.allclose(wa, wb, atol=1e-9), \
        "alpha=1 → weights must not depend on class frequencies"

def test_weights_uniform_ap_uniform_freq_gives_uniform_weights():
    ap   = np.ones(NUM_CLASSES) * 0.5
    freq = np.ones(NUM_CLASSES) / NUM_CLASSES
    w    = compute_weights(ap, freq)
    expected = 1.0 / NUM_CLASSES
    assert np.allclose(w, expected, atol=1e-9), \
        f"Uniform AP + uniform freq should give uniform weights, got {w}"

run("weights sum to 1",                          test_weights_sum_to_one)
run("all weights are positive",                  test_weights_all_positive)
run("rare class gets more weight (alpha=0)",     test_weights_rare_class_gets_more_weight)
run("low-AP class gets more weight (alpha=1)",   test_weights_low_ap_class_gets_more_weight)
run("alpha=0 → weights independent of AP",       test_weights_alpha_0_ignores_ap)
run("alpha=1 → weights independent of freq",     test_weights_alpha_1_ignores_freq)
run("uniform AP + uniform freq → uniform weights", test_weights_uniform_ap_uniform_freq_gives_uniform_weights)

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 3 — compute_weighted_ap_change
# ─────────────────────────────────────────────────────────────────────────────

print(bold("\n── Group 3: compute_weighted_ap_change ──"))

FREQ = np.ones(NUM_CLASSES) / NUM_CLASSES   # uniform for simplicity here

def test_wac_return_keys():
    out = compute_weighted_ap_change(REAL_AP, REAL_AP * 0.9, FREQ)
    for key in ('weighted_change', 'per_class_change', 'weights', 'mean_ap', 'delta_mean_ap'):
        assert key in out, f"Missing key '{key}' in output"

def test_wac_no_change_gives_zero():
    out = compute_weighted_ap_change(REAL_AP, REAL_AP, FREQ)
    vprint(f"    weighted_change (no change): {out['weighted_change']}")
    assert abs(out['weighted_change']) < 1e-9, \
        f"No AP change should give weighted_change=0, got {out['weighted_change']}"

def test_wac_uniform_improvement_is_positive():
    # All classes improve by the same amount → weighted change must be positive
    delta = 0.05
    out = compute_weighted_ap_change(REAL_AP + delta, REAL_AP, FREQ)
    vprint(f"    weighted_change (+{delta} uniform): {out['weighted_change']:.6f}")
    assert out['weighted_change'] > 0, \
        f"Uniform improvement should give positive weighted change, got {out['weighted_change']}"

def test_wac_uniform_degradation_is_negative():
    delta = 0.05
    out = compute_weighted_ap_change(REAL_AP - delta, REAL_AP, FREQ)
    vprint(f"    weighted_change (-{delta} uniform): {out['weighted_change']:.6f}")
    assert out['weighted_change'] < 0, \
        f"Uniform degradation should give negative weighted change"

def test_wac_per_class_change_correct():
    prev    = np.array([0.5] * NUM_CLASSES)
    current = np.array([0.6] * NUM_CLASSES)
    out = compute_weighted_ap_change(current, prev, FREQ)
    expected_delta = np.array([0.1] * NUM_CLASSES)
    assert np.allclose(out['per_class_change'], expected_delta, atol=1e-9), \
        f"per_class_change incorrect: {out['per_class_change']}"

def test_wac_mean_ap_correct():
    out = compute_weighted_ap_change(REAL_AP, REAL_AP * 0.9, FREQ)
    assert abs(out['mean_ap'] - float(np.mean(REAL_AP))) < 1e-9, \
        f"mean_ap mismatch: {out['mean_ap']} vs {np.mean(REAL_AP)}"

def test_wac_delta_mean_ap_correct():
    prev    = REAL_AP * 0.9
    current = REAL_AP
    out     = compute_weighted_ap_change(current, prev, FREQ)
    expected = float(np.mean(current) - np.mean(prev))
    assert abs(out['delta_mean_ap'] - expected) < 1e-9, \
        f"delta_mean_ap mismatch: {out['delta_mean_ap']} vs {expected}"

def test_wac_rare_class_improvement_outweighs_common_class_degradation():
    """
    Rare/low-AP class (GOAL, idx 11) improves a lot.
    Common/high-AP class (PASS, idx 0) degrades slightly.
    Because GOAL has a much higher weight, the metric should still be positive.
    """
    freq = np.array([0.4, 0.2, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03,
                     0.03, 0.02, 0.01, 0.004])
    freq = freq / freq.sum()

    prev    = REAL_AP.copy()
    current = REAL_AP.copy()
    current[0]  -= 0.02    # PASS degrades a little
    current[11] += 0.10    # GOAL improves a lot

    out = compute_weighted_ap_change(current, prev, freq, alpha=0.5)
    vprint(f"    weight PASS={out['weights'][0]:.4f}  weight GOAL={out['weights'][11]:.4f}")
    vprint(f"    weighted_change = {out['weighted_change']:.6f}")
    assert out['weighted_change'] > 0, \
        "Large improvement in rare/hard class should dominate small common-class degradation"

def test_wac_manual_calculation():
    """
    2-class toy example where we can compute the expected answer by hand.
    """
    nc       = 2
    prev     = np.array([0.8, 0.2])
    current  = np.array([0.9, 0.3])
    freq     = np.array([0.7, 0.3])
    alpha    = 0.5
    eps      = 1e-3

    ap_w  = (1.0 / (prev + eps)) ** alpha
    fr_w  = (1.0 / (freq + eps)) ** (1.0 - alpha)
    w_raw = ap_w * fr_w
    w     = w_raw / w_raw.sum()

    expected_wc = float(np.dot(current - prev, w))

    out = compute_weighted_ap_change(current, prev, freq, alpha=alpha, eps=eps)
    vprint(f"    expected={expected_wc:.8f}  got={out['weighted_change']:.8f}")
    assert abs(out['weighted_change'] - expected_wc) < 1e-9, \
        f"Manual calculation mismatch: expected {expected_wc}, got {out['weighted_change']}"

run("output contains all required keys",                    test_wac_return_keys)
run("no AP change → weighted_change = 0",                  test_wac_no_change_gives_zero)
run("uniform improvement → positive metric",               test_wac_uniform_improvement_is_positive)
run("uniform degradation → negative metric",               test_wac_uniform_degradation_is_negative)
run("per_class_change values are correct",                 test_wac_per_class_change_correct)
run("mean_ap value is correct",                            test_wac_mean_ap_correct)
run("delta_mean_ap value is correct",                      test_wac_delta_mean_ap_correct)
run("rare class gain outweighs common class loss",         test_wac_rare_class_improvement_outweighs_common_class_degradation)
run("manual 2-class calculation matches exactly",          test_wac_manual_calculation)

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 4 — early stopping logic (simulated training loop)
# ─────────────────────────────────────────────────────────────────────────────

print(bold("\n── Group 4: early stopping logic (simulated loop) ──"))

def simulate_training(ap_sequence, patience=3, min_improvement=0.0):
    """
    Replays the early-stopping logic from train.py using a pre-defined sequence
    of AP arrays (one per eval step).  Returns the epoch at which training
    would stop (0-indexed into ap_sequence), or None if it runs to the end.
    """
    freq     = np.ones(NUM_CLASSES) / NUM_CLASSES
    prev_ap  = None
    no_improve = 0

    for i, current_ap in enumerate(ap_sequence):
        if prev_ap is None:
            prev_ap = current_ap
            continue

        progress = compute_weighted_ap_change(current_ap, prev_ap, freq)
        wc = progress['weighted_change']

        if wc <= min_improvement:
            no_improve += 1
            if no_improve >= patience:
                return i          # stopped at this eval step
        else:
            no_improve = 0

        prev_ap = current_ap

    return None  # ran to completion

def test_early_stop_triggers_after_patience():
    # AP flatlines → should stop exactly at patience steps
    flat = [REAL_AP.copy()] * 10
    stopped_at = simulate_training(flat, patience=3)
    vprint(f"    stopped at eval step: {stopped_at}")
    assert stopped_at == 3, \
        f"Should stop at eval step 3 (patience=3), stopped at {stopped_at}"

def test_no_early_stop_when_improving():
    # AP steadily improves → should never stop
    improving = [REAL_AP + i * 0.01 for i in range(10)]
    stopped_at = simulate_training(improving, patience=3)
    assert stopped_at is None, \
        f"Steadily improving AP should never trigger early stop, stopped at {stopped_at}"

def test_early_stop_resets_on_improvement():
    # 2 bad evals, then 1 good, then 3 bad → stops at step 6, not step 2
    seq = (
        [REAL_AP.copy()] * 3           # flat (eval 0,1,2)
        + [REAL_AP + 0.05]             # improvement resets counter (eval 3)
        + [REAL_AP + 0.05] * 3        # flat again (eval 4,5,6)
    )
    stopped_at = simulate_training(seq, patience=3)
    vprint(f"    stopped at eval step: {stopped_at}")
    assert stopped_at == 6, \
        f"Counter should reset on improvement; expected stop at 6, got {stopped_at}"

def test_early_stop_never_fires_on_first_eval():
    # First eval has no prev_ap → no delta computed, cannot stop
    seq = [REAL_AP.copy(), REAL_AP.copy()]
    stopped_at = simulate_training(seq, patience=1)
    # With patience=1 and flat AP, it should stop at step 1 (the second eval),
    # not at step 0 (where prev_ap is None and we skip)
    assert stopped_at == 1, \
        f"Should stop at step 1 (not 0), got {stopped_at}"

run("stops after exactly <patience> flat evals",      test_early_stop_triggers_after_patience)
run("never stops when AP steadily improves",          test_no_early_stop_when_improving)
run("counter resets after improvement",               test_early_stop_resets_on_improvement)
run("first eval never triggers early stop",           test_early_stop_never_fires_on_first_eval)

# ─────────────────────────────────────────────────────────────────────────────
# GROUP 5 — eval_every scheduling
# ─────────────────────────────────────────────────────────────────────────────

print(bold("\n── Group 5: eval_every scheduling ──"))

def eval_epochs(num_epochs, eval_every_pct):
    """Return the list of epochs at which an AP eval would fire."""
    eval_every = max(1, int(round(num_epochs * eval_every_pct)))
    fired = [(e, True) for e in range(num_epochs)
             if (e + 1) % eval_every == 0 or e == num_epochs - 1]
    return eval_every, [e for e, _ in fired]

def test_eval_fires_at_last_epoch():
    for n in [7, 10, 20, 50]:
        _, epochs = eval_epochs(n, 0.10)
        assert n - 1 in epochs, f"Last epoch {n-1} not in eval epochs for n={n}"

def test_eval_frequency_approx_correct():
    eval_every, fired = eval_epochs(100, 0.10)
    assert eval_every == 10, f"Expected eval_every=10 for 100 epochs at 10%, got {eval_every}"

def test_eval_every_1_when_few_epochs():
    # 3 epochs at 10% → round(0.3)=0 → max(1,0)=1 → eval every epoch
    eval_every, _ = eval_epochs(3, 0.10)
    assert eval_every == 1, f"Expected eval_every=1 for tiny run, got {eval_every}"

run("eval always fires on last epoch",         test_eval_fires_at_last_epoch)
run("eval_every=10 for 100 epochs at 10%",    test_eval_frequency_approx_correct)
run("eval_every≥1 even for very few epochs",  test_eval_every_1_when_few_epochs)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

total  = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(bold(f"\n{'─'*50}"))
print(bold(f"  Results: {passed}/{total} passed"))
if failed:
    print(red(f"  {failed} test(s) FAILED:"))
    for name, ok, msg in results:
        if not ok:
            print(f"    • {name}")
            if msg:
                print(f"      {msg}")
else:
    print(green("  All tests passed ✓"))
print(bold(f"{'─'*50}\n"))

sys.exit(0 if failed == 0 else 1)
