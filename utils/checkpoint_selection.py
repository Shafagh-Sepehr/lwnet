"""Validation metric policy parsing and ordered checkpoint comparison."""

import math


METRICS = ("auc", "dice", "loss", "tr_auc")
DEFAULT_TOLERANCES = {"auc": 0.0005, "dice": 0.0001, "loss": 0.000001}


def parse_metric_policy(metric, metric_tolerances=None):
    names = [part.strip().lower() for part in str(metric).split(",")]
    if not names or any(not name for name in names):
        raise ValueError("--metric must contain one or more non-empty metric names")
    unknown = [name for name in names if name not in METRICS]
    if unknown:
        raise ValueError("unknown metric(s): {}".format(", ".join(unknown)))
    if len(set(names)) != len(names):
        raise ValueError("--metric must not contain duplicate metrics")
    if "tr_auc" in names and len(names) != 1:
        raise ValueError("tr_auc cannot be combined with validation metrics")

    overrides = {}
    if metric_tolerances:
        for item in str(metric_tolerances).split(","):
            item = item.strip()
            if not item or "=" not in item:
                raise ValueError("metric tolerances must use name=value syntax")
            name, value = (part.strip().lower() for part in item.split("=", 1))
            if name in overrides:
                raise ValueError("duplicate tolerance assignment: {}".format(name))
            if name not in names:
                raise ValueError("tolerance provided for unselected metric: {}".format(name))
            try:
                value = float(value)
            except ValueError:
                raise ValueError("invalid tolerance for {}".format(name))
            if not math.isfinite(value) or value < 0:
                raise ValueError("tolerance for {} must be finite and non-negative".format(name))
            overrides[name] = value

    tolerances = {name: 0.0 for name in names}
    if len(names) > 1:
        for name in names:
            tolerances[name] = DEFAULT_TOLERANCES[name]
    tolerances.update(overrides)
    return {
        "metric_order": names,
        "metric_tolerances": tolerances,
        "metric": ",".join(names),
    }


def compare_checkpoint(current, incumbent, order, tolerances):
    """Return ``(selected, reason, details)`` using the incumbent pairwise rule."""
    missing = [name for name in order if name not in current]
    if incumbent is not None:
        missing += [name for name in order if name not in incumbent and name not in missing]
    if missing:
        raise KeyError("missing requested metric(s): {}".format(", ".join(missing)))
    if any(not math.isfinite(float(current[name])) for name in order):
        return False, "invalid_candidate", {"tie_chain": [], "differences": {}}
    if incumbent is None:
        return True, "first_valid_candidate", {"tie_chain": [], "differences": {}}

    tie_chain = []
    differences = {}
    for name in order:
        direction = -1.0 if name == "loss" else 1.0
        delta = direction * (float(current[name]) - float(incumbent[name]))
        differences[name] = delta
        tolerance = float(tolerances[name])
        if delta == 0.0 or abs(delta) < tolerance:
            tie_chain.append(name)
            continue
        return delta > 0.0, name, {"tie_chain": tie_chain, "differences": differences}
    return False, "all_tied", {"tie_chain": tie_chain, "differences": differences}
