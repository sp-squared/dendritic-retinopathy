"""Shared figure styling: human-readable names and consistent colours/markers,
so every figure in the report uses the same labels for the same model."""
from __future__ import annotations

import ast

DISPLAY_NAMES = {
    "dann-lrf": "dANN (local RF)",
    "dann-r": "dANN (random)",
    "vanilla": "Dense ANN",
}

# Fixed colour + marker per model type (colour-blind-safe Okabe-Ito hues;
# markers keep models distinguishable in greyscale print).
STYLES = {
    "dann-lrf": {"color": "#0072B2", "marker": "o"},
    "dann-r": {"color": "#E69F00", "marker": "s"},
    "vanilla": {"color": "#009E73", "marker": "^"},
}
DEFAULT_STYLE = {"color": "#555555", "marker": "D"}


def display_name(model: str) -> str:
    return DISPLAY_NAMES.get(model, model)


def style(model: str) -> dict:
    return STYLES.get(model, DEFAULT_STYLE)


def _first_int(value):
    """'[128]' / '[128, 64]' / 128 -> '128' or '128-64'; '' / NaN -> None."""
    if value is None or (isinstance(value, float) and value != value) or value == "":
        return None
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return str(value)
    if isinstance(parsed, (list, tuple)):
        return "-".join(str(v) for v in parsed)
    return str(parsed)


def config_label(row) -> str:
    """Readable label for one configuration, e.g. 'dANN (random), 128 somas'
    or 'Dense ANN, 64 hidden'. `row` is a dict-like with model/somas/hidden."""
    model = row["model"]
    if model == "vanilla":
        size = _first_int(row.get("hidden"))
        return f"{display_name(model)}, {size} hidden" if size else display_name(model)
    size = _first_int(row.get("somas"))
    return f"{display_name(model)}, {size} somas" if size else display_name(model)


def _human_count(x: float) -> str:
    """20000 -> '20k', 1500000 -> '1.5M'."""
    if x >= 1e6:
        s = f"{x / 1e6:.1f}".rstrip("0").rstrip(".")
        return f"{s}M"
    if x >= 1e3:
        s = f"{x / 1e3:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return f"{x:g}"


def format_params_axis(ax) -> None:
    """Log-scale x axis with plain labels (10k, 20k, 50k, 100k ...) at 1-2-5 steps,
    instead of overlapping scientific-notation minor labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, NullLocator

    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: _human_count(x)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
