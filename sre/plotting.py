"""Publication-quality plotting for DynamicSpectrum objects."""
from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from sre.processing import percentile_clip


PUBLICATION_RC = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
}


def plot_spectrum(ds, cmap: str = "inferno", vmin: Optional[float] = None,
                  vmax: Optional[float] = None, vmin_pct: float = 1.0,
                  vmax_pct: float = 99.0, log_freq: bool = False,
                  norm: str = "linear",
                  linthresh: Optional[float] = None,
                  title: Optional[str] = None,
                  xlabel: str = "Time (UTC)",
                  ylabel: Optional[str] = None,
                  cbar_label: Optional[str] = None,
                  figsize=(8.0, 4.0),
                  publication_style: bool = True,
                  rasterized: bool = True):
    """Return a matplotlib Figure of the dynamic spectrum.

    Parameters
    ----------
    norm : {'linear', 'log', 'symlog'}
        Intensity normalisation. 'log' is appropriate for raw flux (which spans
        several decades); 'symlog' handles spectra with both positive and
        negative values (e.g. after background subtraction).
    linthresh : float, optional
        Symlog linear-region half-width. Auto-derived from the data if omitted.
    ylabel : str, optional
        Frequency-axis label. If omitted, derived from ``ds.freq_unit``.
    """
    rc = PUBLICATION_RC if publication_style else {}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        if vmin is None or vmax is None:
            v_lo, v_hi = percentile_clip(ds.data, vmin_pct, vmax_pct)
            vmin = v_lo if vmin is None else vmin
            vmax = v_hi if vmax is None else vmax

        cnorm = _build_norm(ds.data, norm, vmin, vmax, linthresh)

        t_mpl = mdates.date2num(pd.to_datetime(ds.times).to_pydatetime())
        # Choose a display unit so tick labels stay readable (≤ 3 digits).
        native_unit = getattr(ds, "freq_unit", "MHz")
        display_unit, scale = _auto_freq_unit(float(np.nanmax(ds.frequencies)),
                                              native_unit)
        f = np.asarray(ds.frequencies, dtype=float) * scale
        # pcolormesh wants cell edges; derive them from centres.
        t_edges = _edges_from_centres(t_mpl)
        f_edges = _edges_from_centres(f)

        im = ax.pcolormesh(
            t_edges, f_edges, ds.data,
            cmap=cmap, norm=cnorm,
            shading="auto", rasterized=rasterized,
        )

        if log_freq:
            ax.set_yscale("log")
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel if ylabel is not None
                      else f"Frequency ({display_unit})")

        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        if title is None:
            title = f"{ds.instrument} — {pd.Timestamp(ds.times[0]).strftime('%Y-%m-%d')}"
        ax.set_title(title)

        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(cbar_label or ds.unit)
        cb.outline.set_linewidth(0.5)

    return fig


_FREQ_UNIT_HZ = {
    "hz": 1.0, "hertz": 1.0,
    "khz": 1e3, "kilohertz": 1e3,
    "mhz": 1e6, "megahertz": 1e6,
    "ghz": 1e9, "gigahertz": 1e9,
}


def _auto_freq_unit(fmax_native: float, native_unit: str) -> tuple[str, float]:
    """Pick a display unit so the largest tick has ≤ 3 integer digits.

    Returns (display_unit, multiply_by) where ``frequencies_native * multiply_by``
    yields values in the chosen unit. Unknown units are passed through unchanged.
    """
    native_hz = _FREQ_UNIT_HZ.get(str(native_unit).lower().strip())
    if native_hz is None or not np.isfinite(fmax_native) or fmax_native <= 0:
        return native_unit, 1.0

    fmax_hz = fmax_native * native_hz
    if fmax_hz < 1e3:
        target_unit, target_hz = "Hz", 1.0
    elif fmax_hz < 1e6:
        target_unit, target_hz = "kHz", 1e3
    elif fmax_hz < 1e9:
        target_unit, target_hz = "MHz", 1e6
    else:
        target_unit, target_hz = "GHz", 1e9
    return target_unit, native_hz / target_hz


def _build_norm(data: np.ndarray, kind: str, vmin: float, vmax: float,
                linthresh: Optional[float]):
    """Construct a matplotlib Normalize from a string spec.

    'log' falls back to symlog if the data contain non-positive values, so a
    background-subtracted spectrum still renders without raising.
    """
    kind = (kind or "linear").lower()
    if kind == "linear":
        return Normalize(vmin=vmin, vmax=vmax)

    if kind == "log":
        finite_pos = data[np.isfinite(data) & (data > 0)]
        if finite_pos.size == 0:
            # No positive samples — log makes no sense; degrade to symlog.
            kind = "symlog"
        else:
            lo = max(vmin, float(finite_pos.min())) if vmin and vmin > 0 else float(finite_pos.min())
            hi = max(vmax, lo * 10.0)
            return LogNorm(vmin=lo, vmax=hi)

    if kind == "symlog":
        if linthresh is None:
            # Pick a small fraction of the dynamic range so the linear region
            # captures noise around zero without flattening real structure.
            scale = max(abs(vmin), abs(vmax), 1.0)
            linthresh = max(scale * 1e-3, 1e-12)
        return SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)

    raise ValueError(f"unknown norm {kind!r}; use 'linear', 'log', or 'symlog'")


def _edges_from_centres(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    if len(c) == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])
