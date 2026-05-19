"""Solar Orbiter / RPW reader.

Targets the L3 survey-flux CDFs distributed via the LESIA / CDPP archives:

  solo_L3_rpw-tnr-surv-flux_*.cdf    (~4 - 100 kHz, TNR receiver)
  solo_L3_rpw-hfr-surv-flux_*.cdf    (~0.4 - 16 MHz, HFR receiver)

Variables used:
  Epoch        : CDF_TT2000 timestamps
  FREQUENCY    : channel centres (kHz, may be 1-D or per-record 2-D)
  PSD_SFU      : calibrated flux density (SFU)
"""
from __future__ import annotations

import numpy as np

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, cdf_var_unit


def read(path, sensor: str = "AGC1", product: str = "auto", **_) -> DynamicSpectrum:
    """Load a Solar Orbiter / RPW L3 survey-flux CDF.

    Parameters
    ----------
    sensor : str
        Antenna configuration label (e.g. 'AGC1', 'AGC2'). L3 flux files are
        already calibrated, so the value is recorded in metadata but not used
        to pick a variable; kept for API parity with the L2 reader path.
    product : 'tnr' | 'hfr' | 'auto'
        Force TNR or HFR; 'auto' infers from the filename.
    """
    import cdflib

    p = resolve_path(path)
    product = _detect_product(p.name) if product == "auto" else product.lower()
    cdf = cdflib.CDF(str(p))

    times = cdf_epoch_to_datetime64(cdf.varget("Epoch"))

    freqs = np.asarray(cdf.varget("FREQUENCY"), dtype=float)
    # Some releases store frequency per record; collapse to 1-D.
    if freqs.ndim == 2:
        freqs = freqs[0]

    # Pull axis/intensity units from VAR_ATTRS rather than hard-coding; the
    # RPW L3 manual currently writes 'kHz', but earlier releases used 'Hz'.
    freq_unit = cdf_var_unit(cdf, "FREQUENCY", default="kHz")
    intensity_unit = cdf_var_unit(cdf, "PSD_SFU", default="SFU")

    data = np.squeeze(np.asarray(cdf.varget("PSD_SFU"), dtype=np.float32))

    return DynamicSpectrum(
        data=data.T,   # (n_time, n_freq) -> (n_freq, n_time)
        times=times,
        frequencies=freqs,
        freq_unit=freq_unit,
        instrument=f"SolO/RPW {product.upper()}",
        unit=intensity_unit,
        metadata={"sensor": sensor, "product": product.upper()},
    )


def _detect_product(name: str) -> str:
    low = name.lower()
    if "tnr" in low:
        return "tnr"
    if "hfr" in low:
        return "hfr"
    raise ValueError(
        f"cannot infer SolO/RPW product from filename {name!r}; "
        f"pass product='tnr' or 'hfr'"
    )