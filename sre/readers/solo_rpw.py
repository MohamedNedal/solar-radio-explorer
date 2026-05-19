"""Solar Orbiter / RPW reader (minimal, script-compatible version)."""

from __future__ import annotations

import numpy as np

from sre.readers.base import resolve_path
from sre.spectrum import DynamicSpectrum
from sre.utils import cdf_epoch_to_datetime64, ensure_mhz


def read(local_file) -> DynamicSpectrum:
    import cdflib
    from cdflib.epochs import CDFepoch

    cdf = cdflib.CDF(local_file)

    # --------------------
    # TIME (same as script)
    # --------------------
    epoch = cdf["Epoch"][:]
    times = CDFepoch.to_datetime(epoch)
    times = np.array(times, dtype='datetime64[ns]')

    # ------------------------
    # FREQUENCY (same as script)
    # ------------------------
    freq = cdf['FREQUENCY'][:]

    # ------------------------
    # DATA 
    # ------------------------
    data = np.squeeze(cdf['PSD_SFU'][:])

    # ------------------------
    # FORMAT OUTPUT
    # ------------------------
    freqs = ensure_mhz(freq, "kHz")

    return DynamicSpectrum(
        data=data.T,   # match (freq, time) expectation in many readers
        times=times,
        frequencies=freqs,
        instrument="SolO/RPW",
        unit="SFU",
        metadata={"sensor": "PSD_SFU"},
    )