"""Le prétraitement vectorisé de mlrp égale celui du code de 2024 sur des données aléatoires avec trous."""

import numpy as np
import pandas as pd

from ml_returns_pred.preprocess_data.data_preprocessor import DataPreprocessor
from ml_returns_pred.resample_data.data_resampler import DataResampler
from mlrp.data import (
    align_common_period,
    arithmetic_returns,
    binarize,
    forward_fill_within_history,
    monthly_frequency,
    resample_to_reference,
    synthetic_dataset,
    truncate,
)


def _prices():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2012-01-02", periods=520)
    p = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(days), 5)), axis=0)), index=days,
                     columns=list("ABCDE"))
    p.iloc[:30, 0] = np.nan          # début tardif
    p.iloc[100:110, 1] = np.nan      # trou interne
    p.iloc[480:, 2] = np.nan         # fin anticipée
    return p


def test_forward_fill_within_history_matches_legacy():
    p = _prices()
    new = forward_fill_within_history(p)
    old = DataPreprocessor().preprocess(data=p.copy())
    pd.testing.assert_frame_equal(new, old)
    assert new.iloc[:30, 0].isna().all() and new.iloc[480:, 2].isna().all()
    assert new.iloc[100:110, 1].notna().all()


def test_resample_matches_legacy():
    p = forward_fill_within_history(_prices())
    months = pd.date_range("2012-02-01", periods=20, freq="MS")
    macro = pd.DataFrame(np.random.default_rng(2).normal(size=(20, 3)), index=months, columns=list("xyz"))
    pa, ma = align_common_period(p, macro)
    new = resample_to_reference(pa, ma)
    dr = DataResampler()
    old = dr.specify_datetime_index_frequency(dr.resample_and_forward_fill(data_to_resample=pa.copy(),
                                                                           reference_data=ma.copy()), freq="MS")
    pd.testing.assert_frame_equal(new, old)
    pd.testing.assert_frame_equal(monthly_frequency(ma), dr.specify_datetime_index_frequency(ma.copy(), "MS"))


def test_truncate_returns_binarize():
    ds = synthetic_dataset(n_months=24, n_assets=4)
    r = arithmetic_returns(ds.prices_monthly)
    assert len(r) == len(ds.prices_monthly) - 1
    b = binarize(r)
    assert set(np.unique(b.values)) <= {0, 1}
    t = truncate(ds.prices_daily, "2010-06-30")
    assert t.index.max() <= pd.Timestamp("2010-06-30")
    assert ds.exog_monthly.index.equals(ds.returns_monthly.index)
    assert len(ds.fingerprint()) == 12
