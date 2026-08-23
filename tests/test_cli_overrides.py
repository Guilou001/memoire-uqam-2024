"""La ligne de commande surcharge les YAML sans les modifier et résout les chemins depuis workdir/."""

import copy
import os

import yaml

from ml_returns_pred import cli
from ml_returns_pred.paths import CONFIG_DIR, ROOT, WORKDIR


def _merged_config():
    # même fusion que les classes de pipeline : méta + régression + stratégie
    cfg = {}
    for rel in ["meta_config/prediction_pipeline.yaml", "meta_config/regression_pipeline.yaml",
                "strategy_config/ridge_regressor.yaml"]:
        with open(CONFIG_DIR / rel) as f:
            part = yaml.safe_load(f)
        for k, v in part.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def test_root_and_workdir_layout():
    assert (ROOT / "config" / "meta_config" / "prediction_pipeline.yaml").exists()
    assert WORKDIR.parent == ROOT


def test_overrides_set_country_period_signal_and_mode():
    cfg = copy.deepcopy(_merged_config())
    out = cli.apply_overrides(cfg, country="usa", period="2020-2024", signal="top", top=20, n_trials=3,
                              fee=0.001, long_short_mode="corrected", download=False, benchmark=None, tuning=True)
    assert out["read_data"]["relative_prices_data_path"].endswith("us_stocks_2000-01-01_to_2024-06-01.csv")
    assert out["read_data"]["relative_macro_data_path"].endswith("Fred-MD.csv")
    assert out["prediction_pipeline"]["cutoff_date"] == "2019-12-31"
    assert out["preprocess_data"]["max_date"] == "2024-01-01"
    assert out["create_long_short_portfolio"]["fix_threshold"] == 20
    assert out["create_long_short_portfolio"]["use_ranking"] is True
    assert out["prediction_pipeline"]["bayes_search_params"]["n_trials"] == 3
    assert out["compute_strategy_returns"]["transaction_fee"] == 0.001
    assert out["compute_strategy_returns"]["long_short_mode"] == "corrected"
    assert out["folder_cleaner"]["clean_data"] is False
    assert out["download_data"]["download_data"] is False


def test_positive_signal_preset():
    cfg = copy.deepcopy(_merged_config())
    out = cli.apply_overrides(cfg, country="canada", period="2008-2024", signal="positive", top=10, n_trials=None,
                              fee=0.0, long_short_mode="as_published", download=False, benchmark=None, tuning=True)
    lsp = out["create_long_short_portfolio"]
    assert lsp["use_ranking"] is False and lsp["fix_threshold"] == 1
    assert lsp["transform_continuous_to_binary"] is True
    assert out["download_data"]["tickers"] == cli.CANADIAN_TICKERS
    assert len(set(cli.CANADIAN_TICKERS)) == 49 and len(cli.US_TICKERS) == 50


def test_yaml_files_are_not_modified_by_overrides(tmp_path):
    before = (CONFIG_DIR / "meta_config" / "prediction_pipeline.yaml").read_bytes()
    cfg = copy.deepcopy(_merged_config())
    cli.apply_overrides(cfg, country="usa", period="2008-2012", signal="top", top=10, n_trials=1, fee=0.0,
                        long_short_mode="as_published", download=False, benchmark="SP500", tuning=False)
    after = (CONFIG_DIR / "meta_config" / "prediction_pipeline.yaml").read_bytes()
    assert before == after
    assert os.path.basename(cfg["read_data"]["benchmark_prices_relative_path"]).startswith("SP500_")
    assert cfg["prediction_pipeline"]["optimize_hyperparameters"] is False


def test_list_command_runs(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ridge_regressor" in out and "canada" in out
