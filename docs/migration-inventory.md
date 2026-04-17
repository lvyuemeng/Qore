# Migration Inventory

## Purpose

Track how legacy `src/quant_trade` modules map into the new `qore-*` crates during
the rewrite.

## Legacy To Target

| Legacy path | Rewrite action | Target |
| --- | --- | --- |
| `src/quant_trade/transform.py` | retire | none |
| `src/quant_trade/client/eastmoney.py` | reverse engineer and rewrite | `crates/qore-data/src/qore_data/fetcher/eastmoney.py` |
| `src/quant_trade/provider/akshare.py` | do not port directly | reference only via `.ai/refs/akshare/` |
| `src/quant_trade/provider/baostock.py` | optional later adapter | `crates/qore-data/` |
| `src/quant_trade/config/arctic.py` | retire | `crates/qore-data/src/qore_data/store/duckdb.py` |
| `src/quant_trade/feature/process.py` | port formulas only | `crates/qore-factor/src/qore_factor/` |
| `src/quant_trade/feature/store.py` | retire and split | `qore-core` + `qore-data` |
| `src/quant_trade/model/process.py` | port concepts only | `crates/qore-intelligence/src/qore_intelligence/model/` |
| `src/quant_trade/model/lgb.py` | port concepts only | `crates/qore-intelligence/src/qore_intelligence/model/` |
| `src/quant_trade/model/store.py` | rewrite | `crates/qore-intelligence/src/qore_intelligence/model/pipeline.py` |
| `scripts/smoke_train.py` | retire | future CLI/examples |

## Immediate Build Order

1. `qore-core`
2. `qore-data`
3. `qore-factor`
4. `qore-intelligence`
5. `qore-runner`
6. `qore-backtest`
