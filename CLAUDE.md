# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the web app (loads accounts, computes forecasts, serves at `localhost:8050`):**
```bash
python -m frontend
```

**Run unit tests:**
```bash
python -m pytest tests/test_accounting_module.py -v
# Single test:
python -m pytest tests/test_accounting_module.py::TestForecastStrategies::test_monte_carlo_strategy -v
```

## Architecture

### Account loading pipeline

`AccountFactory` (in `backend/__init__.py`) scans `backend/accounts/*/config.yaml` at startup and instantiates one `Account` per directory. Accounts are stored both as module-level globals and collected into an `AccountsList`. The factory also exposes `get_account_from_group()` to filter by institution (NationalBank, WealthSimple, etc.).

Each `Account` (in `backend/Account.py`) is backed by:
- A `config.yaml` that defines column names, separator, encoding, date format, numerical columns with signs (+1/-1), and an optional `balance_column`.
- Monthly CSVs under `csv_data/`. New files are detected by MD5 hash of filename; already-processed files are skipped.
- `assignations.json` for account-specific keyword→code mapping (common mappings live in `backend/accounts/common_assignation.json`).
- `planned_transactions.json` for recurring (monthly/yearly) and one-off future transactions.
- A pickle at `pickle_objects/<AccountName>.pkl` that caches `processed_data_files` and `transaction_data` between runs.

**Balance computation:** If `config.yaml` specifies a `balance_column`, that column is read directly from the CSV and the rows are reordered using a permutation/greedy search to maintain balance continuity (`_compute_transaction_order`). Otherwise the balance is computed by cumsum of signed numerical columns plus `initial_balance`.

### Forecast strategies

All strategies inherit `ForecastStrategy` (in `backend/forecast_strategies.py`) and follow the same pattern:

1. The public `predict()` method sets up outer state (planned transactions, simulation date) and defines a nested `_predict(stats)` closure.
2. `_prediction_wraper()` handles caching: if a `.pkl` for this account + strategy + date exists it loads it; otherwise it calls `_predict()` and serializes the result.
3. After `_prediction_wraper()` returns, `predict()` post-processes: merges planned transactions, sorts by `[ITERATION, date]`, and computes the `balance_column` via grouped cumsum.

| Strategy | Behaviour |
|---|---|
| `PlannedTransactionsStrategy` | Only planned transactions from `planned_transactions.json` |
| `MeanTransactionsStrategy` | Planned transactions + daily mean of unplanned codes |
| `MonteCarloStrategy` | Stochastic; draws Gaussian amounts per code per day |
| `ParallelMonteCarloStrategy` | Same as Monte Carlo but distributes iterations across `ProcessPoolExecutor` workers via `_monte_carlo_worker` |
| `NoTransactionsStrategy` | Flat balance, no new transactions |
| `FixedLoanPaymentForecastStrategy` | Computes monthly interest + capital on a loan, mutates two `Forecast` objects in-place |
| `CreditCardPaymentForecastStrategy` | Models monthly CC statement payments, mutates two `Forecast` objects in-place |

`ForecastFactory.__call__()` accepts a `RequestedForecastList` (list of `(Account, Strategy, kwargs)` tuples) and returns a `Dict[str, Forecast]`.

### Key data flow in `accounts_tests.py`

1. `AccountFactory()` loads all accounts.
2. A `RequestedForecastList` is built pairing each account with a strategy.
3. `ForecastFactory()` runs all forecasts and returns a dict keyed by account name.
4. A second `RequestedForecastList` handles cross-account strategies (`FixedLoanPaymentForecastStrategy`, `CreditCardPaymentForecastStrategy`) that receive previously-computed `Forecast` objects and mutate them.
5. `AccountsList.plot_forecasts()` renders mean ± std bands for each forecast.

### Config schema (config.yaml)

Critical fields:
- `numerical_columns`: list of `[column_name, sign]` pairs; sign `+1` = credit, `-1` = debit.
- `balance_column`: `[column_name, sign]` or `null`. Presence determines whether ordering logic runs.
- `sorting_order` / `sorting_ascendancy`: used when concatenating new CSV data.
- `rows_selection`: optional filter `{filter_column, filter_value}` applied after CSV parse.
- `statement_day`: used by `CreditCardPaymentForecastStrategy` to anchor monthly payment dates.
