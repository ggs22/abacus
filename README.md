# Abacus

Personal finance dashboard for tracking balances, categorizing transactions, and forecasting account trajectories across multiple institutions.

## Features

- **Multi-account ledger** — supports chequing, savings, credit cards, TFSAs, RRSPs, FHSAs, loans, brokerage, and crypto accounts across National Bank, Desjardins, CIBC, WealthSimple, Capital One, and IBKR
- **Transaction management** — import CSV bank statements, auto-assign transaction codes via keyword rules, edit codes inline
- **Cross-account search** — filter transactions by regex on text columns, exact amount, or code across all accounts simultaneously
- **Spending breakdown** — bar chart of income and expenses by category over any period
- **Balance history** — time-series view of individual account balances or custom sums
- **Forecasting** — pluggable strategies (planned transactions, mean, Monte Carlo, parallelized Monte Carlo) with mean ± std bands
- **Automation bots** — Selenium scrapers for National Bank, CIBC, and WealthSimple to auto-download CSV statements

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) for dependency management

## Running

```bash
uv sync
uv run python -m web_ui
```

Serves the dashboard at `http://localhost:8050`.

## Project structure

```
accounting/
  Account.py                  # Account class: CSV ingestion, balance computation, stats
  account_list.py             # AccountsList: aggregation, plotting helpers
  forecast_strategies.py      # ForecastStrategy subclasses + ForecastFactory
  plotting.py                 # Plotly trace builders
  __init__.py                 # AccountFactory: scans accounts/ and instantiates all accounts
  accounts/
    <account_name>/
      config.yaml             # Column mapping, encoding, balance mode, statement day, etc.
      csv_data/               # Raw bank CSV exports (one file per statement)
      assignations.json       # Keyword → code rules for this account
      planned_transactions.json
      pickle_objects/         # Cached parsed data
    common_assignation.json   # Shared keyword → code rules across all accounts
bots/
  cibc_bot.py
  nationalbank_ind_bot.py
  wealthsimple_bot.py
web_ui/
  dashboard.py                # Dash app: layout and callbacks
  transactions_ui.py          # Transactions tab layout and callbacks
  assignations_ui.py          # Assignations panel layout and callbacks
  profiles_ui.py              # Filter profile save/load
  assets/                     # Static CSS and JS served by Dash
  __main__.py                 # Entry point: python -m web_ui
utils/
  period_utils.py             # Period string parser (2026, 2026-01, 2026Q1, ranges)
  datetime_utils.py
  path_utils.py
tests/
  test_accounting_module.py
```

## Account config

Each account directory contains a `config.yaml` that controls how CSVs are parsed:

| Field | Description |
|---|---|
| `columns_names` | CSV column names in order |
| `numerical_columns` | `[column, sign]` pairs; `+1` = credit, `-1` = debit |
| `balance_column` | `[column, sign]` if the CSV includes a running balance; `null` to compute via cumsum |
| `sorting_order` / `sorting_ascendancy` | Row ordering when concatenating statements |
| `rows_selection` | Optional `{filter_column, filter_value}` to drop unwanted rows |
| `initial_balance` | Starting balance for cumsum-mode accounts |
| `statement_day` | Day of month used by `CreditCardPaymentForecastStrategy` |
| `encoding` / `separator` / `date_format` | CSV parsing parameters |
| `status` | `OPEN` or `CLOSED` |

## Forecast strategies

| Strategy | Description |
|---|---|
| `PlannedTransactionsStrategy` | Only scheduled transactions from `planned_transactions.json` |
| `MeanTransactionsStrategy` | Planned transactions + daily mean per unplanned code |
| `MonteCarloStrategy` | Gaussian draws per code per day, N iterations |
| `ParallelMonteCarloStrategy` | Same as Monte Carlo, distributed across CPU cores |
| `NoTransactionsStrategy` | Flat balance, no new transactions |
| `FixedLoanPaymentForecastStrategy` | Monthly interest + capital repayment across two accounts |
| `CreditCardPaymentForecastStrategy` | Monthly statement payment across chequing and CC accounts |

## Transaction filters (Transactions tab)

| Filter | Behaviour |
|---|---|
| Regex | Matched against all `object`-dtype columns (e.g. `description`) |
| Amount | Exact float match within tolerance `1e-2`; prefix with `abs()` to ignore sign |
| Code | Multi-select; searches across all accounts in the current filter scope |

Any active filter triggers cross-account search mode — results are grouped by account in the left panel.
