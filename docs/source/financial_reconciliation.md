---
file_format: mystnb
---

(financial-reconciliation)=

# Reconcile financial transactions

Two exports can have valid columns and still disagree about the same
transaction. This tutorial uses Pandera to validate each input, then check
that every transaction appears in both exports with the same currency and
amount. It uses only synthetic data and runs without network access.

Install the pandas extra to run the examples:

```bash
pip install 'pandera[pandas]'
```

## Define the input contract

Assume a bank export and a ledger export have already been normalized to
three columns:

- `transaction_id`: a shared, nonempty reference, unique within each export.
- `currency`: one of the currencies supported by this example.
- `amount_minor`: a signed integer amount in the currency's minor unit.

Here, GBP, EUR and USD all use 100 minor units per major unit: `12500` GBP
means GBP 125.00. Both exports use the same sign convention. In a real
pipeline, establish these rules when reading the source files; different
currencies can have different scales. This example does not match
transactions by approximate amounts, dates, or descriptions.

Use pandas' nullable `Int64` dtype for amounts so an outer join can represent
missing matches without converting integers to floating point. Leave
coercion disabled: silently truncating a fractional amount would hide an
input error.

```{code-cell} python
import pandas as pd
import pandera.pandas as pa

transactions = pa.DataFrameSchema(
    {
        "transaction_id": pa.Column(
            str, pa.Check.str_length(min_value=1), unique=True
        ),
        "currency": pa.Column(str, pa.Check.isin(["GBP", "EUR", "USD"])),
        "amount_minor": pa.Column(pd.Int64Dtype()),
    },
    checks=pa.Check(lambda df: not df.empty, name="nonempty_export"),
    strict=True,
)

bank = pd.DataFrame(
    {
        "transaction_id": ["txn-001", "txn-002", "txn-003"],
        "currency": ["GBP", "GBP", "EUR"],
        "amount_minor": pd.array([12500, -2400, 9000], dtype="Int64"),
    }
)
ledger = bank.copy()

transactions.validate(bank, lazy=True)
transactions.validate(ledger, lazy=True)
```

Columns are required and non-nullable by default. `strict=True` also rejects
unexpected columns instead of silently dropping them. The nonempty check
prevents two accidentally empty exports from passing the reconciliation.

## Check the relationship between exports

First validate the inputs, then use an outer join. An inner join would lose
unmatched transactions, while a one-sided join would miss transactions
present only on the other side. `validate="one_to_one"` also expresses the
join's cardinality in pandas.

The second schema contains dataframe-level checks. It checks presence,
currency and amount separately, so a failure report can show which rule
failed. Filling missing comparisons with `False` and setting
`ignore_na=False` ensures that a missing amount cannot be skipped.

```{code-cell} python
matches = pa.DataFrameSchema(
    checks=[
        pa.Check(
            lambda df: df["_merge"].eq("both"),
            name="present_in_both",
            ignore_na=False,
        ),
        pa.Check(
            lambda df: df["currency_bank"].eq(df["currency_ledger"]),
            name="same_currency",
            ignore_na=False,
        ),
        pa.Check(
            lambda df: df["amount_minor_bank"]
            .eq(df["amount_minor_ledger"])
            .fillna(False),
            name="same_amount",
            ignore_na=False,
        ),
    ]
)


def reconcile(bank_export, ledger_export):
    bank_valid = transactions.validate(bank_export, lazy=True)
    ledger_valid = transactions.validate(ledger_export, lazy=True)
    joined = bank_valid.merge(
        ledger_valid,
        on="transaction_id",
        how="outer",
        suffixes=("_bank", "_ledger"),
        indicator=True,
        validate="one_to_one",
    )
    return matches.validate(joined, lazy=True)


reconciled = reconcile(bank, ledger)
assert len(reconciled) == 3
assert reconciled["amount_minor_bank"].dtype == pd.Int64Dtype()
print(reconciled[["transaction_id", "currency_bank", "amount_minor_bank"]])
```

## Diagnose discrepancies

Create a ledger with an incorrect amount, a different currency and a missing
transaction. The missing transaction still appears in the joined dataframe.

```{code-cell} python
incorrect_ledger = ledger.iloc[:2].copy()
incorrect_ledger.loc[0, "amount_minor"] = 12499
incorrect_ledger.loc[1, "currency"] = "USD"

try:
    reconcile(bank, incorrect_ledger)
except pa.errors.SchemaErrors as exc:
    failures = exc.failure_cases
    assert set(failures["check"]) == {
        "present_in_both", "same_currency", "same_amount"
    }
    print(failures[["check", "index"]].drop_duplicates())
else:
    raise AssertionError("The inconsistent ledger must fail validation")
```

`exc.failure_cases` identifies the failing rules and joined row indexes.
`exc.data` contains the joined dataframe, including the transaction references
and both source values. Keep these together when investigating discrepancies.
The failure report can include several records for one row; it is not a
count of distinct transactions.

Duplicates are input errors, so they are rejected before the join can inflate
row counts:

```{code-cell} python
duplicate_ledger = pd.concat([ledger, ledger.iloc[[0]]], ignore_index=True)

try:
    reconcile(bank, duplicate_ledger)
except pa.errors.SchemaErrors as exc:
    assert "field_uniqueness" in set(exc.failure_cases["check"])
    print("Duplicate reference rejected before reconciliation")
else:
    raise AssertionError("Duplicate references must fail validation")
```

## Use the result as a pipeline boundary

Only continue to calculations, a report, or an AI-assisted explanation after
`reconcile` returns successfully. Let validation errors stop the pipeline or
route them for investigation. Do not repair discrepancies by dropping rows,
filling missing amounts with zero, or coercing an invalid amount to an integer.

A pass establishes agreement under these specific rules. It cannot detect a
transaction missing from both exports, prove that either source is complete,
or establish that a posting is economically correct. Source completeness,
period cutoffs, and any required matching transformations need their own
controls. If an AI model reads a validated result, validation also does not
establish that its explanation is correct.

See {ref}`lazy-validation` for error handling and {ref}`checks` for more
ways to express dataframe relationships.
