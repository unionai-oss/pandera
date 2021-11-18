from typing import Optional

import pandera as pa


class Transactions(pa.SchemaModel):
    id: pa.typing.Series[int]
    cost: pa.typing.Series[float]

    class Config:
        coerce = True


@pa.check_types
def create_transactions(
    transactions: pa.typing.DataFrame[Transactions],
) -> pa.typing.Columns[Transactions]:
    return transactions


out = create_transactions({"id": [1], "cost": [11.99]})
print(out)
print(type(out))
