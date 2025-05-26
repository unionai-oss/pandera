from collections import namedtuple

# Minimal Column class for demonstration
class Column:
    def __init__(self, dtype):
        self.dtype = dtype

    def __eq__(self, other):
        return isinstance(other, Column) and self.dtype == other.dtype

    def __repr__(self):
        return f"Column(dtype={self.dtype})"

# DataFrameSchema class with __hash__ and __eq__
class DataFrameSchema:
    def __init__(self, columns, checks=None, index=None, metadata=None):
        self.columns = columns
        self.checks = checks or []
        self.index = index
        self.metadata = metadata

    def __hash__(self):
        columns_hash = tuple(
            sorted((col, str(self.columns[col].dtype)) for col in self.columns)
        )
        checks_hash = tuple(str(check) for check in self.checks)
        index_hash = str(self.index) if self.index is not None else ""
        metadata_hash = (
            tuple(sorted(self.metadata.items())) if isinstance(self.metadata, dict) else str(self.metadata)
        )
        return hash((columns_hash, checks_hash, index_hash, metadata_hash))

    def __eq__(self, other):
        if not isinstance(other, DataFrameSchema):
            return False
        return (
            self.columns == other.columns and
            self.checks == other.checks and
            self.index == other.index and
            self.metadata == other.metadata
        )

    def __repr__(self):
        return (
            f"DataFrameSchema(columns={self.columns}, checks={self.checks}, "
            f"index={self.index}, metadata={self.metadata})"
        )

# Example usage and tests
if __name__ == "__main__":
    # Two schemas with the same columns (order doesn't matter)
    schema1 = DataFrameSchema({
        "a": Column(int),
        "b": Column(float)
    }, checks=["check1"], index="idx", metadata={"source": "test"})

    schema2 = DataFrameSchema({
        "b": Column(float),
        "a": Column(int)
    }, checks=["check1"], index="idx", metadata={"source": "test"})

    # A different schema
    schema3 = DataFrameSchema({
        "a": Column(int),
        "b": Column(float)
    }, checks=["check2"], index="idx", metadata={"source": "test"})

    # Test hash and equality
    print("schema1 == schema2:", schema1 == schema2)
    print("hash(schema1) == hash(schema2):", hash(schema1) == hash(schema2))
    print("schema1 == schema3:", schema1 == schema3)
    print("hash(schema1) == hash(schema3):", hash(schema1) == hash(schema3))

    # Test using as dict keys
    d = {schema1: "first schema"}
    print("Retrieve from dict with schema2 as key:", d[schema2])  # Should work

    # Test using in a set
    s = {schema1, schema2, schema3}
    print("Set of schemas:", s) 