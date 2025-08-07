import pandas as pd  
import pandera.pandas as pa  
from typing import Annotated  
from pandera.typing import Series
  
print("=" * 60)  
print("PANDERA SCHEMA DESCRIPTION METADATA TEST")  
print("=" * 60)  
  
# Data to validate  
df = pd.DataFrame({  
    "name": ["Alice", "Bob", "Charlie"],  
    "age": [25, 30, 35],  
    "month": [1, 2, 3]  
})  
  
print("\n📊 INPUT DATA:")  
print(df)  
print(f"Data shape: {df.shape}")  
  
# Define schema  
class Schema(pa.DataFrameModel):  
    name: Annotated[str, pa.Field(title="What the helly", description="Name of the person", unique=True)]  
    age: int = pa.Field(ge=0, description="Age of the person")  
    month: Annotated[int, pa.Field(ge=1, le=12, description="Month of the year")]  
  
    @pa.check("name")  
    def custom_check(cls, series: pd.Series) -> pd.Series:  
        return series.apply(lambda x: isinstance(x, str))  
  
print("\n" + "=" * 60)  
print("VALIDATION RESULTS")  
print("=" * 60)  
  
try:  
    validated_df = Schema.validate(df)  
    print("✅ VALIDATION PASSED")  
    print(f"Validated DataFrame shape: {validated_df.shape}")  
except Exception as e:  
    print(f"❌ VALIDATION FAILED: {e}")  
  
print("\n" + "=" * 60)  
print("SCHEMA METADATA ANALYSIS")  
print("=" * 60)  
  
# Get metadata using the get_metadata() method  
metadata = Schema.get_metadata()  
print("\n🔍 SCHEMA METADATA (via get_metadata()):")  
if metadata:  
    for schema_name, schema_info in metadata.items():  
        print(f"\nSchema: '{schema_name}'")  
        print(f"  DataFrame metadata: {schema_info.get('dataframe', 'None')}")  
        print(f"  Columns metadata:")  
        for col_name, col_metadata in schema_info.get('columns', {}).items():  
            print(f"    '{col_name}': {col_metadata}")  
else:  
    print("  No metadata found")  
  
print("\n" + "=" * 60)  
print("COLUMN DESCRIPTION ANALYSIS")  
print("=" * 60)  
  
# Convert to schema and check column descriptions  
schema = Schema.to_schema()  
print(f"\n🏗️  SCHEMA OBJECT: {type(schema).__name__}")  
print(f"Schema name: {schema.name}")  
  
print(f"\n📋 COLUMN DESCRIPTIONS:")  
for column_name, column in schema.columns.items():  
    description = getattr(column, 'description', 'NOT FOUND')  
    title = getattr(column, 'title', 'NOT FOUND')  
    print(f"  Column '{column_name}':")  
    print(f"    Description: {description}")  
    print(f"    Title: {title}")  
    print(f"    Type: {column.dtype}")  
  
print("\n" + "=" * 60)  
print("FIELD INSPECTION (DEBUG)")  
print("=" * 60)  
  
# Debug: Check the actual field objects on the class  
print(f"\n🔧 CLASS FIELD INSPECTION:")  
for attr_name in ['name', 'age', 'month']:  
    if hasattr(Schema, attr_name):  
        field_obj = getattr(Schema, attr_name)  
        print(f"  Field '{attr_name}':")  
        print(f"    Object type: {type(field_obj).__name__}")  
        print(f"    Description: {getattr(field_obj, 'description', 'NOT FOUND')}")  
        print(f"    Title: {getattr(field_obj, 'title', 'NOT FOUND')}")  
        print(f"    Properties: {getattr(field_obj, 'properties', 'NOT FOUND')}")  
    else:  
        print(f"  Field '{attr_name}': NOT FOUND ON CLASS")  
  
print("\n" + "=" * 60)  
print("TEST SUMMARY")  
print("=" * 60)  
  
# Summary of what should be expected  
expected_descriptions = {  
    'name': 'Name of the person',  
    'age': 'Age of the person',   
    'month': 'Month of the year'  
}  
  
print(f"\n📝 EXPECTED vs ACTUAL DESCRIPTIONS:")  
all_correct = True  
for col_name, expected_desc in expected_descriptions.items():  
    if col_name in schema.columns:  
        actual_desc = getattr(schema.columns[col_name], 'description', None)  
        status = "✅" if actual_desc == expected_desc else "❌"  
        print(f"  {status} '{col_name}': Expected='{expected_desc}', Actual='{actual_desc}'")  
        if actual_desc != expected_desc:  
            all_correct = False  
    else:  
        print(f"  ❌ '{col_name}': Column not found in schema")  
        all_correct = False  
  
print(f"\n🎯 OVERALL RESULT: {'✅ ALL DESCRIPTIONS CORRECT' if all_correct else '❌ DESCRIPTION ISSUES DETECTED'}")  
print("=" * 60)









# from typing import Annotated

# import pandas as pd
# import pandera as pa
# from pandera.typing import Series


# # Define a schema using Annotated typing
# class MySchema(pa.DataFrameModel):
#     something: Series[
#         Annotated[
#             str,
#             pa.Field(description="Collected info on something.")
#         ]
#     ]

# # Sample DataFrame
# df = pd.DataFrame({
#     "something": ["info1", "info2", "info3"]
# })

# # Validate using schema.validate
# validated_df = MySchema.validate(df)

# print(validated_df)

# print(MySchema.to_schema().columns["something"].description)




# import pandas as pd
# import pandera.pandas as pa
# from typing import Annotated

# # data to validate
# df = pd.DataFrame({
#     "name": ["Alice", "Bob", "Charlie"],
#     "age": [25, 30, 35],
#     "month": [1, 2, 3]
# })

# # define a schema
# class Schema(pa.DataFrameModel):
#     # name
#     name: Annotated[str, pa.Field(title="What the helly",description="Name of the person", unique=True)]
#     # age
#     age: int = pa.Field(ge=0, description="Age of the person")
#     # month
#     month: Annotated[int, pa.Field(ge=1, le=12, description="Month of the year")]

#     @pa.check("name")
#     def custom_check(cls, series: pd.Series) -> pd.Series:
#         return series.apply(lambda x: isinstance(x, str))
    

# print("Used Schema.validate(df):\n" + str(Schema.validate(df)) + "\n")

# print("Current metadata written into Schema:" + str(Schema.get_metadata()) + "\n")

# schema = Schema.to_schema()
# for column_name, column in schema.columns.items():
#     print(f"Column: {column_name}, Description: {column.description}")

#    column1  column2 column3
# 0        1      1.1       a
# 1        2      1.2       b
# 2        3      1.3       c