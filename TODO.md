0. implement core.hypotheses module
1. move/rename modules for each extra:
    - strategies
     - pandas.py
   - io
     - pandas.py
   - schema_statistics
     - pandas.py
   - schema_inference
     - pandas.py
   - accessors
     - pandas.py
     - modin.py
     - pyspark.py
   - model
     - pandas
       - model.py
       - model_components.py
       - to_json-schema.py
2. add check extensions.register_check_method kwarg validation to register_check
3. deprecate extensions.py module
4. rename SchemaModel -> DataFrameModel, keep SchemaModel as an alias
