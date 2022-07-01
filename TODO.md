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
   - core
     - base
       - model.py
       - model_components.py
       - model_config.py
   - model
     - pandas
       - model.py
       - model_components.py
       - model_config.py
2. add check extensions.register_check_method kwarg validation to register_check
3. add deprecation warning to extensions.py module
4. rename SchemaModel -> DataFrameModel, keep SchemaModel as an alias
