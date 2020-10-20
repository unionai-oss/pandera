.. empty 

{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

   {% block classes %}
   .. rubric:: Pandas object annotations

   .. autosummary::

   {% for item in classes %}
     {% if item != "AnnotationInfo" %}
      {{ item }}
     {% endif %}
   {%- endfor %}

   {% endblock %}

   {% block attributes %}
   .. rubric:: Dtype annotations

   .. autosummary::

    Bool
    DateTime
    Timedelta
    Category
    Float
    Float16
    Float32
    Float64
    Int
    Int8
    Int16
    Int32
    Int64
    UInt8
    UInt16
    UInt32
    UInt64
    INT8
    INT16
    INT32
    INT64
    UINT8
    UINT16
    UINT32
    UINT64
    Object
    String

    {% endblock %}
