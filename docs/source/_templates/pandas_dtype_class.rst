{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :exclude-members:

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
      :nosignatures:

   {% for item in attributes %}
     ~{{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}

   .. autoattribute:: str_alias
   .. automethod:: from_str_alias
   .. automethod:: from_pandas_api_type
