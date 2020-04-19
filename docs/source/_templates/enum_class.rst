{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: PandasDtype
   :show-inheritance:
   :exclude-members:

   .. autoattribute:: str_alias
   .. automethod:: from_str_alias
   .. automethod:: from_pandas_api_type




.. autoclass:: {{ objname }}

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

   {% block methods %}
   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
      :toctree: methods

   {% for item in methods %}
   {%- if item not in inherited_members %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}
   {% endif %}

   {%- if '__call__' in members %}
      ~{{ name }}.__call__
   {%- endif %}

   {% endblock %}
