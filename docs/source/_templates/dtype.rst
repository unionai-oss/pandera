{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

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

   {# Ignore the DateTime alias to avoid `WARNING: document isn't included in any toctree`#}
   {% if objname != "DateTime" %}
     {% for item in methods %}
       ~{{ name }}.{{ item }}
     {%- endfor %}

     {%- if members and '__call__' in members %}
       ~{{ name }}.__call__
     {%- endif %}
   {%- endif %}

   {%- endif %}
   {% endblock %}
