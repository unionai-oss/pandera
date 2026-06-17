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

   {% for item in methods %}
   .. automethod:: {{ item }}
   {%- endfor %}

   {%- if members and '__call__' in members %}
   .. automethod:: __call__
   {%- endif %}

   {%- endif %}
   {% endblock %}
