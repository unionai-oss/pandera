.. empty

{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

   {% block classes %}

     {% for item in classes %}
        .. autoclass:: {{ item }}
           :members:
           :member-order: bysource
           :show-inheritance:
           :exclude-members:
     {%- endfor %}

   {% endblock %}

   {% block functions %}

     {% for item in functions %}
        .. autofunction:: {{ item }}
     {%- endfor %}

   {% endblock %}
