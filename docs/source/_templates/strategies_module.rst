.. empty

{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}

   {% block functions %}

     {% for item in functions %}
       .. autofunction:: {{ item }}
     {%- endfor %}

   {% endblock %}
