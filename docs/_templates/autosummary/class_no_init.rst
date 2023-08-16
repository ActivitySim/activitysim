{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}()

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree: generated/
   {% for item in methods %}
      {% if item != "__init__" %}
         ~{{ [module, name] | join('.') }}.{{ item }}
      {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree: generated/
   {% for item in attributes %}
      ~{{ [module, name] | join('.') }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
