{{ objname }}
{{ underline }}

..
   module is "{{ module }}"
   objname is "{{ objname }}"
   objtype is "{{ objtype }}"

{% if module.startswith('activitysim.core.workflow') %}
.. currentmodule:: {{ module.split('.')[:3] }}

.. autoaccessormethod:: {{ (module.split('.')[3:] + [objname]) | join('.') }}
{% else %}
.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessormethod:: {{ (module.split('.')[1:] + [objname]) | join('.') }}
{% endif %}
