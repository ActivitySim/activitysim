{{ objname }}
{{ underline }}

{% if module.startswith('activitysim.core.workflow') %}
.. currentmodule:: {{ module.split('.')[:3] | join('.') }}

.. autoaccessormethod:: {{ (module.split('.')[3:] + [objname]) | join('.') }}
{% else %}
.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessormethod:: {{ (module.split('.')[1:] + [objname]) | join('.') }}
{% endif %}
