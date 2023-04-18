{{ objname }}
{{ underline }}

{% if module.startswith('activitysim.core.workflow') %}
.. currentmodule:: {{ module.split('.')[:3] | join('.') }}

.. autoaccessor:: {{ (module.split('.')[3:] + [objname]) | join('.') }}
{% else %}
.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessor:: {{ (module.split('.')[1:] + [objname]) | join('.') }}
{% endif %}
