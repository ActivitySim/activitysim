{{ objname }}
{{ underline }}

{% if module.startswith('activitysim.core.workflow') %}
.. currentmodule:: {{ module.split('.')[:3] | join('.') }}

.. autoaccessorcallable:: {{ (module.split('.')[3:] + [objname]) | join('.') }}.__call__
{% else %}
.. currentmodule:: {{ module.split('.')[0] }}

.. autoaccessorcallable:: {{ (module.split('.')[1:] + [objname]) | join('.') }}.__call__
{% endif %}
