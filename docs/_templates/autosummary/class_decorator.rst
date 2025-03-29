{{ objname | escape | underline}}

{% if module.startswith('activitysim.core.workflow') %}
.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :show-inheritance:
{% else %}
.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{% endif %}
