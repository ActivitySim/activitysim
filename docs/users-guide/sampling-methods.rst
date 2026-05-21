.. _sampling_methods_ways_to_run :

Sampling Methods
________________

ActivitySim supports multiple sampling methods for ``activitysim.core.interaction_sample``.
These methods affect how sampled choice sets are constructed for models such as destination
and location choice.

Available methods are:

* ``monte_carlo``: importance sampling with replacement using probabilities and uniform draws
* ``eet``: importance sampling with replacement using explicit error-term draws
* ``poisson``: independent Poisson inclusion sampling using probabilities

Default behavior depends on the global simulation method setting:

* if ``use_explicit_error_terms: False``, the default sampling method is ``monte_carlo``
* if ``use_explicit_error_terms: True``, the default sampling method is ``poisson``

However, any method can be used with either simulation method and can be set
globally in the settings:

.. code-block:: yaml

  sample_method: "poisson"

To override the default for a particular model, set the component's compute settings:

.. code-block:: yaml

  compute_settings:
    sample_method: eet

This override applies only to ``interaction_sample``. It does not change how final choices
are simulated elsewhere in ActivitySim.

Practical differences:

* ``monte_carlo`` and ``eet`` both sample with replacement, so duplicated sampled alternatives
  are possible and their aggregate sampled shares track repeated-draw MNL behavior more closely.
* ``poisson`` samples alternatives by inclusion probability, so each sampled alternative appears
  at most once per chooser. This can change raw sampled shares in highly peaked cases, even though
  the downstream sampling correction remains well defined.
* ``monte-carlo`` is the fastest method, followed by ``poisson``, with ``eet`` being the slowest.
  However, for models like location choice, most runtime comes from logsum calculations and the
  total difference between ``monte-carlo`` and ``poisson`` sampling is usually very small.
* ``poisson`` is the current default when running with simulation method explicit error terms
  because it avoids repeated chooser-by-alternative explicit-error draws during sampling while
  still providing improved noise reduction compared to Monte Carlo sampling.

For implementation details and runtime considerations, see :doc:`/dev-guide/sampling-methods`.
