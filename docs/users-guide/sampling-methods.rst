.. _sampling_methods_ways_to_run :

Sampling Methods
________________

ActivitySim supports multiple sampling methods for ``activitysim.core.interaction_sample``.
These methods affect how sampled choice sets are constructed for models such as destination
and location choice. They are separate from the global final-choice switch controlled by
``use_explicit_error_terms``.

Available methods are:

* ``monte_carlo``: importance sampling with replacement using probabilities and uniform draws
* ``eet``: importance sampling with replacement using explicit error-term draws
* ``poisson``: independent Poisson inclusion sampling

Default behavior depends on the global EET setting:

* if ``use_explicit_error_terms: False``, the default sampling method is ``monte_carlo``
* if ``use_explicit_error_terms: True``, the default sampling method is ``poisson``

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
  at most once per chooser. This can materially change raw sampled shares in highly peaked cases,
  even though the downstream sampling correction remains well defined.
* ``poisson`` is the current default when global EET is enabled because it avoids repeated
  chooser-by-alternative explicit-error draws during sampling.

For implementation details and runtime considerations, see :doc:`/dev-guide/sampling-methods`.
