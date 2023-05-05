(component-config)=
# Component Configuration

Individual components each have their own component-level configuration. These
configuration can include custom component-specific settings, as well as groups
of settings from these boilerplate base classes:

```{eval-rst}
.. currentmodule:: activitysim.core.configuration
.. autosummary::
    :toctree: _generated
    :template: autopydantic-inherits.rst
    :recursive:

    ~base.PydanticReadable
    ~base.PreprocessorSettings
    ~logit.LogitComponentSettings
    ~logit.TemplatedLogitComponentSettings
    ~logit.LogitNestSpec
```
