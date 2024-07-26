# Sharrow

Significant performance improvements can be achieved by using the `sharrow` library,
although doing so requires certain limitations on model design, particularly
relating to what expressions are allowed in utility specifications.  Generally
any model specification can be accommodated in `sharrow` by re-writing problematic
expressions in a more `sharrow`-friendly way, or by moving them to a pre-processor.

If your model is designed to be compatible with `sharrow`, you can activate these
enhancements by setting the `sharrow` configuration setting in the model main
settings (typically `settings.yaml`):

```yaml
sharrow: True
recode_pipeline_columns: True
```

The `recode_pipeline_columns` setting is not absolutely required if the input
data is already in the correct format (i.e. with TAZ ID's starting at zero), but
in practice it is almost always necessary to set this to `True` as most zone systems
do not start at zero.

```{tip}
Sharrow is a powerful tool for improving model performance, but it is not compatible
with all ActivitySim features.  If you need to use tracing or estimation mode,
consider turning off `sharrow` by setting `sharrow: False` in the model settings.
```

Making these settings in the top level configuration will enable `sharrow`
globally for a model, although individual model components can be configured to
not use `sharrow` for various reasons. For most model users, the only sharrow-related
setting that needs to be considered is the `sharrow` setting in the top level
configuration, as this switch is necessary to enable `sharrow` for the model, or
to disable it if the user wants to activate non-sharrow compatible features of
ActivitySim, including tracing and estimation mode.

Instructions on how to work with `sharrow` in ActivitySim are described in
detail in the [Using Sharrow](../../dev-guide/using-sharrow.md)
section of the Developer's Guide. Advanced users are encouraged to read this
section to understand the full capabilities of `sharrow`, how to use it, and how
to troubleshoot issues that may arise when using it.
