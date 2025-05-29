(input_checker)=
# Input Checker

The input checker is an optional model that will ensure the inputs are as expected according to the
configuration of the user. This model does not have any connection to downstream models and does not
add anything to the ActivitySim pipeline.  It is intended to be run at the very start of every ActivitySim
run to quickly examine whether the input data aligns with the user's expectations.

The input checker is built on the Pandera python package. Users are responsible for writing the
Pandera code that will perform the checks they would like to be performed. By allowing the user to
write python code, they are provided with maximum flexiblity to check their input data.  (Pydantic
was also explored as a potential technology choice for input checker implementation. Pydantic
development got as far as performing checks and outputting errors, but further development around
reporting warnings and more in-depth testing was dropped in favor of the Pandera approach. The
in-development code has remained as-is in the repository in the event future data model development
favors this approach.)

If any checks fail, ActivitySim will crash and direct you to the output input_checker.log file which
will provide details of the checks that did not pass. The user can also setup checks to output
warnings instead of fatal errors. Warning details will be output to the input_checker.log file for
user review and documentation. Syntax examples for both errors and warnings are demonstrated in the
[`prototype_mtc_extended`] and [`production_semcog`] examples.

Setup steps for new users:
  * Copy the data_model directory in the [`prototype_mtc_extended`] or
    [`production_semcog`] example folder to your setup space. You will need the enums.py and
    input_checks.py scripts. The additional input_checks_pydantic_dev.py script in
    [`prototype_mtc_extended`] is there for future development and can be discarded.
  * Modify the input_checker.py to be consistent with your input data. This can include changing
    variable names and adding or removing checks.  The amount and types of checks to perform are
    completely up to you! Syntax is shown for many different checks in the example.
  * Modify enums.py to be consistent with your implesmentation by changing / adding / removing variable
    definitions.
  * Copy the input_checker.yaml script from [`prototype_mtc_extended`] or
    [`production_semcog`] configs into your configs
    directory. Update the list of data tables you would like to check in the input_checker.yaml
    file. The "validator_class" option should correspond to the name of the corresponding class in
    the input_checker.py file you modified in the above step.
  * Add the input_checker model to the models option in your settings.yaml file to include it in the
    model run. When running activitysim, you will also need to include the name of the data_model
    directory from the first step, e.g. activitysim run -c configs -d data -o outout --data_model
    data_model.

```{note}
If you are running ActivitySim with the input checker module active, you must supply
a --data_model argument that points to where the input_checks.py file exists!
```

## Implementation

```{eval-rst}
.. automodule:: activitysim.abm.models.input_checker
   :members:
```
