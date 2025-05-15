
# Estimation Mode

ActivitySim includes the ability to re-estimate submodels using choice model estimation
tools. It is possible to output the data needed for estimation and then use more or less
any parameter estimation tool to find the best-fitting parameters for each model, but
ActivitySim has a built-in integration with the [`larch`](https://larch.driftless.xyz)
package, which is an open source Python package for estimating discrete choice models.

## Estimation Workflow, Summarized

The general workflow for estimating models is shown in the following figures and
explained in more detail below.

![estimation workflow](https://activitysim.github.io/activitysim/develop/_images/estimation_tools.jpg)

First, the user converts their household travel survey into ActivitySim-format
households, persons, tours, joint tour participants, and trip tables.  The
households and persons tables must have the same fields as the synthetic population
input tables since the surveyed households and persons will be run through the same
set of submodels as the simulated households and persons.

The ActivitySim estimation example [``scripts\infer.py``](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/example_estimation/scripts/infer.py)
module reads the ActivitySim-format household travel survey files and checks for
inconsistencies in the input tables versus the model design, and calculates
additional fields such as the household joint tour frequency based on the trips
and joint tour participants table.  Survey households and persons observed choices
much match the model design (i.e. a person cannot have more work tours than the model
allows).

ActivitySim is then run in estimation mode to read the ActivitySim-format
travel survey files, and apply the ActivitySim submodels to write estimation data bundles
(EDBs) that contains the model utility specifications, coefficients, chooser data,
and alternatives data for each submodel.

The relevant EDBs are read and transformed into the format required by the model
estimation tool (i.e. larch) and then the coefficients are re-estimated. The
``activitysim.estimation.larch`` library is included for integration with larch
and there is a Jupyter Notebook estimation example for most core submodels.
Certain kinds of changes to the model specification are allowed during the estimation
process, as long as the required data fields are present in the EDB.  For example,
the user can add new expressions that transform existing data, such as converting
a continuous variable into a categorical variable, a polynomial transform, or a
piecewise linear form.  More intensive changes to the model specification, such as
adding data that is not in the EDB, or adding new alternatives, are generally not
possible without re-running the estimation mode to write a new EDB.

Based on the results of the estimation, the user can then update the model
specification and coefficients file(s) for the estimated submodel.

```{eval-rst}
.. toctree::
   :maxdepth: 2

    Running ActivitySim in Estimation Mode <asim-est>
    Using Larch to Re-estimate Models <larch>
    ActivitySim Larch Tool API <larch-api>
```
