(component-cdap)=
# Coordinated Daily Activity Pattern

```{eval-rst}
.. currentmodule:: activitysim.abm.models.cdap
```

The Coordinated Daily Activity Pattern (CDAP) model predicts the choice of daily activity pattern (DAP)
for each member in the household, simultaneously. The DAP is categorized in to three types as
follows:
* Mandatory: the person engages in travel to at least one out-of-home mandatory activity - work, university, or school. The mandatory pattern may also include non-mandatory activities such as separate home-based tours or intermediate stops on mandatory tours.
* Non-mandatory: the person engages in only maintenance and discretionary tours, which, by definition, do not contain mandatory activities.
* Home: the person does not travel outside the home.

The CDAP model is a sequence of vectorized table operations:

* create a person level table and rank each person in the household for inclusion in the CDAP model.  Priority is given to full time workers (up to two), then to part time workers (up to two workers, of any type), then to children (youngest to oldest, up to three).  Additional members up to five are randomly included for the CDAP calculation.
* solve individual M/N/H utilities for each person
* take as input an interaction coefficients table and then programmatically produce and write out the expression files for households size 1, 2, 3, 4, and 5 models independent of one another
* select households of size 1, join all required person attributes, and then read and solve the automatically generated expressions
* repeat for households size 2, 3, 4, and 5. Each model is independent of one another.

The main interface to the CDAP model is the [run_cdap](activitysim.abm.models.util.cdap.run_cdap)
function.  This function is called by the Inject step `cdap_simulate` which is
registered as an Inject step in the example Pipeline.  There are two cdap class definitions in
ActivitySim.  The first is at [cdap](activitysim.abm.models.cdap) and contains the Inject
wrapper for running it as part of the model pipeline.  The second is
at [cdap](activitysim.abm.models.util.cdap) and contains CDAP model logic.

## Structure

- *Configuration File*: `cdap.yaml`
- *Core Table*: `persons`
- *Result Field*: `cdap_activity`

## Configuration

```{eval-rst}
.. autopydantic_model:: CdapSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/cdap.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/cdap.yaml)

## Implementation

```{eval-rst}
.. autofunction:: cdap_simulate
```
