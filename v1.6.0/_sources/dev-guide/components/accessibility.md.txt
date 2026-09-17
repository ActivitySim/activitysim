(component-accessibility)=
# Accessibility

```{eval-rst}
.. currentmodule:: activitysim.abm.models.accessibility
```

The accessibilities model is an aggregate model that calculates multiple origin-based accessibility
measures by origin zone to all destination zones.

The accessibility measure first multiplies an employment variable by a mode-specific decay function.  The
product reflects the difficulty of accessing the activities the farther (in terms of round-trip travel time)
the jobs are from the location in question. The products to each destination zone are next summed over
each origin zone, and the logarithm of the product mutes large differences.  The decay function on
the walk accessibility measure is steeper than automobile or transit.  The minimum accessibility is zero.

Level-of-service variables from three time periods are used, specifically the AM peak period (6 am to 10 am), the
midday period (10 am to 3 pm), and the PM peak period (3 pm to 7 pm).

*Inputs*

* Highway skims for the three periods.  Each skim is expected to include a table named "TOLLTIMEDA", which is the drive alone in-vehicle travel time for automobiles willing to pay a "value" (time-savings) toll.
* Transit skims for the three periods.  Each skim is expected to include the following tables: (i) "IVT", in-vehicle time; (ii) "IWAIT", initial wait time; (iii) "XWAIT", transfer wait time; (iv) "WACC", walk access time; (v) "WAUX", auxiliary walk time; and, (vi) "WEGR", walk egress time.
* Zonal data with the following fields: (i) "TOTEMP", total employment; (ii) "RETEMPN", retail trade employment per the NAICS classification.

*Outputs*

* taz, travel analysis zone number
* autoPeakRetail, the accessibility by automobile during peak conditions to retail employment for this TAZ
* autoPeakTotal, the accessibility by automobile during peak conditions to all employment
* autoOffPeakRetail, the accessibility by automobile during off-peak conditions to retail employment
* autoOffPeakTotal, the accessibility by automobile during off-peak conditions to all employment
* transitPeakRetail, the accessibility by transit during peak conditions to retail employment
* transitPeakTotal, the accessibility by transit during peak conditions to all employment
* transitOffPeakRetail, the accessiblity by transit during off-peak conditions to retail employment
* transitOffPeakTotal, the accessiblity by transit during off-peak conditions to all employment
* nonMotorizedRetail, the accessibility by walking during all time periods to retail employment
* nonMotorizedTotal, the accessibility by walking during all time periods to all employment

The main interface to the accessibility model is the
[compute_accessibility](activitysim.abm.models.accessibility.compute_accessibility)
function.  This function is registered as an Inject step in the example Pipeline.

## Structure

- *Configuration File*: `accessibility.yaml`
- *Core Table*: `skims`
- *Result Table*: `accessibility`


## Configuration

```{eval-rst}
.. autopydantic_model:: AccessibilitySettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

### Examples

- [Prototype MTC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_mtc/configs/accessibility.yaml)
- [Prototype ARC](https://github.com/ActivitySim/activitysim/blob/main/activitysim/examples/prototype_arc/configs/accessibility.yaml)


## Implementation

```{eval-rst}
.. autofunction:: compute_accessibility
```
