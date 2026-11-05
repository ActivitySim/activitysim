(component-telework_duration)=
# Telework Duration

```{eval-rst}
.. currentmodule:: activitysim.abm.models.telework_duration
```

The telework duration model predicts the number of hours of in-home work on the
simulation day for people selected by
{doc}`telework_arrangement`. This includes both hybrid
workers and workers who work only at home on the simulation day. It predicts a duration category,
not a continuous number of hours.

The component supports two choice methods:

- **MNL**: choose a duration category from a utility specification and
  coefficients.
- **PROBABILISTIC**: draw a category directly from a configured probability
  distribution, optionally joined to choosers by one or more columns.

The SANDAG ABM3 example uses MNL with estimates from the 2022 SANDAG Household
Travel Survey. In that model sequence, `telework_duration` follows
`telework_arrangement` and precedes mandatory tour and trip models.

The main interface is the
{py:func}`activitysim.abm.models.telework_duration.telework_duration` workflow
step.

## Duration component structure

- *Configuration File*: `telework_duration.yaml`
- *MNL Specification*: `telework_duration.csv`
- *MNL Coefficients*: `telework_duration_coeffs.csv`
- *Alternative-to-hours mapping*: `telework_duration_alts.csv`
- *Probability table*: `telework_duration_probs.csv`
- *Chooser Table*: merged `persons` rows where `has_in_home_work_activity` is true
- *Result Fields*: `persons.telework_duration_category` and
  `persons.telework_duration_hours`

The SANDAG example's `ALTS` file maps the four categories to representative
durations: 3-5 hours to 4, 5-7 hours to 6, 7-9 hours to 8, and 9+ hours to 10.
The open-ended 9+ category is therefore represented as 10 hours in the numeric
output. People who are not selected into this model receive an empty category
and zero hours.

## Estimation data preparation and analysis

Telework duration is estimated conditional on the telework arrangement choice: its sample
consists of workers with qualifying in-home work activity on the survey day.
As in the arrangement model, the observed outcome uses a **three-hour
telecommute-time threshold**, intended to exclude short spillover activity.
The response is grouped into four hour ranges: 3-5, 5-7, 7-9, and 9+ hours.
The implementation treats these as named choice alternatives; it does not
estimate a continuous duration or a distribution within a range.

The survey question measured telecommute time that could include work
performed away from home. In the final data review, 249 reports of more than
seven hours alongside an out-of-home work tour were recategorized as commuting
rather than hybrid working. This resolved part of the earlier concern that
such records could incorrectly enter the in-home-work sample.

For interpretation, distinguish the sample distribution used to estimate
the conditional duration model from the distribution in all workers: the
latter also depends on the arrangement model's prediction of who has in-home
work. The final weighted SANDAG survey duration shares were 17%, 14%, 59%, and 10%
across the four categories, while the uncalibrated example MNL run
produced 23%, 20%, 48%, and 9%. Both distributions are conditional on in-home
work; the modeled shares are not calibration targets and do not measure the
population-wide telework rate.

## Duration estimation and validation considerations

As with arrangement, the duration estimation sample is based on worker-day
reports of at least three hours of telecommute time. The threshold, consistent
with the earlier Phase 9 methodology, was intended to omit short spillover
episodes such as checking email at home. Duration is therefore conditional on
crossing that threshold; it does not model the full range of work-at-home
episodes, including those shorter than three hours.

The survey duration question covered work performed from any location, not
only at home. In the final data review, 249 workers reporting over seven hours
of telecommuting while also making out-of-home work tours were recategorized as
commuting rather than hybrid working. These records are likely workers 
who may telework from alternative locations such as coffee
shops, coworking spaces, field sites, or client locations. They represented
about 4% of unweighted workers and 6% of weighted workers in the review.

The MNL estimates indicate that duration varies by person type, income, and
whether the worker exclusively works from home. The workplace
mode-choice logsum is positive in the shorter-duration alternatives in the
final specification. Early estimation discussion described the duration
logsum as positive but insignificant for shorter-duration alternatives. The
arrangement model later added the intrazonal indicator, which improved the
logsum's significance/scale.

The initial empirical probability approach was intended to support an overall
duration distribution or distributions segmented by characteristics such as
person type. The SANDAG example contains a person-type probability table, but
the active configuration uses MNL. As a result, the SANDAG run shares reported
above are MNL simulation outputs, not draws from the configured probability
table and not calibration targets.

The duration model was estimated from the post-COVID 2022 SANDAG Household
Travel Survey. Validate duration shares by person type and against the
full-scale application; the meeting notes flag that full-model person-type
patterns can differ from estimation results, potentially because CDAP does not
control by person type and work status. The duration results should also be interpreted
alongside arrangement: duration is conditional on being assigned in-home work,
so its distribution alone does not indicate the prevalence of telework.

See the [June 30, 2026 meeting notes](https://github.com/ActivitySim/meeting-notes/issues/115)
for the initial estimation discussion and the
[August 11, 2026 update](https://github.com/ActivitySim/meeting-notes/issues/128)
for the later survey recoding and full-model validation considerations.

## Probabilistic duration choice

In probabilistic mode, `PROBS_SPEC` supplies one probability per duration
category. If `PROBS_JOIN_COLS` is configured, the table is joined to choosers
using those columns; otherwise the probability file must contain exactly one
row. Probabilities are normalized by row before sampling. The SANDAG example
includes a `ptype`-specific probability table as an alternative to MNL, but its
active `CHOICE_MODEL` is `MNL`. If switching to probabilistic mode, ensure that
the table provides positive probability mass for every eligible chooser group.

## Duration behavioral hypotheses and specification exploration

The estimation work considered person characteristics (age, sex, person type,
usual telecommute frequency, and work-from-home status), workplace and job
attributes (workplace mode-choice logsum, industry, free parking, and
disaggregate accessibility), household characteristics (size, income,
children, household type, home urbanicity, and aggregate accessibility), and
day-level CDAP and children-at-home indicators. The final duration utility
retains a more limited set: person type, work-from-home status, CDAP status,
income, workplace mode-choice logsum, and the home/workplace intrazonal
indicator.

The final specification represents these hypotheses:

- Person type differentiates duration patterns: part-time workers and
  university students have higher utility for the shorter categories shown in
  the coefficient table.
- Mandatory day pattern workers may have shorter in-home work periods than people who work
  at home without an out-of-home work activity. In the final specification,
  the CDAP indicator is applied to the 7-9 and 9+ hour alternatives for
  full- and part-time workers.
- Household income and workplace mode-choice logsum may be associated with
  duration. The final logsum estimate is positive for the 3-5 and 5-7 hour
  alternatives.
- Exclusively working from home may distinguish very long in-home days; in the
  active specification this term appears only for the 9+ hour alternative.

The specification is sparse, with no 7-9-hour alternative-specific constant;
that alternative provides the utility reference. The active SANDAG YAML selects
MNL, although the implementation and config also support empirical
probabilities. The included person-type probability table was an alternative
distribution approach, not the choice method used to produce the reported MNL
example.

## MNL duration specification

The four alternatives are **3-5 hours**, **5-7 hours**, **7-9 hours**, and
**9+ hours**. The utility specification is sparse: an empty cell in the spec
means that term is not included for that alternative. Estimates below are from
the active SANDAG MNL coefficient file; the **7-9 hours** alternative is 
the reference alternative and has no alternative-specific constant.

| Term | 3-5 hours | 5-7 hours | 7-9 hours | 9+ hours |
| --- | ---: | ---: | ---: | ---: |
| Alternative-specific constant | -1.710 | -1.800 | — | -1.530 |
| Part-time worker | 2.000 | 1.850 | — | -1.640 |
| University student | 2.160 | 1.850 | — | — |
| Hybrid worker | — | — | -6.700 | -6.700 |
| Usually works from home | — | — | — | -0.264 |
| Household income $150k or more | -0.194 | -0.194 | — | — |
| Workplace mode-choice logsum | 0.0624 | 0.0624 | — | — |
| Home and workplace in same zone | 0.495 | — | — | — |

The hybrid indicator is true when CDAP is `M` and `ptype < 3`. The workplace
mode-choice logsum is set to zero if the chooser has no valid workplace zone.
The missing-workplace coefficients was used in the estimation and present 
in the coefficient file but are commented out in the MNL specification 
and are not active model terms.

Positive part-time and university-student estimates increase the utility of
the shorter duration alternatives shown. The strongly negative hybrid estimate
for 7+ hours reduces the utility of those long-duration alternatives for
full-time and part-time workers with a mandatory day pattern. Interpret these
as relative utilities within the choice model, not as direct changes in hours.

## Duration configuration reference

```{eval-rst}
.. autopydantic_model:: TeleworkDurationSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

For `CHOICE_MODEL: MNL`, both `SPEC` and `COEFFICIENTS` are required. For
`CHOICE_MODEL: PROBABILISTIC`, `PROBS_SPEC` is required. In either mode, `ALTS`
provides the output category names and their numeric duration mapping.

## Duration software implementation and model sequence

The component selects merged persons rows where
`has_in_home_work_activity` is true. In MNL mode it evaluates the configured
utility specification and maps each chosen alternative to the configured
category name. In probabilistic mode it joins the configured probability table
to choosers, and draws a category. In both modes it maps
the selected category to the numeric representative value in `ALTS`, writes
the category and hours to `persons`, and sets non-choosers to an empty category
and zero hours. In estimation mode, the ActivitySim estimator can replace
simulated choices with observed survey values.

Run duration after arrangement so its chooser filter reflects that day's
in-home-work choice. In the SANDAG ABM3 sequence both steps run after CDAP and
before mandatory tour frequency and downstream tour/trip models. The returned
duration hours are a daily person attribute; they do not, by themselves,
schedule tours or trips.

## Duration regional calibration guidance and data requirements

First validate that regional duration observations are defined on the same
conditional population as the model: workers with qualifying at-home work,
using the same three-hour threshold and final treatment of alternative work
locations. Then compare weighted observed and simulated category shares,
overall and by person type. Also validate the arrangement rate and the
hybrid/teleworking split, since the duration sample is selected by that model.
Compare estimation-sample shares, weighted survey shares, and full-model
shares separately; differences in full-model person-type patterns may arise
from upstream CDAP/tour-making as well as the duration model.

If regional conditional duration shares require calibration while retaining
the estimated relative effects, adjust the category-specific constants in the
MNL specification and iteratively compare weighted predicted shares with
regional targets. The 7-9-hour alternative has no explicit constant and is
the reference alternative; calibrating its share therefore requires
coordinated adjustments to the other alternatives' constants. Since
multinomial probabilities are coupled, do not adjust each share independently
or assume the change in a constant equals a percentage-point change. Re-run
the simulation after each adjustment and verify all category shares and
subgroup patterns. The component has no built-in duration calibration loop.

The current specification requires regional person-day records for eligible
workers with:

- reported hours of work specifically performed at home, the worker's work
  location(s), and any out-of-home work tours on that day;
- worker status, person type, work-from-home status, and hybrid/day-pattern
  status;
- income, home/workplace zones, workplace mode-choice logsum, and the
  regional survey expansion weight;
- category definitions consistent with the four alternatives and enough
  observations in each category to assess reliable shares.

For MNL calibration and validation, these items are needed alongside the
arrangement inputs and an observed conditional duration category. For
probabilistic mode instead, the immediate requirements are duration category
counts/probabilities for each join group (for example, person type), with
positive probability mass for every eligible group. If a region lacks enough
observations to estimate a detailed MNL, the empirical probability method can
be a transparent baseline.

As a practical regional workflow, retain the SANDAG coefficients initially,
check variable definitions, units, missing-value handling, and logsum scale,
and establish weighted observed and uncalibrated predicted shares before
changing constants. Calibrate conditional duration only after arrangement
behavior is credible, then re-run the coupled full model and report the
resulting differences. Re-estimation or new terms are preferable to
calibration when regional relationships differ materially from the
transferred behavioral specification.

## Duration data for future model improvements

The current specification cannot identify whether reported telecommute hours
were worked at home or at another remote location, nor can it represent exact
hours within a category. Future data collection could record work location
and duration for each work period, usual and survey-day schedules separately,
and whether reported remote work overlaps with client, field, or other
out-of-home work tours. More precise start/end times and repeated observations
across days would support continuous or finer-grained duration models and
separation of usual arrangements from day-to-day choices.

Occupation, employer telework policy, schedule flexibility, expected in-office
days, and day-of-week context could help explain duration differences not
represented by the current person-type and hybrid indicators. These items
could support future model refinements; they are not required by the active
SANDAG MNL.

## Duration API

```{eval-rst}
.. autofunction:: telework_duration
```
