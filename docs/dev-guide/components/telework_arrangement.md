(component-telework_arrangement)=
# Telework Arrangement

```{eval-rst}
.. currentmodule:: activitysim.abm.models.telework_arrangement
```

The telework arrangement model predicts whether a worker has any in-home work
activity **on the simulated day**. Its binary alternatives are **has in-home work
activity** and **no in-home work activity**. The first alternative includes both
hybrid workers, who also work outside the home, and workers who work only at
home; it does not mean that the person works at home for the entire day.

The SANDAG ABM3 implementation estimates a binary multinomial logit (MNL) model
using the 2022 SANDAG Household Travel Survey. In that model sequence,
`telework_arrangement` follows `telecommute_frequency` and `cdap_simulate`, and
precedes mandatory tour frequency. It uses person, work, and day-pattern
attributes prepared by earlier steps.

The main interface is the
{py:func}`activitysim.abm.models.telework_arrangement.telework_arrangement`
workflow step.

## Arrangement component structure

- *Configuration File*: `telework_arrangement.yaml`
- *Model Specification*: `telework_arrangement.csv`
- *Coefficients*: `telework_arrangement_coeffs.csv`
- *Chooser Table*: merged `persons` rows where `is_worker` is true
- *Result Field*: `persons.has_in_home_work_activity` (boolean)

## Estimation data preparation and analysis

The SANDAG estimation used person-day observations from the post-COVID 2022
Household Travel Survey. The observed outcome is derived from reported
telecommute time on the travel day, combined with work and tour information.
The modeling threshold is **at least three hours** of reported telecommute
time to be considered as having in-home work activity. This Phase 9-consistent 
threshold is intended to exclude short episodes such as answering email at home 
after an otherwise out-of-home workday.

The underlying analysis distinguishes four worker-day classifications:

| Classification | In-home work activity? | Interpretation |
| --- | --- | --- |
| Commuting | No | Out-of-home work tour(s), without qualifying in-home work |
| Hybrid working | Yes | Qualifying in-home work and out-of-home work tour(s) |
| Teleworking | Yes | Qualifying in-home work without an out-of-home work tour |
| Not working | No | No work activity on the modeled day |

The model collapses these four observed classes into the two alternatives in
the binary choice specification: hybrid working and teleworking map to **has
in-home work activity**; commuting and not working map to **no in-home work
activity**. Thus, the binary choice alone does not distinguish a hybrid day
from a day spent working entirely at home.

In the final survey review, 249 records with over seven hours of reported
telecommute time and an out-of-home work tour were recategorized as commuting.
The review concluded the reported telecommute time could represent work from
an alternative location, not work at home. In the final weighted survey
classification, the four shares were approximately 39% commuting, 2% hybrid,
30% not working, and 29% teleworking. These are weighted survey shares which will be used
as model calibration targets, they do not necessarily reflect
the underlying unweighted estimation sample used for estimation.

The estimation data also contain inconsistencies that should be preserved and
documented when replicating the model. Roughly 10% of workers (unweighted) reported never
telecommuting in a typical week while reporting at least three hours on the
survey day. This may be due to misreporting or misunderstanding of the survey question. 
The weekly frequency was left unchanged for consistency with other
SANDAG estimation work. Separately, early data checks found that around
11-12% of non-home-based workers (unweighted) had no workplace location. A missing-workplace location
term was included in the estimation as a dummy variable to isolate the effect of missing workplace location, 
and the corresponding term is commented out in the final
arrangement specification for production runs. The workplace logsum is set to zero when no valid
workplace zone is available, which happens in the model for workers who exclusively work from home, 
i.e., work_from_home == True.

Before estimating or validating in another region, derive the binary outcome
using the same worker-day unit, worker definition, three-hour rule, and
distinction between home and other work locations. Keep survey weights and
sample filters with the prepared observations so that estimation results,
weighted descriptive shares, and full-model shares can be compared on
consistent bases.

## Estimation and validation considerations

The dependent variable was constructed from reported worker-day telecommute
time using a **3-hour threshold** for in-home work activity. The threshold,
consistent with the earlier Phase 9 methodology, was intended to exclude short
spillover episodes such as answering emails after returning home. The
arrangement choice is binary, but the result distinguishes
four worker-day outcomes: commuting, hybrid working, teleworking, and not
working. Hybrid working and teleworking are the two outcomes with in-home work.
Because the source question asked about telecommute time rather than strictly
work performed at home, the survey-based in-home-work outcome requires careful
interpretation.

Several survey issues affected estimation and should be considered when
interpreting the coefficients and shares:

- **Missing workplace locations:** Early estimation found that roughly 11-12%
  of non-home-based workers had no reported workplace location. A
  missing-workplace indicator was considered to avoid treating a missing
  workplace as a substantive attribute. Although a corresponding coefficient
  remains in the SANDAG coefficient file, its term is commented out in the
  active arrangement specification and is not used by the model.
- **Reported frequency versus day-level time:** About 10% of workers reported
  that they never telecommute in a typical week but also reported at least
  three hours of telecommute time on the survey day. The reported weekly
  frequency was retained rather than recoded, to remain consistent with other
  SANDAG estimation work. As a result, frequency-by-CDAP interactions should
  not be interpreted as perfectly consistent observed measures for every
  survey record.
- **Work performed away from home:** The survey's telecommute-time question
  included work from any location, including coffee shops, coworking spaces,
  field sites, or client locations. A person could therefore report telecommute
  time and also make an out-of-home work tour without having worked at home.
  The final data review reclassified 249 workers with over seven hours of
  reported telecommuting and an out-of-home work tour as commuting rather than
  hybrid workers. Meeting notes first raised this as an unresolved source of
  possible overcounting, then documented this recategorization in the later
  update.
- **Intrazonal workplaces and logsums:** The home-equals-workplace-zone
  indicator was added during estimation. Meeting notes report that it improved
  the significance/scale of the workplace mode-choice logsum estimate relative
  to the earlier specification. The final logsum coefficient is negative
  (`-0.0349`), but that direction and estimate are specific to this
  specification and data; the meeting notes do not establish that the final
  estimate is statistically significant.
- **Industry terms:** Business services is the reference category. Industry
  terms were added because they were mostly intuitive in the final model, but
  some findings (including military) were noted as unexplained. These terms
  should be treated as empirical controls rather than causal effects.

The reported hybrid share changed from roughly 7-8% to about 2% in the final
estimated survey classification after reviewing work-location reporting.
However, the hybrid share rose back to roughly 7% when the updated models 
were applied in the full SANDAG model. Full- and
part-time worker shares matched reasonably, while student shares did not in that 
the commuting and hybrid shares were significantly lower than observed;
this suggests that student workers' mandatory day pattern may not be well captured in CDAP.
These differences warrant checking the upstream model and full-model outputs 
in production together rather than assuming that
estimated survey shares will carry through unchanged. Person-type calibration
may be considered if needed, but is beyond the scope of this document.

The estimates use the post-COVID 2022 SANDAG Household Travel Survey. The
observed prevalence and relationships may reflect that survey period and
should not automatically be assumed to represent other years or regions.

For discussion of these data and estimation issues, see the
[June 30, 2026 meeting notes](https://github.com/ActivitySim/meeting-notes/issues/115)
and the [August 11, 2026 update](https://github.com/ActivitySim/meeting-notes/issues/128).

## Behavioral hypotheses and specification exploration

The specification development considered worker and person attributes
(including age, sex, person type, usual telecommute frequency, and usual
work-from-home status), workplace and job characteristics (workplace
mode-choice logsum, industry, free parking, and disaggregate accessibility),
household characteristics (size, income, children, household type, urban
location, and aggregate accessibility), and day-level context (CDAP pattern
and children at home). These are candidate variables considered in the
estimation work, not all terms retained in the final model.

The final SANDAG specification encodes the following behavioral hypotheses:

- Greater usual telecommute frequency and exclusively working from home are
  associated with a higher chance of in-home work, with effects interacted
  with the `H` and `N` CDAP patterns.
- A higher workplace mode-choice logsum represents a more attractive
  out-of-home work commute and was hypothesized to reduce in-home work
  propensity.
- More non-mandatory accessibility near home may make a home-based day more
  useful by enabling errands and other activities; the final `N`-pattern
  accessibility term has the hypothesized positive sign.
- Income, age, person type, preschool children at home, and industry can
  capture differences in work flexibility and household constraints.
- The home-equals-workplace-zone indicator was added during model development.
  In a re-estimation of the telecommute frequency effort, 
  the home-equals-workplace-zone indicator was tested and proven significant. 
  It improved the significance/scale of the workplace-logsum estimate 
  compared with the earlier specification.

Sex, parking, and several household and accessibility candidates do not appear
in the final active arrangement utility. Treat the final specification and its
coefficient file as the authoritative list of included terms; the considered
variables are not evidence that each was retained or independently
significant.

## Binary arrangement specification

The model uses the no-in-home-work alternative (more observed alternative) as the utility reference:

```text
V(no in-home work activity) = 0
V(has in-home work activity) = ASC + sum(beta_k * x_k)
```

Each coefficient below contributes to the utility of the has-in-home-work
alternative. The MNL converts the utility difference to a choice probability;
coefficients are not direct percentage-point changes in probability.

| Term | Estimate | Interpretation |
| --- | ---: | --- |
| Alternative-specific constant | -1.390 | Baseline utility for in-home work, confirming lower observed share relative to no in-home work |
| Part-time worker | -0.631 | `ptype == 2`, compared to full-time workers, part-time workers are less likely to have in-home work |
| University student | -0.737 | `ptype == 3`, compared to full-time and part-time workers, university students are less likely to have in-home work |
| Driving-age student | -999 | `ptype == 6`; compared to full-time workers, driving-age students are effectively ruled out from the in-home-work alternative |
| Preschool child at home | 0.341 | Worker has a preschool child staying at home on the modeled day is more likely to have in-home work |
| Income $100k-$150k | 0.128 | compared to households with income below $100k, households with income between $100k and $150k are more likely to have in-home work |
| Income $150k or more | 0.222 | compared to households with income below $100k and households with income between $100k and $150k, households with income of $150k or more are more likely to have in-home work |
| Age over 35 | -0.206 | `age >= 35`, compared to workers under 35, workers over 35 are less likely to have in-home work |
| Workplace mode-choice logsum | -0.0349 | Set to zero when there is no valid workplace zone, the easier it is for the worker to access the workplace by different modes, the less likely they are to have in-home work |
| One telecommute day/week, CDAP `N` | 0.397 | Interaction of weekly frequency and non-mandatory day pattern, one-day telecommuters with a non-mandatory day pattern are more likely to have in-home work on the simulation day |
| Two, three, or four telecommute days/week, CDAP `N` | 1.810 | Shared coefficient for the two-to-three and four-day terms, compared to one telecommute day/week, more frequent telecommuters are more likely to have in-home work on the simulation day |
| One telecommute day/week, CDAP `H` | 1.430 | Interaction of weekly frequency and home day pattern, one-day telecommuters with a home day pattern are more likely to have in-home work on the simulation day |
| Two or three telecommute days/week, CDAP `H` | 2.270 | Interaction of weekly frequency and home day pattern, more frequent telecommuters are more likely to have in-home work on the simulation day |
| Four telecommute days/week, CDAP `H` | 2.540 | Interaction of weekly frequency and home day pattern, more frequent telecommuters are more likely to have in-home work on the simulation day |
| Worker exclusively works from home, CDAP `N` | 2.590 | Interaction of `work_from_home` and non-mandatory day pattern, home-based workers who also have a non-mandatory day pattern are more likely to have in-home work on the simulation day |
| Worker usually works from home, CDAP `H` | 2.810 | Interaction of `work_from_home` and home day pattern, home-based workers who also have a home day pattern are more likely to have in-home work on the simulation day |
| Retail industry | -0.746 | NAICS 44 or 45, `business` industry is set as reference, compared to the reference, other industry workers are less likely to have in-home work on the simulation day |
| Construction industry | -0.851 | NAICS 23 |
| Other services industry | -0.986 | NAICS 81 |
| Healthcare industry | -0.466 | NAICS 62 |
| Education industry | -0.541 | NAICS 61 |
| Government industry | -0.517 | NAICS 92 |
| Management services industry | -0.594 | NAICS 55 |
| Manufacturing industry | -0.415 | NAICS 31-33 or 3000 |`
| Entertainment industry | -1.090 | NAICS 71 |
| Accommodation industry | -0.798 | NAICS 721 or 722 |
| Military industry | -0.645 | NAICS 9000 |
| CDAP `N` non-mandatory accessibility | 0.0584 | Mean of other-discretionary and shopping accessibility at home, higher accessibility at home increases the likelihood of in-home work |
| Home and workplace in same zone | 0.586 | Intrazonal indicator, workers with home and workplace in the same zone are more likely to have in-home work on the simulation day |

Business services is the reference industry in the final SANDAG specification, since it 
was the most common industry among the surveyed workers. 
As a result, all other industry coefficients are interpreted relative to business services.
Agriculture industry had the fewest workers among the surveyed sample, 
which resulted in an insignificant coefficient in the estimation and is 
therefore omitted from the final specification.


## Arrangement configuration reference

```{eval-rst}
.. autopydantic_model:: TeleworkArrangementSettings
    :inherited-members: BaseModel, PydanticReadable
    :show-inheritance:
```

The SANDAG example uses `LOGIT_TYPE: MNL` and sets
`HAS_IN_HOME_WORK_ACTIVITY_ALT: 0`. That alternative index is converted to a
boolean `has_in_home_work_activity` result. During estimation, observed survey
choices can override simulated choices through ActivitySim's estimator.

## Software implementation and model sequence

The component filters the merged persons table to rows where `is_worker` is
true, annotates any configured preprocessors, loads the model specification
and coefficients, and calls ActivitySim's simple MNL simulation. It converts
the configured in-home-work alternative index to a boolean and writes
`has_in_home_work_activity` back to `persons`. In estimation mode, survey
choices can override simulated choices through the estimator interface.
Persons outside the chooser filter receive `False`.

In the SANDAG ABM3 sequence, `telework_arrangement` runs after `cdap_simulate` and before
`telework_duration`, `mandatory_tour_frequency`, and tour/trip models. Duration
uses the arrangement result as its chooser filter, so sequence order and the
`has_in_home_work_activity` field must be preserved when integrating the
components.

## Regional calibration guidance and data requirements

Treat the SANDAG estimates as a starting specification, not universal regional
parameters. First validate the regional model against a representative
person-day survey using the same worker definition and at-home-work outcome
definition, including the three-hour threshold. Compare both the overall
in-home-work share and the commuting/hybrid/teleworking/not-working split share. 
Review results by person type and day pattern; meeting notes report
that full- and part-time shares transferred more closely than student shares,
and that the hybrid share in the full SANDAG application differed from the
re-estimated survey share.

If the regional overall in-home-work rate needs adjustment while preserving
the estimated relative effects, calibrate the alternative-specific constant
and re-run the model iteratively against the region's weighted target. 
Keep the other coefficients fixed unless regional evidence supports re-estimation. 
The current specification has interactions that may support further subgroup
investigation, but does not provide a separate constant for every segment;
do not claim subgroup calibration without validating the resulting choice
probabilities and model behavior.

At minimum, calibration and validation require regional person-day data with:

- worker status and person type, and the survey/person expansion weight;
- telecommute hours on the travel day, plus enough information to identify
  work performed at home versus at another location;
- usual telecommute frequency and work-from-home status;
- day-level CDAP/activity pattern and out-of-home work-tour status;

Calibration of this component does not by itself ensure that the full model 
reproduces observed shares, because CDAP and downstream tour-making 
also affect applied results.

## Data for future model improvements

The existing specification cannot distinguish genuine at-home work from
remote work performed at another site when the survey records both only as
telecommuting. Future surveys could improve outcome classification by recording
the work location for each reported work period (home, regular workplace,
client/field site, coworking/other), hours at each location, and whether
out-of-home work tours were part of the same workday. Capturing the usual
weekly pattern and the specific survey-day arrangement as separate questions
would also help resolve conflicts between reported weekly frequency and
day-level behavior.

Additional job/worker information—such as occupation, employer telework
policy, schedule flexibility, and expected versus realized in-office days—
could support more behaviorally specific hypotheses than broad industry
categories. Repeated observations across days and explicit survey-day context
could help separate stable work arrangements from day-to-day variation.
These are candidate future data items, not inputs required by the current
model specification.

## Arrangement API

```{eval-rst}
.. autofunction:: telework_arrangement
```
