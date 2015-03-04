"""
This model predicts the departure time and duration of each activity for
mandatory tours
"""


@sim.table()
def tdd_alts():
    # right now this file just contains the start and end hour
    f = os.path.join("configs",
                     "tour_departure_and_duration_alternatives.csv")
    return pd.read_csv(f)


# used to have duration in the actual alternative csv file,
# but this is probably better as a computed column
@sim.column("tdd_alts")
def duration(tdd_alts):
    return tdd_alts.end - tdd_alts.start


@sim.table()
def tdd_mandatory_spec():
    f = os.path.join('configs', 'tour_departure_and_duration_mandatory.csv')
    return asim.read_model_spec(f, stack=False)


@sim.table()
def tdd_non_mandatory_spec():
    f = os.path.join('configs', 'tour_departure_and_duration_nonmandatory.csv')
    return asim.read_model_spec(f, stack=False)


@sim.model()
def tour_departure_and_duration_mandatory(mandatory_tours,
                                          persons,
                                          households,
                                          land_use,
                                          tdd_alts,
                                          tdd_mandatory_spec):

    choosers = sim.merge_tables(mandatory_tours.name, tables=[mandatory_tours,
                                                              persons,
                                                              households,
                                                              land_use])

    print "Running %d mandatory tour scheduling choices" % len(choosers)

    # assert there's only a first or second mandatory tour - that's a basic
    # assumption of this model formulation right now
    assert choosers.tour_num.isin([1, 2]).value_counts()[True] == len(choosers)

    first_tours = choosers[choosers.tour_num == 1]
    second_tours = choosers[choosers.tour_num == 2]

    spec = tdd_mandatory_spec.to_frame().head(27)
    alts = tdd_alts.to_frame()

    print choosers.mandatory_tour_frequency.value_counts()
    print spec.tail()

    # FIXME the "windowing" variables are not currently implemented

    # FIXME this version hard-codes the 2 passes of the mandatory models -
    # FIXME the non-mandatory version below does not - might should use that
    # FIXME kind of structure here instead?

    # this is a bit odd to python - we can't run through in for loops for
    # performance reasons - we first have to do a pass for the first tours and
    # then for the second tours - this is mainly because the second tours are
    # dependent on the first tours' scheduling

    print "Running %d mandatory first tour choices" % len(first_tours)

    alts["end_of_previous_tour"] = -1

    # FIXME - a note to remember that this also needs the mode choice logsum
    alts["mode_choice_logsum"] = 0

    first_choices, _ = \
        asim.simple_simulate(first_tours, alts, spec, mult_by_alt_col=False)

    print "Running %d mandatory second tour choices" % len(second_tours)

    # FIXME need to set end_of_previous_tour to the ends computed above
    second_choices, _ = \
        asim.simple_simulate(second_tours, alts, spec, mult_by_alt_col=False)

    choices = pd.concat([first_choices, second_choices])

    # as with non-mandatory tour generation, this stores the INDEX of
    # the alternative in the tour_departure and_duration_alts dataframe -
    # to actually use it we'll have ot go back and grab the start and end times
    print "Choices:\n", choices.describe()

    sim.add_column("mandatory_tours", "mandatory_tdd", choices)


"""
This model predicts the departure time and duration of each activity for
non-mandatory tours
"""


@sim.model()
def tour_departure_and_duration_non_mandatory(non_mandatory_tours,
                                              persons,
                                              households,
                                              land_use,
                                              tdd_alts,
                                              tdd_non_mandatory_spec):

    tours = sim.merge_tables(non_mandatory_tours.name,
                             tables=[non_mandatory_tours,
                                     persons,
                                     households,
                                     land_use])

    print "Running %d non-mandatory tour scheduling choices" % len(tours)

    spec = tdd_non_mandatory_spec.Coefficient.head(4)
    print spec
    alts = tdd_alts.to_frame()

    max_num_trips = tours.groupby('person_id').size().max()

    # because this is Python, we have to vectorize everything by doing the
    # "nth" trip for each person in a for loop (in other words, because each
    # trip is dependent on the time windows left by the previous decision) -
    # hopefully this will work out ok!

    choices = []

    for i in range(max_num_trips):

        nth_tours = tours.groupby('person_id').nth(i)

        print "Running %d non-mandatory #%d tour choices" % \
              (len(nth_tours), i+1)

        nth_tours["end_of_previous_tour"] = -1

        nth_choices, _ = \
            asim.simple_simulate(nth_tours, alts, spec, mult_by_alt_col=False)

        choices.append(nth_choices)

    choices = pd.concat(choices)

    print "Choices:\n", choices.describe()

    sim.add_column("non_mandatory_tours", "non_mandatory_tdd", choices)
