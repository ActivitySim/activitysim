import os
import pandas as pd
import numpy as np
import urbansim.sim.simulation as sim
from activitysim import activitysim as asim


"""
This model predicts the frequency of making non-mandatory trips
(alternatives for this model come from a seaparate csv file which is
configured by the user) - these trips include escort, shopping, othmaint,
othdiscr, eatout, and social trips in various combination.
"""


@sim.injectable()
def non_mandatory_tour_frequency_spec(configs_dir):
    f = os.path.join(configs_dir, 'configs', "non_mandatory_tour_frequency.csv")
    # this is a spec in already stacked format
    # it also has multiple segments in different columns in the spec
    return asim.read_model_spec(f, stack=False)


@sim.table()
def non_mandatory_tour_frequency_alts(configs_dir):
    f = os.path.join(configs_dir, "configs",
                     "non_mandatory_tour_frequency_alternatives.csv")
    return pd.read_csv(f)


# this a computed variable of the alts used in the model
@sim.column("non_mandatory_tour_frequency_alts")
def tot_tours(non_mandatory_tour_frequency_alts):
    # this assumes that the alt dataframe is only counts of trip types
    return non_mandatory_tour_frequency_alts.local.sum(axis=1)


@sim.model()
def non_mandatory_tour_frequency(persons_merged,
                                 non_mandatory_tour_frequency_alts,
                                 non_mandatory_tour_frequency_spec):

    choosers = persons_merged.to_frame()

    # filter based on results of CDAP
    choosers = choosers[choosers.cdap_activity.isin(['M', 'N'])]
    print "%d persons run for non-mandatory tour model" % len(choosers)

    choices_list = []
    # segment by person type and pick the right spec for each person type
    for name, segment in choosers.groupby('ptype_cat'):

        print "Running segment '%s' of size %d" % (name, len(segment))

        choices, _ = \
            asim.simple_simulate(segment,
                                 non_mandatory_tour_frequency_alts.to_frame(),
                                 # notice that we pick the column for the
                                 # segment for each segment we run
                                 non_mandatory_tour_frequency_spec[name],
                                 mult_by_alt_col=False)
        choices_list.append(choices)

    choices = pd.concat(choices_list)

    print "Choices:\n", choices.value_counts()
    # this is adding the INDEX of the alternative that is chosen - when
    # we use the results of this choice we will need both these indexes AND
    # the alternatives themselves
    sim.add_column("persons", "non_mandatory_tour_frequency", choices)


"""
We have now generated non-mandatory tours, but they are attributes of the
person table - this function creates a "tours" table which
has one row per tour that has been generated (and the person id it is
associated with)
"""


@sim.table()
def non_mandatory_tours(persons,
                        non_mandatory_tour_frequency_alts):

    # get the actual alternatives for each person - have to go back to the
    # non_mandatory_tour_frequency_alts dataframe to get this - the choice
    # above just stored the index values for the chosen alts
    tours = non_mandatory_tour_frequency_alts.local.\
        loc[persons.non_mandatory_tour_frequency]

    # assign person ids to the index
    tours.index = persons.index[~persons.non_mandatory_tour_frequency.isnull()]

    # reformat with the columns given below
    tours = tours.stack().reset_index()
    tours.columns = ["person_id", "tour_type", "num_tours"]

    # now do a repeat and a take, so if you have two trips of given type you
    # now have two rows, and zero trips yields zero rows
    tours = tours.take(np.repeat(tours.index.values, tours.num_tours.values))

    # make index unique and drop num_tours since we don't need it anymore
    tours = tours.reset_index(drop=True).drop("num_tours", axis=1)

    """
    Pretty basic at this point - trip table looks like this so far
           person_id tour_type
    0          4419    escort
    1          4419    escort
    2          4419  othmaint
    3          4419    eatout
    4          4419    social
    5         10001    escort
    6         10001    escort
    """
    return tours


# broadcast trips onto persons using the person_id
sim.broadcast('persons', 'non_mandatory_tours',
              cast_index=True, onto_on='person_id')
sim.broadcast('persons_merged', 'non_mandatory_tours',
              cast_index=True, onto_on='person_id')
