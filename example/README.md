#### A few overarching principles

* A little discussion of "NOT so object oriented" - this is more like a
database - data is in standard tables, NOT in objects...  although the
simulation framework is sort of like adding methods to objects

* The implications of this are that most of the core code is pandas and thus
the quality is controlled by the larger community.  We are thankful that its
quality is very high.  Specifically, there's not so much code in activitysim
"proper" - activitysim will probably be a set of utilities to do esoteric
things, like CDAP or like tour scheduling.

* What it takes to add a new model
    * define a new model
    * define any new data sources necessary
    * add any new assumptions in settings.yaml
    * co-create the spec and any variables that are too complicated (or
    reusable) for the spec
    * run in notebook

* Literally everything is really Python functions that compute something.  
Case study of `num_under16_not_at_school` to show the inter-dependencies.


#### A few questions about "best practices"

* What to put into the defaults directory and what to put in the example 
/ client-specific stuff?  In other words, how much can be shared between
regions?

* Are we ok with file organization as it exists?  Models directory and
tables directory?

* How much variable computation to put in excel versus Python.  For now I've
been putting as much an excel as possible, which allows more "configuration."

* I've made an effort to not have such "magic numbers" in Python code, but
these definitely occur in the csv configuration.  For instance, a variable
for income > 100 thousand has `100` hard-coded in the csv.

* Want to name or number the person types in the spec files (currently 
numbered)?

* We could possilby use YAML to configure the model runs - is the
model code too much code for a modeler to manage or is this just 
right as it makes the model execution transparent to the modeler?
Too much boilerplate or just the right amount?  Depends on whether
this can be shared between regions and pretty much stays constant?

* Big issue: testing for client-specific code?  It's harder because outputs are "data
dependent."  It's easier to take a small dataset and make sure it always runs, which
is what we do now.  At some point we can start actually checking results?
Test against current model system on a model-by-model basis (ouch)?

* Should I go back and put the Q&A I've had with Dave as issues on 
github to save for posterity?  Seems like questions as issues is a pretty
good repo for FAQs.


#### This is a list of items to double check before using in practice:

* Make sure the units in things like distance_to_work match the walk thresholds
 in the mandatory tour frequency spec.  The original divided by 100.  This is
  true also of round trip auto to work and round trip auto to school.

* There might be a few variables left off of some of the models.  Look for
`head` in reading of the spec files as this is meant to eliminate some of the
 rows.  Also can look for `#` to comment out variables in the spec.

* Go back to the 3 school location choices, and run the models for the
appropriate persons.

* Probably needs code review of the variable definitions.  How much of the
variable definitions are shared between regions and how much unique?  Age
categories are shared?  Income categories are unique?

* This will be pretty easy to catch, but need to make sure the  models run
with Matt's changes to simple simulate that are coming.
