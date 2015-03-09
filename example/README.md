This is a list of items to double check before using in practice:

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

* This will be pretty easy to catch, but need to make sure the 
non_mandatory_tour model runs with Matt's changes to simple simulate that are
 coming.



A few overarching principles

* A little discussion of "NOT so object oriented" - this is more like a
database - data is in standard tables, NOT in objects...  although the
simulation framework is sort of like adding methods to objects

* The implications of this are that most of the core code is pandas and thus
the quality is controlled by the larger community.  We are thankful that its
quality is very high.  Specifically, there's not so much code in activitysim
"proper"

* What it takes to add a new model
    * define a new model
    * define any new data sources necessary
    * add any new assumptions in settings.yaml
    * co-create the spec and any variables that are too complicated (or
    reusable) for the spec
    * run in notebook

* Literally everything is really Python functions that compute something.  
Case study of `num_under16_not_at_school` to show the inter-dependencies.




A few questions about "best practices"

* What to put into the default data sources and variable specs and what to
put in the example / client-specific stuff?

* Want to split up injectables from variables from tables or all one big file
 so it's easier to search?

* How much variable computation to put in excel versus Python

* There were some hard coded limits in the original csv - (area_type < 4 and
distance_to_work < 3) - these are now just left in the csv spec.  Why would
this be different than (income_in_thousands > 50)?  I've made an effort to
not have such "magic numbers" in Python code.  EDIT: I've now added an 
`isurban` variable which reads the area_type from the settings.yaml.  So my 
convention so far is to leave hard-coded numbers out of the Python, 
but putting them in the CSV is ok.  (Elizabeth: MAX_NUM_AUTOS exists now)

* Want to name or number the person types in the spec files?

* We're verging on the need to use YAML to configure the model runs - give 
the non_mandatory_tour model as an example.  Is this too much code for a 
modeler to manage or is this just right as it makes the model execution 
transparent to the modeler?

* Big issue: testing for client-specific code?  It's harder because outputs are "data
dependent."  It's easier to take a small dataset and make sure it always runs.
