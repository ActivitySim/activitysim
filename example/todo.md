This is a list of items to double check before using in practice:

* Make sure the units in things like distance_to_work match the walk thresholds
 in the mandatory tour frequency spec.  The original divided by 100.  This is
  true also of round trip auto to work and round trip auto to school.

* There might be a few variables left off of some of the models.  Look for 
`head` in reading of the spec files as this is meant to eliminate some of the
 rows.  Also can look for `#` to comment out variables in the spec.
 
* Go back to the 3 school location choices, and run the models for the 
appropriate persons.