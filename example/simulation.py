# coding: utf-8
import orca
from activitysim import defaults


import pandas as pd
import numpy as np
import os
orca.add_injectable("store", pd.HDFStore(
        os.path.join("..", "activitysim", "defaults", "test", "test.h5"), "r"))
orca.add_injectable("nonmotskm_matrix", np.ones((1454, 1454)))


# In[4]:

orca.run(["school_location_simulate"])


# In[5]:

orca.run(["workplace_location_simulate"])


# In[7]:

orca.run(["auto_ownership_simulate"])


# In[8]:

orca.run(["cdap_simulate"])


# In[9]:

orca.run(['mandatory_tour_frequency'])

# In[11]:

orca.run(["mandatory_scheduling"])


# In[12]:

orca.run(['non_mandatory_tour_frequency'])

# In[14]:

orca.run(["destination_choice"])


# In[15]:

orca.run(["non_mandatory_scheduling"])


# In[16]:

orca.run(['mode_choice_simulate'])
