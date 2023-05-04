"""
Data Model Parameters

Instructions: add any constant values you would like to use to help you
validate inputs or outputs. The current values are examples.
"""
import math

from pydantic import BaseModel


class Parameters(BaseModel):
    """
    Constant values used in the Data Model validation:
        * `income_quartile_breakpoints_in_usd2019`: These are the breakpoints in
        2011 USD for calculating income quartiles. The breakpoints are currently
        USD$30,000, USD$60,000, and USD$120,000
        * `maximum_parking_cost_per_hour`: This is the maximum allowable input parking cost per hour
        * `maximum_internal_zone_number`: The largest zone index for internal (within region) travel
    """

    income_quartile_breakpoints_in_usd2011 = [30000, 60000, 120000, math.inf]
    maximum_parking_cost_per_hour = 50
    maximum_internal_zone_number = 1700
