#Physical constants and Earth parameters for orbital mechanics calculations.
import math

# Earth's physical parameters
EARTH_RADIUS = 6378.137  # km
EARTH_MU = 398600.4418   # km³/s² - Earth's gravitational parameter

# J2 perturbation coefficient (represents Earth's flattening at the poles)
J2 = 0.00108262998905

# Mathematical constants
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Time constants
SECONDS_PER_DAY = 86400      # seconds in one day
MINUTES_PER_DAY = 1440       # minutes in one day
HOURS_PER_DAY = 24           # hours in one day
