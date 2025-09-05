#Physical constants and Earth parameters for orbital mechanics calculations.
import math

# Earth's physical parameters
EARTH_RADIUS = 6378.137  # km
EARTH_NU = 398600.4418   # km³/s² - Earth's gravitational parameter
OMEGA_EARTH = 7.2921159e-5  # rad/s - Earth's rotation rate

# J2 perturbation coefficient (represents Earth's flattening at the poles)
J2 = 0.00108262998905

# Celestial body parameters
AU_KM = 149_597_870.7  # km - Astronomical Unit
MU_SUN = 1.32712440018e11  # km³/s² - Sun gravitational parameter
MU_MOON = 4902.800066  # km³/s² - Moon gravitational parameter
MOON_MEAN_DISTANCE_KM = 384_400.0  # km

# Angular rates (simplified circular models)
SUN_ANGULAR_RATE = 2 * math.pi / (365.25 * 86400.0)  # rad/s - Earth around Sun (heliocentric)
MOON_ANGULAR_RATE = 2 * math.pi / (27.321661 * 86400.0)  # rad/s - Moon around Earth (sidereal)

# Solar radiation pressure at 1 AU (N/m²)
SOLAR_PRESSURE_1AU = 4.56e-6

# Mathematical constants
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi

# Time constants
SECONDS_PER_DAY = 86400      # seconds in one day
MINUTES_PER_DAY = 1440       # minutes in one day
HOURS_PER_DAY = 24           # hours in one day
