from dis import code_info

import numpy as np
import math
from constants import EARTH_MU, DEG_TO_RAD, RAD_TO_DEG
class OrbitalElements:
    def __init__(self, a, e, i, raan, perigee, mean_anomaly):
        self.a = a         # Semi-major axis (km)
        self.e = e         # Eccentricity
        self.i = i * DEG_TO_RAD       # Inclination (convert to radians)
        self.raan = raan * DEG_TO_RAD  # Right Ascension of the Ascending Node (convert to radians)
        self.perigee = perigee * DEG_TO_RAD  # Perigee (convert to radians)
        self.mean_anomaly = mean_anomaly * DEG_TO_RAD  # Mean anomaly (convert to radians)

    def get_orbital_period(self):
        """ Calculates the orbital period using Kepler's third law """
        return 2 * math.pi * math.sqrt(self.a ** 3 / EARTH_MU)

    def solve_kepler_equation(self, tolerance=1e-10):
        """ Solves Kepler's equation M = E - e*sin(E) (Iterative) """
        E = self.mean_anomaly  #Initial guess
        for _ in range(100):   #Max iterations
            E_new = self.mean_anomaly + self.e * math.sin(E)
            if abs(E_new - E) < tolerance:
                break
            E = E_new
        return E

    def to_cartesian(self):
        """Convert orbital elements to Cartesian coordinates."""
        #1. Solve Kepler's equation
        E = self.solve_kepler_equation()
        #2. Calculate true anomaly
        nu = 2 * math.atan(math.sqrt((1 + self.e)/(1 - self.e)) * math.tan(E / 2))
        #3. Distance from Earth's center
        r = self.a * (1 - self.e * math.cos(E))
        #4. Position in orbital plane
        x = r * math.cos(nu)
        y = r * math.sin(nu)
        z = 0.0 #Because it's seen in 2D
        #5. Velocity in orbital plane
        h = math.sqrt(EARTH_MU * self.a*(1 - self.e ** 2)) #Angular momentum
        vx = -(EARTH_MU / h) * math.sin(nu)
        vy = (EARTH_MU / h) * (self.e + math.cos(nu))
        vz = 0.0
        #6. Transform to inertial frame
        position = self._rotate_to_inertial([x, y, z])
        velocity = self._rotate_to_inertial([vx, vy, vz])

        return np.array(position), np.array(velocity)

    def _rotate_to_inertial(self, vector):
        x, y, z = vector
        #1. Rotation on Z-axis R3(-ω)
        cos_w = math.cos(-self.perigee)
        sin_w = math.sin(-self.perigee)
        x1 = cos_w * x - sin_w * y
        y1 = sin_w * x + cos_w * y
        z1 = z
        #2. Rotation on X-axis R1(-i)
        cos_i = math.cos(-self.i)
        sin_i = math.sin(-self.i)
        x2 = x1
        y2 = cos_i * y - sin_i * z
        z2 = sin_i * y + cos_i * z
        #3. Rotation on Z-axis R3(-Ω)
        cos_raan = math.cos(-self.raan)
        sin_raan = math.sin(-self.raan)
        x3 = cos_raan * x - sin_raan * y
        y3 = sin_raan * x + cos_raan * y
        z3 = z2

        return [x3, y3, z3]

