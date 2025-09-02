import numpy as np
import math
from constants import EARTH_NU, DEG_TO_RAD, RAD_TO_DEG
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
        return 2 * math.pi * math.sqrt(self.a ** 3 / EARTH_NU)

    def solve_kepler_equation(self, tolerance=1e-12):
        """ Solves Kepler's equation M = E - e*sin(E) (Iterative) """
        E = self.mean_anomaly #Initial guess
        for iteration in range(50): #Max iterations lower because Newton-Raphson converges much faster
            f = E - self.e * math.sin(E) - self.mean_anomaly
            df_dE = 1 - self.e * math.cos(E)
            if abs(df_dE) < 1e-15:
                E_new = E-f/df_dE
                if abs(E_new - E) < tolerance:
                    return E_new
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
        h = math.sqrt(EARTH_NU * self.a*(1 - self.e ** 2)) #Angular momentum
        vx = -(EARTH_NU / h) * math.sin(nu)
        vy = (EARTH_NU / h) * (self.e + math.cos(nu))
        vz = 0.0
        #6. Transform to inertial frame
        position = self._rotate_to_inertial([x, y, z])
        velocity = self._rotate_to_inertial([vx, vy, vz])

        return np.array(position), np.array(velocity)

    def _rotate_to_inertial(self, vector):
        x, y, z = vector
        # Apply forward rotations from PQW to ECI: R3(Ω) · R1(i) · R3(ω)
        # 1) Rotate by ω around Z
        cos_w = math.cos(self.perigee)
        sin_w = math.sin(self.perigee)
        x1 = cos_w * x - sin_w * y
        y1 = sin_w * x + cos_w * y
        z1 = z
        # 2) Rotate by i around X
        cos_i = math.cos(self.i)
        sin_i = math.sin(self.i)
        x2 = x1
        y2 = cos_i * y1 - sin_i * z1
        z2 = sin_i * y1 + cos_i * z1
        # 3) Rotate by Ω around Z
        cos_O = math.cos(self.raan)
        sin_O = math.sin(self.raan)
        x3 = cos_O * x2 - sin_O * y2
        y3 = sin_O * x2 + cos_O * y2
        z3 = z2

        return [x3, y3, z3]

    @staticmethod
    def from_cartesian(r, v):
        """Converts Cartesian state vectors (r, v) to Orbital Elements."""
        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)
        r_mag = np.linalg.norm(r)
        v_mag = np.linalg.norm(v)
        #Specific orbital energy
        energy = v_mag**2 / 2 - EARTH_NU / r_mag
        #Semi_major axis
        a = -EARTH_NU / (2 * energy)
        #Angular momentum vector
        h_vec = np.cross(r, v)
        h_mag = np.linalg.norm(h_vec)
        #Eccentricity vector
        e_vec = ((v_mag ** 2 - EARTH_NU / r_mag) * r - np.dot(r, v) * v) / EARTH_NU
        e = np.linalg.norm(e_vec)
        #Inclination
        i = math.acos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))
        #Node vector (points to ascending node)
        k_vec = np.array([0, 0, 1])
        n_vec = np.cross(k_vec, h_vec)
        n_mag = np.linalg.norm(n_vec)
        #RAAN (Right Ascension of Ascending Node)
        if n_mag > 1e-10:  # on-equatorial orbit
            raan = math.acos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
            if n_vec[1] < 0:
                raan = 2 * math.pi - raan
        else:
            raan = 0.0  #Equatorial orbit

        #Argument of perigee
        if n_mag > 1e-10 and e > 1e-10:  #Non-equatorial, non-circular
            cos_omega = np.dot(n_vec, e_vec) / (n_mag * e)
            omega = math.acos(np.clip(cos_omega, -1.0, 1.0))
            if e_vec[2] < 0:
                omega = 2 * math.pi - omega
        elif e > 1e-10:  #Equatorial, non-circular
            omega = math.atan2(e_vec[1], e_vec[0])
            if omega < 0:
                 omega += 2 * math.pi
        else:
            omega = 0.0  #Circular orbit

         #True anomaly
        if e > 1e-10:  #Non-circular
            cos_nu = np.dot(e_vec, r) / (e * r_mag)
            nu = math.acos(np.clip(cos_nu, -1.0, 1.0))
            if np.dot(r, v) < 0:
                nu = 2 * math.pi - nu
        else:  #Circular orbit
            if n_mag > 1e-10:  #Non-equatorial circular
                cos_nu = np.dot(n_vec, r) / (n_mag * r_mag)
                nu = math.acos(np.clip(cos_nu, -1.0, 1.0))
                if r[2] < 0:
                    nu = 2 * math.pi - nu
            else:  #Equatorial circular
                nu = math.atan2(r[1], r[0])
                if nu < 0:
                    nu += 2 * math.pi

        #Convert true anomaly to eccentric anomaly
        if e > 1e-10:
            cos_E = (e + math.cos(nu)) / (1 + e * math.cos(nu))
            E = math.acos(np.clip(cos_E, -1.0, 1.0))
            if nu > math.pi:
                E = 2 * math.pi - E
        else:
            E = nu  #For circular orbits, E = nu

        #Convert eccentric anomaly to mean anomaly
        if e > 1e-10:
            M = E - e * math.sin(E)
        else:
            M = E  #For circular orbits, M = E = nu

        #Normalize angles to [0, 2π) range
        i = i % (2 * math.pi)
        raan = raan % (2 * math.pi)
        omega = omega % (2 * math.pi)
        M = M % (2 * math.pi)

        #Convert to degrees and create OrbitalElements object
        return OrbitalElements(
            a=a,
            e=e,
            i=i * RAD_TO_DEG,
            raan=raan * RAD_TO_DEG,
            perigee=omega * RAD_TO_DEG,
            mean_anomaly=M * RAD_TO_DEG
        )