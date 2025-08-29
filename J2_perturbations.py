import math
import numpy as np
from constants import EARTH_RADIUS, EARTH_NU, J2, RAD_TO_DEG

class J2Perturbations:
    def __init__(self, orbital_elements):
        self.orbit = orbital_elements
    def calculate_secular_rates(self):
        a = self.orbit.a  # Semi-major axis (km)
        e = self.orbit.e  # Eccentricity
        i = self.orbit.i  # Inclination (radians)
        #1. Calculate Keplerian mean motion
        n_kepler = math.sqrt(EARTH_NU/(a**3))
        #2. Calculate orbital parameter
        p0 = a*(1-e**2)
        #!Safety check for degenerate orbits
        if p0 <= 0:
            raise ValueError(f"Invalid orbital parameter p0 = {p0:.6f} km")
        #3. Calculate J2 base coefficient
        j2_base_coef = 3/2 * J2 * (EARTH_RADIUS/ p0)**2
        #4. Calculate modified mean motion n_bar
        if e < 0.99: #Normal orbit
            sqrt_1_minus_e2 = math.sqrt(1 - e**2)
        else: #Highly elliptical orbits
            sqrt_1_minus_e2 = 0.1
        sin_i_squared = math.sin(i)**2
        inclination_factor = 1-(3/2) * sin_i_squared
        n_bar = (1 + j2_base_coef * sqrt_1_minus_e2 * inclination_factor)*n_kepler # Modified mean motion n̄
        #5.Calculate RAAN precession rate
        raan_dot = (-j2_base_coef * math.cos(i))*n_bar
        #6.Calculate argument of perigee precession rate
        perigee_factor = 2-(5/2)*sin_i_squared
        perigee_dot = (j2_base_coef *  perigee_factor) * n_bar
        #7.Mean anomaly rate is modified mean motion
        delta_M_j2 = j2_base_coef * n_kepler * sqrt_1_minus_e2 * inclination_factor
        mean_anomaly_dot = n_kepler + delta_M_j2
        return {
            # Primary results
            'raan_dot': raan_dot,         # Ω̇_J2 (rad/s)
            'perigee_dot': perigee_dot,   # ω̇_J2 (rad/s)
            'mean_anomaly_dot': mean_anomaly_dot,   # n̄ (rad/s)
            #Elements without secular J2 changes, 0 because no drag ṅ
            'semi_major_axis_dot': 0.0,           # ȧ = 0 for J2
            'eccentricity_dot': 0.0,              # ė = 0 for J2
            'inclination_dot': 0.0,               # i̇ = 0 for J2
            #Diagnostic info
            'n_kepler': n_kepler,  # Original mean motion
            'n_bar': n_bar,  # Modified mean motion
            'delta_M_j2': delta_M_j2, # J2 correction to mean motion - FIXED!
            'p0': p0,  # Orbital parameter
            'j2_base_coef': j2_base_coef,  # Base J2 coefficient
        }

    def calculate_J2_acceleration(self, position):
        x,y,z = position
        r = np.linalg.norm(position)
        if r < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        #J2 acceleration coefficient
        nu_re2_j2 = EARTH_NU * (EARTH_RADIUS ** 2) * J2
        base_coef = (3/2) * nu_re2_j2 / (r ** 5)
        #Z-dependant factor
        z_factor = 5 * (z ** 2) / (r ** 2)
        # Acceleration components
        ax_j2 = base_coef * x * (z_factor - 1)
        ay_j2 = base_coef * y * (z_factor - 1)
        az_j2 = base_coef * z * (z_factor - 3)
        return np.array([ax_j2, ay_j2, az_j2])

    def propagate_elements (self,  time_span_seconds):
        from orbital_elements import OrbitalElements
        #Get secular rates
        rates = self.calculate_secular_rates()
        #Propagate angular elements
        new_raan = self.orbit.raan + rates['raan_dot'] * time_span_seconds
        new_perigee = self.orbit.perigee + rates['perigee_dot'] * time_span_seconds
        new_mean_anomaly = self.orbit.mean_anomaly + rates['mean_anomaly_dot'] * time_span_seconds
        #Normalize angles to [0, 2pi] range
        new_raan = new_raan % (2 * math.pi)
        new_perigee = new_perigee % (2 * math.pi)
        new_mean_anomaly = new_mean_anomaly % (2 * math.pi)
        #Create new orbital elements
        return OrbitalElements(
            a=self.orbit.a,   #No secular change
            e=self.orbit.e,   #No secular change
            i=self.orbit.i * RAD_TO_DEG,   #No secular change
            raan=new_raan * RAD_TO_DEG,
            perigee=new_perigee * RAD_TO_DEG,
            mean_anomaly=new_mean_anomaly * RAD_TO_DEG,
        )

    def get_analysis_report(self):
        rates = self.calculate_secular_rates()
        #Convert to practical units
        raan_deg_per_day = rates['raan_dot'] * RAD_TO_DEG * 86400
        perigee_deg_per_day = rates['perigee_dot'] * RAD_TO_DEG * 86400
        period_hours = 2 * math.pi / rates['n_bar'] / 3600
        #Calculate precession periods (days for 360 def change)
        if abs(raan_deg_per_day) > 1e-6:
            raan_period_days = 360.0 / abs(raan_deg_per_day)
        else:
            raan_period_days = float('inf')

        if abs(perigee_deg_per_day) > 1e-6:
            perigee_period_days = 360.0 / abs(perigee_deg_per_day)
        else:
            perigee_period_days = float('inf')
        j2_correction_percent = abs(rates['delta_M_j2']/rates['n_kepler']) * 100

        return {
            # === ORBITAL PARAMETERS ===
            'altitude_km': self.orbit.a - EARTH_RADIUS,
            'eccentricity': self.orbit.e,
            'inclination_deg': self.orbit.i * RAD_TO_DEG,
            'orbital_parameter_p0_km': rates['p0'],

            # === ORBITAL PERIOD ===
            'kepler_period_hours': 2 * math.pi / rates['n_kepler'] / 3600,
            'j2_modified_period_hours': period_hours,
            'period_change_percent': ((rates['n_kepler'] - rates['n_bar']) / rates['n_kepler']) * 100,

            # === J2 SECULAR RATES ===
            'raan_precession_deg_day': raan_deg_per_day,
            'perigee_precession_deg_day': perigee_deg_per_day,
            'j2_correction_percent': j2_correction_percent,

            # === PRECESSION PERIODS ===
            'raan_period_days': raan_period_days,
            'perigee_period_days': perigee_period_days,

            # === CLASSIFICATION ===
            'orbit_type': self._classify_orbit(),
            'j2_significance': self._assess_j2_significance(j2_correction_percent),
        }

    def _classify_orbit(self):
        """Classify orbit type based on inclination"""
        i_deg = self.orbit.i * RAD_TO_DEG

        if i_deg < 5:
            return "Equatorial"
        elif 80 <= i_deg <= 100:
            return "Polar"
        elif 95 <= i_deg <= 110:
            return "Sun-synchronous candidate"
        elif i_deg > 90:
            return "Retrograde"
        else:
            return f"Inclined ({i_deg:.1f}°)"

    def _assess_j2_significance(self, correction_percent):
        """Assess significance of J2 effects"""
        if correction_percent < 0.01:
            return "Negligible"
        elif correction_percent < 0.1:
            return "Small"
        elif correction_percent < 1.0:
            return "Moderate"
        else:
            return "Significant"


