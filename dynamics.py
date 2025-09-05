"""
Orbital dynamics module with numerical integration
Includes gravitational, J2, and atmospheric drag accelerations with RK4 integrator
"""
import numpy as np
import math
from typing import Tuple, Dict, Any, Optional
from constants import EARTH_RADIUS, J2, EARTH_NU, OMEGA_EARTH
from J2_perturbations import J2Perturbations
from atmospheric_density import AtmosphericModel
from orbital_elements import OrbitalElements
from third_body_perturbations import combined_third_body_acceleration
from solar_radiation_pressure import srp_perturbation_cannonball


class OrbitalDynamics:

    # --- Perturbation models ---
    def solar_radiation_pressure_acceleration(self, position: np.ndarray, velocity: np.ndarray, time_seconds: float) -> np.ndarray:
        if not getattr(self, 'include_srp', False):
            return np.zeros(3)
        
        # Convert position from km to m for SRP module
        position_m = position * 1000
        velocity_m = velocity * 1000  # m/s
        time_hours = time_seconds / 3600
        
        satellite_params = {
            'mass': self.mass,
            'cross_section': self.area,
            'reflectivity': self.Cr
        }
        
        # Get F10.7 from atmospheric model
        f107 = self.atmosphere.f107_current if hasattr(self, 'atmosphere') else 120.0
        
        # Get SRP acceleration in m/s²
        a_srp_m = srp_perturbation_cannonball(position_m, velocity_m, time_hours, satellite_params, f107)
        
        # Convert back to km/s²
        return a_srp_m * 1e-3

    def third_body_acceleration(self, position: np.ndarray, time_seconds: float) -> np.ndarray:
        if not getattr(self, 'include_third_bodies', False):
            return np.zeros(3)
        
        # Convert position from km to m for third-body module
        position_m = position * 1000
        time_hours = time_seconds / 3600
        
        # Get third-body acceleration in m/s²
        a_third_body_m = combined_third_body_acceleration(position_m, time_hours)
        
        # Convert back to km/s²
        return a_third_body_m * 1e-3

    def __init__(self, include_drag: bool = True, drag_coefficient: float = 2.2,
                 satellite_mass: float = 1.0, cross_sectional_area: float = 1.0,
                 include_srp: bool = False, reflectivity_coefficient: float = 1.3,
                 include_third_bodies: bool = False, include_sun: bool = True, include_moon: bool = True):
        self.include_drag = include_drag
        self.Cd = drag_coefficient
        self.mass = satellite_mass
        self.area = cross_sectional_area

        # SRP and third-body flags
        self.include_srp = include_srp
        self.Cr = reflectivity_coefficient
        self.include_third_bodies = include_third_bodies
        self.include_sun = include_sun
        self.include_moon = include_moon

        #Initialize atmospheric model if drag or SRP is enabled (both need F10.7)
        if self.include_drag or self.include_srp:
            self.atmosphere = AtmosphericModel()

    def gravitational_acceleration(self, position: np.ndarray) -> np.ndarray:
        #Calculate gravitational acceleration (central body only)

        r = np.linalg.norm(position)
        if r < 1e-6:
            return np.zeros(3)

        # Newton's law of gravitation: a = -μ/r³ * r_vector
        acceleration_magnitude = -EARTH_NU / (r ** 3)
        return acceleration_magnitude * position

    def j2_acceleration(self, position: np.ndarray) -> np.ndarray:
        #Calculate J2 perturbation acceleration
        x, y, z = position
        r = np.linalg.norm(position)

        if r < 1e-6:
            return np.zeros(3)

        # J2 acceleration coefficient
        mu_re2_j2 = EARTH_NU * (EARTH_RADIUS ** 2) * J2
        base_coef = (3 / 2) * mu_re2_j2 / (r ** 5)

        # Z-dependent factor
        z_factor = 5 * (z ** 2) / (r ** 2)

        # Acceleration components
        ax_j2 = base_coef * x * (z_factor - 1)
        ay_j2 = base_coef * y * (z_factor - 1)
        az_j2 = base_coef * z * (z_factor - 3)

        return np.array([ax_j2, ay_j2, az_j2])

    def atmospheric_drag_acceleration(self, position: np.ndarray,
                                      velocity: np.ndarray, time_seconds: float = 0.0) -> np.ndarray:
        #Calculate atmospheric drag acceleration
        if not self.include_drag:
            return np.zeros(3)

        r = np.linalg.norm(position)
        altitude = r - EARTH_RADIUS

        #No drag above 1000 km
        if altitude > 1000:
            return np.zeros(3)

        #Get atmospheric density with time variation
        density = self.atmosphere.density(position, time_seconds)  #kg/m³

        #Calculate relative velocity (account for Earth's rotation)
        earth_rot_vel = np.array([
            -OMEGA_EARTH * position[1],  #-ωy
            OMEGA_EARTH * position[0],  #ωx
            0.0  # 0
        ])

        #Relative velocity (satellite velocity - atmospheric velocity)
        v_rel = velocity - earth_rot_vel  # km/s
        v_rel_magnitude = np.linalg.norm(v_rel)

        if v_rel_magnitude < 1e-6:
            return np.zeros(3)

        #Drag acceleration: a_drag = -0.5 * ρ * Cd * A * v_rel² / m * v̂_rel
        #Unit analysis: [kg/m³] * [m²] * [km/s]² / [kg] = [m²/s²] = [km²/s²] * 1e-6
        drag_coef = -0.5 * density * self.Cd * self.area / self.mass
        drag_magnitude = drag_coef * (v_rel_magnitude ** 2) * 1e3  # Convert using (km/s)^2 -> m^2/s^2 (1e6) and m/s^2 -> km/s^2 (1e-3) => 1e3

        #Drag direction opposite to relative velocity
        drag_direction = v_rel / v_rel_magnitude

        return drag_magnitude * drag_direction

    def total_acceleration(self, state: np.ndarray, time_seconds: float = 0.0) -> np.ndarray:
        #Calculate total acceleration from all perturbations

        position = state[:3]
        velocity = state[3:]

        #Sum all accelerations
        a = np.zeros(3)
        a += self.gravitational_acceleration(position)
        a += self.j2_acceleration(position)
        a += self.atmospheric_drag_acceleration(position, velocity, time_seconds)
        a += self.third_body_acceleration(position, time_seconds)
        a += self.solar_radiation_pressure_acceleration(position, velocity, time_seconds)
        return a

    def state_derivative(self, state: np.ndarray, time_seconds: float) -> np.ndarray:
        #Calculate state derivative for numerical integration with time-dependent forces
        position = state[:3]
        velocity = state[3:]
        acceleration = self.total_acceleration(state, time_seconds)
        # State derivative: [velocity, acceleration]
        return np.concatenate([velocity, acceleration])

    def rk4_step(self, state: np.ndarray, dt: float, t: float) -> np.ndarray:
        #Single Runge-Kutta 4th order integration step with time-dependent acceleration
        k1 = dt * self.state_derivative(state, t)
        k2 = dt * self.state_derivative(state + k1 / 2, t + dt / 2)
        k3 = dt * self.state_derivative(state + k2 / 2, t + dt / 2)
        k4 = dt * self.state_derivative(state + k3, t + dt)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def propagate_orbit(self, initial_elements: OrbitalElements,
                        duration_hours: float, time_step_seconds: float = 60.0) -> Dict[str, Any]:

        #Convert to Cartesian coordinates
        r0, v0 = initial_elements.to_cartesian()
        initial_state = np.concatenate([r0, v0])

        # Time parameters
        total_time = duration_hours * 3600  # Convert to seconds
        num_steps = int(total_time / time_step_seconds)

        #Storage arrays
        times = np.zeros(num_steps + 1)
        states = np.zeros((num_steps + 1, 6))
        accelerations = np.zeros((num_steps + 1, 3))

        #Initial conditions
        times[0] = 0.0
        states[0] = initial_state
        accelerations[0] = self.total_acceleration(initial_state, 0.0)

        #Numerical integration loop
        current_state = initial_state.copy()
        current_time = 0.0
        for i in range(num_steps):
            #RK4 integration step
            current_state = self.rk4_step(current_state, time_step_seconds, current_time)
            # Advance time
            current_time += time_step_seconds
            #Store results
            times[i + 1] = current_time
            states[i + 1] = current_state
            accelerations[i + 1] = self.total_acceleration(current_state, current_time)

        return {
            'times': times / 3600,  #Convert to hours
            'positions': states[:, :3],  #km
            'velocities': states[:, 3:],  #km/s
            'accelerations': accelerations,  #km/s²
            'initial_elements': initial_elements,
            'time_step': time_step_seconds,
            'duration_hours': duration_hours
        }

    def get_orbital_energy(self, position: np.ndarray, velocity: np.ndarray) -> float:
        #Calculate specific orbital energy

        r = np.linalg.norm(position)
        v = np.linalg.norm(velocity)

        #Specific energy: E = v²/2 - μ/r
        return (v ** 2 / 2) - (EARTH_NU / r)

    def get_analysis_metrics(self, propagation_result: Dict) -> Dict[str, Any]:
        #Analyze propagation results and compute orbital metrics

        positions = propagation_result['positions']
        velocities = propagation_result['velocities']
        times = propagation_result['times']

        #Calculate metrics for each time step
        altitudes = np.array([np.linalg.norm(pos) - EARTH_RADIUS for pos in positions])
        energies = np.array([self.get_orbital_energy(pos, vel)
                             for pos, vel in zip(positions, velocities)])
        speeds = np.array([np.linalg.norm(vel) for vel in velocities])

        #Summary statistics
        return {
            'altitude_stats': {
                'initial_km': altitudes[0],
                'final_km': altitudes[-1],
                'min_km': np.min(altitudes),
                'max_km': np.max(altitudes),
                'mean_km': np.mean(altitudes),
                'decay_rate_km_per_hour': (altitudes[-1] - altitudes[0]) / times[-1]
            },
            'energy_stats': {
                'initial': energies[0],
                'final': energies[-1],
                'change': energies[-1] - energies[0],
                'decay_rate_per_hour': (energies[-1] - energies[0]) / times[-1]
            },
            'speed_stats': {
                'initial_km_s': speeds[0],
                'final_km_s': speeds[-1],
                'min_km_s': np.min(speeds),
                'max_km_s': np.max(speeds),
                'mean_km_s': np.mean(speeds)
            },
            'propagation_info': {
                'duration_hours': times[-1],
                'time_step_seconds': propagation_result['time_step'],
                'total_steps': len(times),
                'drag_enabled': self.include_drag
            }
        }

    def compare_with_analytical(self, initial_elements: OrbitalElements,
                                duration_hours: float) -> Dict[str, Any]:
        #Compare numerical propagation with analytical J2 theory
        #Numerical propagation
        numerical_result = self.propagate_orbit(initial_elements, duration_hours)

        #Analytical J2 propagation
        j2_model = J2Perturbations(initial_elements)
        analytical_elements = j2_model.propagate_elements(duration_hours * 3600)

        #Convert final numerical state back to orbital elements
        final_pos = numerical_result['positions'][-1]
        final_vel = numerical_result['velocities'][-1]
        final_numerical_elements = OrbitalElements.from_cartesian(final_pos, final_vel)

        return {
            'initial_elements': initial_elements,
            'final_numerical_elements': final_numerical_elements,
            'final_analytical_elements': analytical_elements,
            'differences': {
                'semi_major_axis_km': final_numerical_elements.a - analytical_elements.a,
                'eccentricity': final_numerical_elements.e - analytical_elements.e,
                'inclination_deg': final_numerical_elements.i - analytical_elements.i,
                'raan_deg': final_numerical_elements.raan - analytical_elements.raan,
                'perigee_deg': final_numerical_elements.perigee - analytical_elements.perigee,
                'mean_anomaly_deg': final_numerical_elements.mean_anomaly - analytical_elements.mean_anomaly
            },
            'numerical_result': numerical_result
        }