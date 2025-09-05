"""
Solar Radiation Pressure (SRP) perturbations for orbital propagation.
Uses F10.7 solar flux from atmospheric_density module and includes Earth shadow modeling.
"""

import numpy as np
from third_body_perturbations import sun_position, is_satellite_in_shadow

# Constants
SOLAR_FLUX_STANDARD = 1361.0  # W/m² - Solar constant at 1 AU
C_LIGHT = 299792458.0  # m/s - Speed of light
AU = 1.495978707e11  # m - Astronomical Unit

def f107_to_solar_flux_factor(f107: float) -> float:
    """
    Convert F10.7 index to solar flux variation factor.
    """
    # Use same ranges as atmospheric model
    f107_min = 70   # Solar minimum
    f107_max = 250  # Solar maximum
    
    # Normalize F10.7 to 0-1 range
    f107_normalized = (f107 - f107_min) / (f107_max - f107_min)
    f107_normalized = np.clip(f107_normalized, 0, 1)
    
    # Solar flux varies approximately ±0.1% with solar cycle
    # F10.7 correlation with TSI (Total Solar Irradiance)
    flux_variation_factor = 0.999 + 0.002 * f107_normalized  # Range: 0.999 to 1.001
    
    return flux_variation_factor

def solar_radiation_pressure(r_sat: np.ndarray, v_sat: np.ndarray, time_hours: float, 
                           satellite_mass: float, cross_section: float, 
                           reflectivity: float = 1.3, f107: float = 120.0) -> np.ndarray:
    """
    Calculate solar radiation pressure acceleration.
    """
    # Check if satellite is in Earth's shadow
    if is_satellite_in_shadow(r_sat, time_hours):
        return np.zeros(3)
    
    # Get Sun position and direction to satellite
    r_sun = sun_position(time_hours)
    
    # Vector from Sun to satellite
    sun_to_sat = r_sat - r_sun
    sun_to_sat_mag = np.linalg.norm(sun_to_sat)
    sun_to_sat_unit = sun_to_sat / sun_to_sat_mag
    
    # Solar flux at satellite position (inverse square law)
    sat_sun_distance = sun_to_sat_mag  # Distance from satellite to Sun
    
    # Get solar flux based on F10.7
    base_flux = SOLAR_FLUX_STANDARD * f107_to_solar_flux_factor(f107)
    
    # Adjust for satellite's distance from Sun (inverse square law)
    solar_flux = base_flux * (AU / sat_sun_distance)**2
    
    # Solar radiation pressure force
    # P = (Φ/c) * A * C_r, where Φ is flux, c is speed of light, A is area, C_r is reflectivity
    radiation_pressure = solar_flux / C_LIGHT
    force_magnitude = radiation_pressure * cross_section * reflectivity
    
    # Force direction is along Sun-to-satellite vector
    force_vector = force_magnitude * sun_to_sat_unit
    
    # Convert to acceleration
    acceleration = force_vector / satellite_mass
    
    return acceleration

def srp_perturbation_cannonball(r_sat: np.ndarray, v_sat: np.ndarray, time_hours: float,
                                satellite_params: dict, f107: float = 120.0) -> np.ndarray:
    """
    Calculate SRP perturbation using cannonball model (spherical satellite).
    """
    return solar_radiation_pressure(
        r_sat, v_sat, time_hours,
        satellite_params['mass'],
        satellite_params['cross_section'],
        satellite_params.get('reflectivity', 1.3),
        f107
    )

def get_shadow_function(r_sat: np.ndarray, time_hours: float) -> float:
    """
    Calculate shadow function (0 = full shadow, 1 = full sunlight).
    """
    return 0.0 if is_satellite_in_shadow(r_sat, time_hours) else 1.0

def calculate_beta_angle(r_sat: np.ndarray, v_sat: np.ndarray, time_hours: float) -> float:
    """
    Calculate beta angle - angle between orbital plane and Sun direction.
    """
    r_sun = sun_position(time_hours)
    
    # Orbital angular momentum vector (orbital normal)
    h_vector = np.cross(r_sat, v_sat)
    h_unit = h_vector / np.linalg.norm(h_vector)
    
    sun_unit = r_sun / np.linalg.norm(r_sun)
    
    # Beta angle is 90° - angle between orbital normal and sun vector
    cos_beta = np.abs(np.dot(h_unit, sun_unit))
    beta = np.arcsin(np.clip(cos_beta, 0.0, 1.0))
    
    return beta

def eclipse_duration_estimate(r_sat: np.ndarray, v_sat: np.ndarray, time_hours: float) -> float:
    """
    Estimate eclipse duration for current orbit.
    """
    earth_radius = 6378137.0  # m
    r_mag = np.linalg.norm(r_sat)
    altitude = r_mag - earth_radius
    
    # Calculate beta angle
    beta_angle = calculate_beta_angle(r_sat, v_sat, time_hours)
    
    # Orbital period
    mu_earth = 3.986004418e14  # m³/s²
    period = 2 * np.pi * np.sqrt(r_mag**3 / mu_earth)  # seconds
    
    # Eclipse fraction (simplified)
    if np.abs(beta_angle) > np.arcsin(earth_radius / r_mag):
        return 0.0  # No eclipse
    
    # Maximum eclipse angle
    max_eclipse_angle = 2 * np.arcsin(earth_radius / r_mag)
    
    # Actual eclipse angle considering beta angle
    eclipse_angle = max_eclipse_angle * np.cos(beta_angle)
    
    # Eclipse duration
    eclipse_duration = (eclipse_angle / (2 * np.pi)) * period
    
    return eclipse_duration / 60.0  # Convert to minutes

def srp_acceleration_magnitude_estimate(satellite_mass: float, cross_section: float, 
                                      reflectivity: float = 1.3, f107: float = 120.0) -> float:
    """
    Estimate typical SRP acceleration magnitude (in sunlight).
    """
    solar_flux = SOLAR_FLUX_STANDARD * f107_to_solar_flux_factor(f107)
    radiation_pressure = solar_flux / C_LIGHT
    force_magnitude = radiation_pressure * cross_section * reflectivity
    acceleration_magnitude = force_magnitude / satellite_mass
    
    return acceleration_magnitude
