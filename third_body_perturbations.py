"""
Third-body perturbations for orbital propagation.
Includes gravitational effects of the Sun and Moon on satellite orbits.
"""

import numpy as np
from typing import Tuple

# Constants
MU_SUN = 1.327124400e20  # m³/s² - Sun's gravitational parameter
MU_MOON = 4.902800066e12  # m³/s² - Moon's gravitational parameter
AU = 1.495978707e11  # m - Astronomical Unit
MOON_DISTANCE = 3.844e8  # m - Average Earth-Moon distance
EARTH_OBLIQUITY = 23.439281 * np.pi / 180  # rad - Earth's obliquity

def sun_position(time_hours: float) -> np.ndarray:
    """
    Simplified Sun position model relative to Earth center.
    Uses circular orbit assumption with 1 AU distance.
    
    Args:
        time_hours: Time since epoch in hours
        
    Returns:
        Sun position vector in ECI coordinates [m]
    """
    # Days since J2000.0 epoch
    days = time_hours / 24.0
    
    # Mean anomaly (365.25 days period)
    mean_anomaly = 2 * np.pi * days / 365.25
    
    # Sun position in ecliptic coordinates
    x_ecliptic = AU * np.cos(mean_anomaly)
    y_ecliptic = AU * np.sin(mean_anomaly)
    z_ecliptic = 0.0
    
    # Transform to ECI (account for Earth's obliquity)
    cos_obliq = np.cos(EARTH_OBLIQUITY)
    sin_obliq = np.sin(EARTH_OBLIQUITY)
    
    x_eci = x_ecliptic
    y_eci = y_ecliptic * cos_obliq - z_ecliptic * sin_obliq
    z_eci = y_ecliptic * sin_obliq + z_ecliptic * cos_obliq
    
    return np.array([x_eci, y_eci, z_eci])

def moon_position(time_hours: float) -> np.ndarray:
    """
    Simplified Moon position model relative to Earth center.
    Uses circular orbit assumption with average distance.
    
    Args:
        time_hours: Time since epoch in hours
        
    Returns:
        Moon position vector in ECI coordinates [m]
    """
    # Days since epoch
    days = time_hours / 24.0
    
    # Mean anomaly (27.32 days period)
    mean_anomaly = 2 * np.pi * days / 27.32
    
    # Moon orbital inclination to ecliptic (~5.14 degrees)
    inclination = 5.14 * np.pi / 180
    
    # Moon position in orbital plane
    x_orbital = MOON_DISTANCE * np.cos(mean_anomaly)
    y_orbital = MOON_DISTANCE * np.sin(mean_anomaly) * np.cos(inclination)
    z_orbital = MOON_DISTANCE * np.sin(mean_anomaly) * np.sin(inclination)
    
    # Transform to ECI (simplified - no precession)
    # Apply Earth's obliquity transformation
    cos_obliq = np.cos(EARTH_OBLIQUITY)
    sin_obliq = np.sin(EARTH_OBLIQUITY)
    
    x_eci = x_orbital
    y_eci = y_orbital * cos_obliq - z_orbital * sin_obliq
    z_eci = y_orbital * sin_obliq + z_orbital * cos_obliq
    
    return np.array([x_eci, y_eci, z_eci])

def third_body_acceleration(r_sat: np.ndarray, r_body: np.ndarray, mu_body: float) -> np.ndarray:
    """
    Calculate third-body gravitational acceleration on satellite.
    
    Args:
        r_sat: Satellite position vector [m]
        r_body: Third body position vector [m]
        mu_body: Third body gravitational parameter [m³/s²]
        
    Returns:
        Acceleration vector [m/s²]
    """
    # Vector from body to satellite
    r_rel = r_sat - r_body
    r_rel_mag = np.linalg.norm(r_rel)
    r_body_mag = np.linalg.norm(r_body)
    
    # Third-body acceleration
    a_direct = -mu_body * r_rel / r_rel_mag**3
    a_indirect = -mu_body * r_body / r_body_mag**3
    
    return a_direct + a_indirect

def sun_perturbation(r_sat: np.ndarray, time_hours: float) -> np.ndarray:
    """
    Calculate solar gravitational perturbation acceleration.
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        Solar perturbation acceleration [m/s²]
    """
    r_sun = sun_position(time_hours)
    return third_body_acceleration(r_sat, r_sun, MU_SUN)

def moon_perturbation(r_sat: np.ndarray, time_hours: float) -> np.ndarray:
    """
    Calculate lunar gravitational perturbation acceleration.
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        Lunar perturbation acceleration [m/s²]
    """
    r_moon = moon_position(time_hours)
    return third_body_acceleration(r_sat, r_moon, MU_MOON)

def combined_third_body_acceleration(r_sat: np.ndarray, time_hours: float) -> np.ndarray:
    """
    Calculate combined third-body perturbations (Sun + Moon).
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        Combined third-body acceleration [m/s²]
    """
    a_sun = sun_perturbation(r_sat, time_hours)
    a_moon = moon_perturbation(r_sat, time_hours)
    
    return a_sun + a_moon

def get_sun_satellite_angle(r_sat: np.ndarray, time_hours: float) -> float:
    """
    Calculate angle between Sun and satellite as seen from Earth center.
    Used for solar radiation pressure and eclipse calculations.
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        Angle in radians
    """
    r_sun = sun_position(time_hours)
    
    # Calculate angle using dot product
    cos_angle = np.dot(r_sat, r_sun) / (np.linalg.norm(r_sat) * np.linalg.norm(r_sun))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
    
    return np.arccos(cos_angle)

def is_satellite_in_shadow(r_sat: np.ndarray, time_hours: float) -> bool:
    """
    Determine if satellite is in Earth's shadow (simplified cylindrical model).
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        True if satellite is in shadow, False otherwise
    """
    r_sun = sun_position(time_hours)
    
    # Simplified cylindrical shadow model
    # Check if satellite is on the opposite side of Earth from Sun
    sun_satellite_angle = get_sun_satellite_angle(r_sat, time_hours)
    
    # If angle > 90 degrees, satellite might be in shadow
    if sun_satellite_angle > np.pi / 2:
        # Check if satellite is within Earth's shadow cylinder
        earth_radius = 6378137.0  # m
        r_sat_mag = np.linalg.norm(r_sat)
        
        # Project satellite position onto plane perpendicular to Sun direction
        sun_unit = r_sun / np.linalg.norm(r_sun)
        projection_distance = np.linalg.norm(r_sat - np.dot(r_sat, sun_unit) * sun_unit)
        
        # Check if within shadow cylinder
        return projection_distance < earth_radius
    
    return False

def get_moon_satellite_angle(r_sat: np.ndarray, time_hours: float) -> float:
    """
    Calculate angle between Moon and satellite as seen from Earth center.
    Used for lunar eclipse and shadow calculations.
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        Angle in radians
    """
    r_moon = moon_position(time_hours)
    
    # Calculate angle using dot product
    cos_angle = np.dot(r_sat, r_moon) / (np.linalg.norm(r_sat) * np.linalg.norm(r_moon))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure valid range
    
    return np.arccos(cos_angle)

def is_satellite_in_moon_shadow(r_sat: np.ndarray, time_hours: float) -> bool:
    """
    Determine if satellite is in Moon's shadow (lunar eclipse region).
    Uses simplified cylindrical shadow model.
    
    Args:
        r_sat: Satellite position vector [m]
        time_hours: Time since epoch in hours
        
    Returns:
        True if satellite is in Moon's shadow, False otherwise
    """
    r_moon = moon_position(time_hours)
    r_sun = sun_position(time_hours)
    
    # Vector from Sun to Moon
    sun_to_moon = r_moon - r_sun
    sun_to_moon_unit = sun_to_moon / np.linalg.norm(sun_to_moon)
    
    # Vector from Sun to satellite
    sun_to_sat = r_sat - r_sun
    
    # Project satellite position along Sun-Moon line
    projection_length = np.dot(sun_to_sat, sun_to_moon_unit)
    moon_distance_from_sun = np.linalg.norm(sun_to_moon)
    
    # Check if satellite is beyond Moon from Sun's perspective
    if projection_length > moon_distance_from_sun:
        # Calculate perpendicular distance from Sun-Moon line
        projection_point = r_sun + projection_length * sun_to_moon_unit
        perpendicular_distance = np.linalg.norm(r_sat - projection_point)
        
        # Moon's shadow radius (approximate, using Moon's physical radius)
        moon_radius = 1737400.0  # m
        
        # Check if within shadow cylinder
        return perpendicular_distance < moon_radius
    
    return False

def get_moon_shadow_on_earth(time_hours: float) -> Tuple[float, float, float]:
    """
    Calculate where Moon's shadow (umbra center) falls on Earth's surface during solar eclipse.
    This function simulates the Moon's shadow when it passes between Sun and Earth.
    
    Args:
        time_hours: Time since epoch in hours
        
    Returns:
        Tuple of (latitude, longitude, shadow_radius) in degrees and meters
        Returns (0, 0, 0) if no eclipse is happening
    """
    r_moon = moon_position(time_hours)
    r_sun = sun_position(time_hours)
    
    # Check if Moon and Sun are approximately aligned (solar eclipse conditions)
    # Calculate angle between Sun and Moon as seen from Earth
    sun_unit = r_sun / np.linalg.norm(r_sun)
    moon_unit = r_moon / np.linalg.norm(r_moon)
    alignment_angle = np.arccos(np.clip(np.dot(sun_unit, moon_unit), -1, 1))
    
    # Solar eclipse occurs when Moon is between Earth and Sun (alignment < ~5 degrees)
    eclipse_threshold = 5.0 * np.pi / 180.0  # 5 degrees in radians
    
    # DEBUG: Always show moon position, even without eclipse
    # This makes Moon Shadow feature more useful for visualization
    # Comment out the early return to always show Moon position
    # if alignment_angle > eclipse_threshold:
    #     # No eclipse - return zero shadow
    #     return 0.0, 0.0, 0.0
    
    # Calculate where Moon's shadow falls on Earth
    # This is approximately where the Moon appears to be overhead
    x, y, z = r_moon
    
    # Calculate latitude and longitude where Moon is overhead
    r_xy = np.sqrt(x**2 + y**2)
    moon_lat = np.arctan2(z, r_xy) * 180.0 / np.pi
    moon_lon = np.arctan2(y, x) * 180.0 / np.pi
    
    # Normalize longitude to [-180, 180]
    while moon_lon > 180:
        moon_lon -= 360
    while moon_lon < -180:
        moon_lon += 360
    
    # Calculate shadow radius during eclipse
    moon_radius = 1737400.0   # m
    sun_radius = 696340000.0  # m
    au = 1.495978707e11       # m
    moon_distance = np.linalg.norm(r_moon)
    sun_distance = np.linalg.norm(r_sun)
    
    # Umbra radius on Earth's surface (geometric shadow calculation)
    # This is the radius of the Moon's umbra when it hits Earth
    angular_size_sun = sun_radius / sun_distance
    angular_size_moon = moon_radius / moon_distance
    
    # During total solar eclipse, Moon's angular size > Sun's angular size
    if angular_size_moon > angular_size_sun:
        # Total eclipse - calculate umbra radius
        umbra_angle = angular_size_moon - angular_size_sun
        shadow_radius = 6378137.0 * umbra_angle  # Earth radius * angle
        shadow_radius = np.clip(shadow_radius, 50000, 270000)  # 50-270 km realistic range
    else:
        # Annular eclipse - smaller shadow
        shadow_radius = 100000  # 100 km default
    
    # Decide if it's an eclipse based on alignment
    if alignment_angle <= eclipse_threshold:
        # Eclipse conditions met - scale shadow based on alignment quality
        alignment_factor = max(0, 1.0 - alignment_angle / eclipse_threshold)
        shadow_radius *= alignment_factor
    else:
        # No eclipse - but still return Moon position with zero shadow
        shadow_radius = 0.0
    
    return moon_lat, moon_lon, shadow_radius
