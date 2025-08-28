"""
This will be a test script which will be used to verify the written code is working
"""
import sys

from orbital_elements import OrbitalElements
import numpy as np
import math
"""For the purpose of these tests i will be using the ISS orbital parameters"""
def test_iss_orbit():
    print("Testing Orbit")

    iss = OrbitalElements(
        a = 6371 + 408, # I will be using Earth's mean radius : Earth radius + altitude
        e = 0.0001,     # Very low eccentricity due to the low Earth orbit
        i = 51.6,       # Inclination
        raan = 0.0,     # For simplicity
        perigee = 0.0,  # Argument of perigee
        mean_anomaly = 0.0  # Mean anomaly: 0 at perigee
    )

    #Calculate position and velocity
    try:
        position, velocity = iss.to_cartesian()

        print(f"Semi-major axis: {iss.a:.2f} km")
        print(f"Orbital period: {iss.get_orbital_period()/3600:.2f} hours")
        print(f"Position (km): [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        print(f"Velocity (km/s): [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")

        #Calculate orbital speed and radius
        r_magnitude = np.linalg.norm(position) # r_magnitude = math.sqrt(position[0]**2 + position[1]**2 + position[2]**2) this would be without np.linalg.norm
        v_magnitude = np.linalg.norm(velocity) # v_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2 + velocity[2]**2) this would be without np.linalg.norm

        print(f"Distance from Earth center: {r_magnitude:.2f} km")
        print(f"Altitude: {r_magnitude-6371:.2f} km")
        print(f"Orbital speed: {v_magnitude:.3f} km/s")

        return True

    except Exception as e:
        print(f"Error in test: {e}")
        return False

def test_circular_orbit():
    print("Testing Equatorial Circular Orbit") # Satellite stays at equatorial level

    orbit = OrbitalElements(
        a = 6371 + 300, # 300 km altitude
        e = 0.0, # Perfectly circular orbit
        i = 0,
        raan = 0.0,
        perigee = 0.0,
        mean_anomaly = 90,
    )
    try:
        position, velocity = orbit.to_cartesian()
        print(f"Position (km): [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        print(f"Velocity (km/s): [{velocity[0]:.2f}, {velocity[1]:.2f}, {velocity[2]:.2f}]")
        # For equatorial orbit, Z should be ~0
        print(f"Z-component (should be ~0): {position[2]:.6f}")

        return True
    except Exception as e:
        print(f"Error in circular test: {e}")
        return False

def test_polar_orbit():
    print("Testing Polar Orbit") # Satellite sees both poles in its orbit
    polar = OrbitalElements(
        a = 6371 + 500, #Altitude of 500 km
        e = 0.0, # Circular
        i = 90.0, # polar orbit
        raan = 0.0,
        perigee = 0.0,
        mean_anomaly = 0.0,
    )
    try:
        position, velocity = polar.to_cartesian()
        print(f"Position (Km): [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
        print(f"Velocity (Km/s): [{velocity[0]:.3f}, {velocity[1]:.3f}, {velocity[2]:.3f}]")
        print(f"Orbital period: {polar.get_orbital_period()/3600:.2f} hours")

        return True
    except Exception as e:
        print(f"Error in polar test: {e}")
        return False
if __name__ == "__main__":
    print("Testing Orbit Implementation")
    success_count = 0
    total_tests = 3
    if test_iss_orbit():
        success_count += 1
    if test_circular_orbit():
        success_count += 1
    if test_polar_orbit():
        success_count += 1
    print(f"\n Test Results")
    print(f"Passed: {success_count}/{total_tests}")

    if success_count == total_tests:
        print("All tests passed")
    else:
        print("Some tests failed")