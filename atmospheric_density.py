"""Advanced atmospheric density model for orbital dynamics"""
import numpy as np
import math

class AtmosphericLayer:
    def __init__(self, h_min, h_max, h0, rho0, H, T0=None):
        self.h_min = h_min
        self.h_max = h_max
        self.h0 = h0
        self.rho0 = rho0
        self.H = H
        self.T0 = T0 or 200 #Default temperature

    def ContainsAltitude(self, altitude):
        #Check if altitude is within this layer
        return self.h_min <= altitude <= self.h_max

    def base_density(self, altitude):
        if not self.ContainsAltitude(altitude):
            return 0

        exponent = -(altitude - self.h0) / self.H
        return self.rho0 * np.exp(exponent)

class AtmosphericModel:
    def __init__(self):
        #The numbers inside the parenthesis are, in order, minimum height, maximum height
        #median height, density, scale height for this layer, and temperature in Kelvin
        self.layers = [
            #Troposphere (0-11 km) not relevant for satellites
            AtmosphericLayer(0, 11, 5.5, 1.225, 8.5, 288),
            #Stratosphere (11-50 km) not relevant for satellites
            AtmosphericLayer(11, 50, 30, 1.14e-3, 7.0, 220),
            #Mesosphere (50-85 km) not relevant for most satellites
            AtmosphericLayer(50, 85, 67.5, 3.9e-6, 6.0, 180),
            #Lower Thermosphere (85-200 km) relevant for very low orbits
            AtmosphericLayer(85, 200, 142.5, 5.6e-9, 28, 360),
            #MidThermosphere (200-300 km) ISS lower region bounds
            AtmosphericLayer(200, 300, 250, 2.4e-11, 40, 900),
            #Upper Thermosphere (300-500) IIS region
            AtmosphericLayer(300, 500, 400, 5.6e-13, 50, 1200),
            #Very high Thermosphere (500-800) High LEO
            AtmosphericLayer(500, 800, 650, 1.2e-14, 65, 1400),
            #Exosphere transition (800-1200 km) very High LEO
            AtmosphericLayer(800, 1200, 1000, 2.8e-16, 80, 1500),
            #Lower Exosphere (1200+ km) minimal atmosphere
            AtmosphericLayer(1200, 10000, 2000, 1.0e-18, 100, 1600),
            ]
        #Solar activity parameters based on the F10.7 Solar Flux
        self.f107_min = 70 #Solar minimum
        self.f107_max = 250 #Solar maximum
        self.f107_current = 120 #Current value - moderate

        #Geomagnetic activity
        self.ap_current = 10 #Current AP index (geomagnetic activity)

        #Earth rotation parameters for diurnal effects
        self.earth_rotation_rate = 7.2921159e-5 #rad/s

    def get_layer_for_altitude(self, altitude):
        for layer in self.layers:
            if layer.ContainsAltitude(altitude):
                return layer

        #If altitude is beyond maximum layer, we'll use the highest layer with extrapolation
        return self.layers[-1]

    def base_density(self, altitude):
        layer = self.get_layer_for_altitude(altitude)
        return layer.base_density(altitude)

    def diurnal_variation_factor(self, position_vec, time_seconds):
        #Calculate diurnal(day/night) variation factor
        #Atmosphere is denser on day side due to solar heating.
        #Calculate sub-solar point (where Sun is directly overhead)
        sun_angle = self.earth_rotation_rate * time_seconds
        sun_direction = np.array([np.cos(sun_angle), np.sin(sun_angle), 0])
        #Calculate angle between satellite position and sub-solar point
        pos_normalized = position_vec / np.linalg.norm(position_vec)
        sun_angle_satellite = np.arccos(np.clip(np.dot(pos_normalized, sun_direction), -1.0, 1.0))
        # Diurnal variation: maximum factor on day side, minimum on night side
        # Variation is stronger at higher altitudes (thermosphere)
        altitude = np.linalg.norm(position_vec) - 6371 #Earth radius
        if altitude < 200:
            variation_amplitude = 0.1
        elif altitude < 500:
            variation_amplitude = 0.3
        else:
            variation_amplitude = 0.6
        diurnal_factor = 1.0 + variation_amplitude * np.cos(sun_angle_satellite)

        return max(0.1, diurnal_factor)

    def solar_activity_factor(self):
        #Calculate solar activity factor based on F10.7 index
        #Normalize F10.7 to 0-1 range
        f107_normalized = (self.f107_current - self.f107_min) / (self.f107_max - self.f107_min)
        f107_normalized = np.clip(f107_normalized, 0, 1)
        #Factor ranges from 0.5 (solar minimun) to 3.0 (solar maximum)
        solar_factor = 0.5 + 2.5 * f107_normalized
        return solar_factor

    def latitudinal_variation_factor(self, position_vec):
        #Calculate latitude from position vector
        r_magnitude = np.linalg.norm(position_vec)
        latitude = np.arcsin(position_vec[2] / r_magnitude) #Z component gives latitude
        #Amplitude decreases with altitude
        altitude = r_magnitude - 6371
        if altitude < 300:
            lat_amplitude = 0.15
        else:
            lat_amplitude = 0.10

        lat_factor = 1.0 + lat_amplitude * (np.cos(latitude)**2 - 0.5)
        return lat_factor

    def geomagnetic_activity_factor(self):
        #Calculate geomagnetic activity factor based on Ap index
        #typically ranges from 0 (quiet) to 400+ (severe storm)
        #Normal range is 0-50
        ap_normalized = min(self.ap_current / 50.0, 2) #Cap at 2
        # Geomagnetic effect: 1.0 (quiet) to 2.0 (active)
        geo_factor = 1.0 + 1.0 * ap_normalized
        return geo_factor

    def density (self, position_vec, time_seconds=0.0):
        #Calculate total atmospheric density with all dynamic effects
        # Calculate altitude
        r_magnitude = np.linalg.norm(position_vec)
        altitude = r_magnitude - 6371  # Assuming Earth radius = 6371 km

        if altitude < 0:
            return 1.225
        base_rho = self.base_density(altitude)

        if base_rho <= 0:
            return 0

        #Apply dynamic effects
        diurnal_factor = self.diurnal_variation_factor(position_vec, time_seconds)
        solar_factor = self.solar_activity_factor()
        lat_factor = self.latitudinal_variation_factor(position_vec)
        geo_factor = self.geomagnetic_activity_factor()

        #Total density with all effects
        total_density = base_rho * diurnal_factor * solar_factor * lat_factor * geo_factor
        return total_density

    def set_solar_activity(self, f107_index):
        self.f107_current = np.clip(f107_index, self.f107_min, self.f107_max)
    def set_geomagnetic_activity(self, ap_index):
        self.ap_current = max(0, ap_index)
    def get_atmospheric_info(self, position_vec, time_seconds=0.0):
        #Get detailed atmospheric information for display
        r_magnitude = np.linalg.norm(position_vec)
        altitude = r_magnitude - 6371

        base_density = self.base_density(altitude)
        total_density = self.density(position_vec, time_seconds)

        diurnal_factor = self.diurnal_variation_factor(position_vec, time_seconds)
        solar_factor = self.solar_activity_factor()
        lat_factor = self.latitudinal_variation_factor(position_vec)
        geo_factor = self.geomagnetic_activity_factor()

        layer = self.get_layer_for_altitude(altitude)
        latitude_deg = np.degrees(np.arcsin(position_vec[2] / r_magnitude))

        return {
            'altitude_km': altitude,
            'latitude_deg': latitude_deg,
            'base_density': base_density,
            'total_density': total_density,
            'diurnal_factor': diurnal_factor,
            'solar_factor': solar_factor,
            'latitudinal_factor': lat_factor,
            'geomagnetic_factor': geo_factor,
            'atmospheric_layer': f"{layer.h_min}-{layer.h_max} km",
            'f107_index': self.f107_current,
            'ap_index': self.ap_current
        }


# Test atmospheric model
if __name__ == '__main__':
    print("Testing atmospheric model")

    atmosphere = AtmosphericModel()

    test_positions = [
        np.array([6778, 0, 0]),  # 407 km altitude, equator
        np.array([6778, 0, 3000]),  # 407 km altitude, high latitude
        np.array([7000, 0, 0]),  # 629 km altitude, equator
        np.array([7378, 0, 0])  # 1007 km altitude, equator
    ]
    # Test at different times (day/night cycle)
    test_times = [0, 21600, 43200, 64800]  # 0h, 6h, 12h, 18h

    print(f"Solar activity F10.7: {atmosphere.f107_current}")
    print(f"Geomagnetic activity Ap: {atmosphere.ap_current}")
    print()

    for i, pos in enumerate(test_positions):
        altitude = np.linalg.norm(pos) - 6371
        print(f"Position {i + 1}: Altitude {altitude:.0f} km")
        
        for time_h, time_s in zip([0, 6, 12, 18], test_times):
            density = atmosphere.density(pos, time_s)
            info = atmosphere.get_atmospheric_info(pos, time_s)

            print(f"  {time_h:2d}h: ρ = {density:.2e} kg/m³ "
                        f"(diurnal: {info['diurnal_factor']:.2f}, "
                        f"solar: {info['solar_factor']:.2f})")

        print()

    # Test solar activity variations
    print("Solar Activity Variations:")
    test_pos = np.array([6778, 0, 0])  # ISS altitude

    for f107 in [70, 120, 200, 250]:
        atmosphere.set_solar_activity(f107)
        density = atmosphere.density(test_pos)
        print(f"F10.7 = {f107:3d}: ρ = {density:.2e} kg/m³")


