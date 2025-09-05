# Orbital Propagation with Perturbation (Originally was only LEO and J2 but updated to full scale orbits with more perturbations)
A Python implementation of orbital mechanics featuring full-scale orbital propagation from Low Earth Orbit (LEO) to Geostationary (GEO) with comprehensive perturbation modeling and real-time 3D visualization.
## Overview
This project provides a complete orbital mechanics simulation toolkit featuring:

- Full-scale orbital propagation (LEO, MEO, GEO, HEO, and beyond)
- Comprehensive J2 perturbations
- Advanced atmospheric drag modeling with altitude-dependent density
- Third-body perturbations (Sun and Moon)
- Real-time 3D orbital visualization with temporal gradient trails
- Enhanced ground track mapping with day/night cycle
- Interactive satellite coverage analysis
- High-precision numerical integration with adaptive algorithms
  
## Perturbation Models
 
## Orbital Elements
- a - Semi-major axis (km)
- e - Eccentricity
- i - Inclination (degrees)
- Ω - RAAN (Right Ascension of Ascending Node) (degrees)
- ω - Argument of Perigee (degrees)
- M - Mean Anomaly (degrees)
  
## Orbit Visualizer

  The orbital visualizer provides real-time 3D visualization of satellite trajectories with rendering capabilities
-Full-scale orbital propagation
-Comprehensive perturbation modeling
-Real-time Parameter Display
![Untitleddesign2-ezgif com-crop (1)](https://github.com/user-attachments/assets/1c086ad5-abe0-4af2-a723-df26b3d4772b)

## Ground Track Visualizer

 The ground track visualizer provides comprehensive satellite coverage analysis with realistic Earth mapping and advanced environmental modeling.
-Day/Night Cycle Modeling
-Coverage Analysis Tools
-Core Mapping Features
![Untitleddesign1-ezgif com-crop](https://github.com/user-attachments/assets/45183aa7-8371-435e-9198-27b79dc8a1ad)

## Supported Orbital Regimes

  Low Earth Orbit (LEO): 160-2000 km
- ISS-type orbits (~400 km altitude)
- Sun-synchronous polar orbits
- CubeSat constellations
- Earth observation missions

  Medium Earth Orbit (MEO): 2000-35,786 km
- GPS constellation (~20200 km)
- Galileo navigation system
- GLONASS positioning
- Regional navigation systems

  Geostationary Earth Orbit (GEO): 35,786 km
- Communication satellites
- Weather monitoring platforms
- Broadcasting systems
- Earth surveillance missions

  Highly Elliptical Orbits (HEO)
- Molniya orbits (12-hour period)
- Tundra orbits (24-hour period)
- Scientific missions

## Theoretical Background
The orbital mechanics theory and mathematical formulations implemented in this project are based off of:
1. "Introduction to Orbital Mechanics" from Colorado Pressbook, available at : https://colorado.pressbooks.pub/introorbitalmechanics/
2. "Fundamentals of Astrodynamics and Applications" - David A. Vallado, Comprehensive perturbation theory and numerical methods
3. "Orbital Mechanics for Engineering Students" -  Howard D. Curtis, Mathematical foundations and practical applications
 
