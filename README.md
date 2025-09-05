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
- Full-scale orbital propagation
- Comprehensive perturbation modeling
- Real-time Parameter Display
![Untitleddesign2-ezgif com-crop (1)](https://github.com/user-attachments/assets/1c086ad5-abe0-4af2-a723-df26b3d4772b)

## Ground Track Visualizer

 The ground track visualizer provides comprehensive satellite coverage analysis with realistic Earth mapping and advanced environmental modeling.
- Day/Night Cycle Modeling
- Coverage Analysis Tools
- Solar Position Tracking
![Untitleddesign1-ezgif com-crop](https://github.com/user-attachments/assets/45183aa7-8371-435e-9198-27b79dc8a1ad)

##  Implemented Perturbation Models

### **Atmospheric Perturbations**

#### **J2 Perturbations**
- **Earth's oblateness effects** - most significant perturbation for LEO satellites
- **Nodal regression** - RAAN precession rate calculation
- **Apsidal precession** - Argument of perigee rotation
- **Modified mean motion** - J2-corrected orbital period
- **Analytical propagation** using secular rate theory
- **Critical for sun-synchronous orbits**
  
#### **Advanced Atmospheric Drag**
- **Multi-layer atmospheric model** (9 atmospheric layers from troposphere to exosphere)
- **Altitude-dependent density** with exponential decay models
- **Solar activity effects** using F10.7 solar flux index
- **Diurnal variations** - day/night density changes
- **Latitudinal effects** - polar vs equatorial density variations
- **Geomagnetic activity** influence via Ap index
- **Earth rotation effects** - relative velocity calculations

**Supported Altitude Ranges:**
- **LEO (160-2000 km)**: Significant drag effects with orbital decay
- **MEO (2000-35,786 km)**: Minimal but measurable atmospheric effects
- **GEO (35,786+ km)**: Negligible atmospheric interactions

###  **Solar Radiation Pressure**

#### **Direct Solar Pressure**
- **Area-to-mass ratio** dependency
- **Solar flux variations** based on F10.7 index
- **Distance-dependent intensity** (inverse square law)

### **Third-Body Perturbations**

#### **Solar Gravitational Effects**
- **Sun gravitational perturbations** using simplified circular orbit model
- **Direct and indirect acceleration** components
- **Time-dependent sun position** calculations
- **Significant for high-altitude orbits** (MEO, GEO)

#### **Lunar Gravitational Effects**
- **Moon gravitational perturbations** with orbital inclination effects
- **27.32-day lunar period** modeling
- **Moon position calculation** relative to Earth
- **Combined Sun-Moon effects** for comprehensive third-body modeling
  
## Theoretical Background
The orbital mechanics theory and mathematical formulations implemented in this project are based off of:
1. "Introduction to Orbital Mechanics" from Colorado Pressbook, available at : https://colorado.pressbooks.pub/introorbitalmechanics/
   - Chapters 3, 6, 9, and 10* particularly relevant to this implementation
3. "Fundamentals of Astrodynamics and Applications" - David A. Vallado, Comprehensive perturbation theory and numerical methods
4. "Orbital Mechanics for Engineering Students" -  Howard D. Curtis, Mathematical foundations and practical applications
   
### **Key Mathematical Models:**
- **Classical Orbital Elements** - six-parameter orbit description
- **Kepler's Equation** - relating mean, eccentric, and true anomalies
- **J2 Secular Theory** - long-term orbital evolution due to Earth's oblateness
- **Atmospheric Density Models** - exponential and empirical atmospheric representations
- **Third-Body Perturbation Theory** - simplified circular orbit models for Sun and Moon
