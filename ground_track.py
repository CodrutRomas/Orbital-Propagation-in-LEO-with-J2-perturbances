"""
Enhanced Ground Track Visualization with Earth Map and Day/Night Cycle
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from datetime import datetime, timedelta
from collections import deque
import time
import os

from constants import EARTH_RADIUS, RAD_TO_DEG, DEG_TO_RAD


class GroundTrackWindow:
    def __init__(self, parent=None):
        """Initialize enhanced ground track visualization window"""
        # Create new toplevel window
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("Satellite Ground Track - Enhanced")
        self.window.geometry("1400x800")
        
        # Performance settings
        self.performance_mode = False  # Start with performance mode OFF for better visual quality
        self.update_interval = 500 if self.performance_mode else 200  # milliseconds between canvas updates
        self.last_update_time = 0
        self.pending_update = False
        self.update_in_progress = False
        
        # Data storage - using deque for O(1) operations
        self.track_points = deque(maxlen=2000)  # Automatic size limiting
        self.current_position = None
        self.simulation_time = 0  # Store simulation time for day/night cycle
        
        # Default settings
        self.max_points = 1500  # Default track length
        self.use_simulation_time = True  # Use simulation time by default for day/night
        
        # Try to load Earth image
        self.earth_image = None
        earth_image_path = os.path.join(os.path.dirname(__file__), 'earth.jpg')
        if os.path.exists(earth_image_path):
            try:
                self.earth_image = mpimg.imread(earth_image_path)
            except Exception as e:
                print(f"Could not load earth.jpg: {e}")
                self.earth_image = None
        
        # Initialize day/night element tracking variables BEFORE drawing
        self.night_patches = []
        self.sun_marker = None
        self.terminator_line = None
        self.sun_label = None
        self.time_label = None
        self.coverage_circle = None
        self.coverage_text = None
        
        # Setup UI
        self._setup_ui()
        
        # Draw initial map with Earth features
        self._draw_base_map()
        
        # Setup update timer
        self.window.after(self.update_interval, self._scheduled_update)
        
    def _setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel (top)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=5)
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Controls row 1
        row1 = ttk.Frame(control_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Button(row1, text="Clear Track", command=self.clear_track).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Center View", command=self.center_view).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        
        # Show options
        self.show_grid = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Grid", variable=self.show_grid, 
                       command=self._update_display).pack(side=tk.LEFT, padx=10)
        
        self.show_daynight = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Day/Night", variable=self.show_daynight,
                       command=self._update_display).pack(side=tk.LEFT, padx=2)
        
        self.show_coverage = tk.BooleanVar(value=True)
        ttk.Checkbutton(row1, text="Coverage", variable=self.show_coverage,
                       command=self._update_display).pack(side=tk.LEFT, padx=2)
        
        # Day/Night time source toggle
        self.use_sim_time_var = tk.BooleanVar(value=self.use_simulation_time)
        ttk.Checkbutton(row1, text="Sim Time", variable=self.use_sim_time_var,
                       command=self._toggle_time_source).pack(side=tk.LEFT, padx=(10, 2))
        
        # Performance mode toggle
        self.performance_var = tk.BooleanVar(value=False)  # OFF by default
        perf_btn = ttk.Checkbutton(row1, text="Performance Mode", variable=self.performance_var,
                                   command=self._toggle_performance_mode)
        perf_btn.pack(side=tk.LEFT, padx=(10, 2))
        
        # Performance indicator label
        self.perf_label = ttk.Label(row1, text="[PERF OFF]",  # OFF by default
                                   foreground="gray")
        self.perf_label.pack(side=tk.LEFT, padx=2)
        
        # Track length control
        ttk.Label(row1, text="Track Length:").pack(side=tk.LEFT, padx=(20, 5))
        self.track_length_var = tk.IntVar(value=self.max_points)
        track_scale = ttk.Scale(row1, from_=100, to=5000, orient=tk.HORIZONTAL,
                               variable=self.track_length_var, command=self._update_track_length)
        track_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.track_length_label = ttk.Label(row1, text=str(self.max_points))
        self.track_length_label.pack(side=tk.LEFT, padx=2)
        
        # Info panel (bottom)
        info_frame = ttk.LabelFrame(main_frame, text="Satellite Position", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Two-row info display
        self.info_label1 = ttk.Label(info_frame, text="Waiting for data...", 
                                    font=("Courier New", 10))
        self.info_label1.pack(anchor=tk.W)
        
        self.info_label2 = ttk.Label(info_frame, text="", 
                                    font=("Courier New", 10))
        self.info_label2.pack(anchor=tk.W)
        
        # Map canvas
        map_frame = ttk.Frame(main_frame)
        map_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with better aspect ratio
        self.fig = plt.Figure(figsize=(14, 7), dpi=85, tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar_frame = ttk.Frame(map_frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def _toggle_time_source(self):
        """Toggle between real-time and simulation time for day/night cycle"""
        self.use_simulation_time = self.use_sim_time_var.get()
        self._update_display()
    
    def _toggle_performance_mode(self):
        """Toggle performance mode on/off"""
        self.performance_mode = self.performance_var.get()
        
        if self.performance_mode:
            # Enable performance optimizations
            self.update_interval = 500  # Slower updates
            self.max_points = min(1500, self.max_points)
            self.track_points = deque(list(self.track_points)[-self.max_points:], maxlen=self.max_points)
            self.perf_label.config(text="[PERF ON]", foreground="green")
            
            # Simplify display
            if len(self.track_points) > 1000:
                # Keep only recent points
                recent = list(self.track_points)[-1000:]
                self.track_points = deque(recent, maxlen=self.max_points)
        else:
            # Disable performance optimizations
            self.update_interval = 200  # Faster updates
            self.max_points = min(3000, self.max_points)
            self.track_points = deque(list(self.track_points), maxlen=self.max_points)
            self.perf_label.config(text="[PERF OFF]", foreground="gray")
        
        # Update track length control
        self.track_length_var.set(self.max_points)
        self.track_length_label.config(text=str(self.max_points))
        
        # Redraw with new settings
        self._update_plot()
    
    def _draw_base_map(self):
        """Draw the base world map with Earth features"""
        self.ax.clear()
        
        # Set map properties
        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.ax.set_xlabel('Longitude (degrees)', fontsize=10)
        self.ax.set_ylabel('Latitude (degrees)', fontsize=10)
        time_mode = "Simulation Time" if self.use_simulation_time else "Real-Time"
        feature_str = "Day/Night" if self.show_daynight.get() else "Basic"
        self.ax.set_title(f'Satellite Ground Track - {time_mode} ({feature_str})', fontsize=12, fontweight='bold')
        self.ax.set_aspect('equal')
        
        # Draw Earth image if available, otherwise use ocean color
        if self.earth_image is not None:
            # Display the Earth image as background
            # Use interpolation for better quality with high-res images
            self.ax.imshow(self.earth_image, extent=[-180, 180, -90, 90], 
                          aspect='auto', alpha=1.0, zorder=0, interpolation='bilinear')
        else:
            # Set background color (ocean)
            self.ax.set_facecolor('#e6f2ff')  # Light blue for ocean
        
        # Draw Earth features (continents) if no image is available
        if self.earth_image is None:
            self._draw_earth_features()
        
        # Draw day/night terminator
        if self.show_daynight.get():
            self._draw_day_night_terminator()
        
        # Draw grid
        if self.show_grid.get():
            self.ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            # Major grid lines
            for lon in range(-180, 181, 30):
                self.ax.axvline(x=lon, color='gray', alpha=0.2, linewidth=0.5)
            for lat in range(-90, 91, 30):
                self.ax.axhline(y=lat, color='gray', alpha=0.2, linewidth=0.5)
        
        # Draw special latitude lines
        self.ax.axhline(y=0, color='darkblue', linewidth=1.0, alpha=0.4, 
                       linestyle='--', label='Equator')
        self.ax.axhline(y=23.5, color='orange', linewidth=0.6, alpha=0.3, 
                       linestyle=':')  # Tropic of Cancer
        self.ax.axhline(y=-23.5, color='orange', linewidth=0.6, alpha=0.3, 
                       linestyle=':')  # Tropic of Capricorn
        self.ax.axhline(y=66.5, color='cyan', linewidth=0.6, alpha=0.3, 
                       linestyle=':')  # Arctic Circle
        self.ax.axhline(y=-66.5, color='cyan', linewidth=0.6, alpha=0.3, 
                       linestyle=':')  # Antarctic Circle
        
        # Initialize track line with LineCollection for better performance
        self.track_collection = LineCollection([], colors='red', linewidths=2, alpha=0.9)
        self.ax.add_collection(self.track_collection)
        
        # Current position marker
        self.current_pos_marker, = self.ax.plot([], [], 'ro', markersize=10, 
                                               label='Current Position', zorder=10)
        
        # Note: day/night tracking variables are already initialized in __init__
        
        # Add legend
        self.ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        
        self.canvas.draw()
        
    def _draw_earth_features(self):
        """Draw more realistic Earth continents and features"""
        # Define continents with more realistic shapes
        # Using simplified but recognizable continent shapes
        
        # North America
        north_america = [
            [-170, 70], [-160, 70], [-150, 68], [-140, 69], [-130, 65],
            [-125, 60], [-120, 55], [-115, 50], [-110, 48], [-105, 45],
            [-100, 49], [-95, 49], [-90, 48], [-85, 45], [-80, 43],
            [-75, 45], [-70, 45], [-65, 45], [-60, 47], [-55, 50],
            [-52, 52], [-55, 55], [-60, 58], [-65, 60], [-70, 62],
            [-75, 65], [-80, 68], [-85, 70], [-90, 71], [-95, 72],
            [-100, 73], [-105, 72], [-110, 71], [-115, 70], [-120, 68],
            [-125, 70], [-130, 71], [-135, 70], [-140, 71], [-145, 70],
            [-150, 71], [-155, 70], [-160, 71], [-165, 69], [-170, 70]
        ]
        
        # Central America
        central_america = [
            [-105, 22], [-97, 25], [-90, 21], [-88, 18], [-87, 15],
            [-84, 10], [-83, 8], [-80, 8], [-77, 7], [-76, 8],
            [-78, 9], [-80, 10], [-82, 12], [-84, 14], [-86, 16],
            [-88, 17], [-91, 18], [-95, 20], [-100, 20], [-105, 22]
        ]
        
        # South America
        south_america = [
            [-80, 12], [-78, 10], [-76, 8], [-74, 5], [-70, 2],
            [-68, 0], [-70, -3], [-72, -5], [-74, -8], [-76, -10],
            [-77, -12], [-78, -15], [-77, -20], [-75, -25], [-73, -30],
            [-71, -35], [-70, -40], [-68, -45], [-67, -50], [-68, -52],
            [-70, -53], [-72, -52], [-74, -50], [-73, -48], [-71, -45],
            [-69, -40], [-67, -35], [-65, -30], [-63, -28], [-60, -25],
            [-57, -23], [-55, -20], [-52, -15], [-50, -10], [-48, -5],
            [-46, 0], [-44, -2], [-42, -3], [-40, -5], [-38, -7],
            [-35, -5], [-35, -3], [-38, 0], [-40, 2], [-42, 3],
            [-45, 5], [-48, 6], [-50, 5], [-52, 3], [-54, 2],
            [-56, 3], [-58, 5], [-60, 6], [-62, 8], [-64, 9],
            [-66, 10], [-68, 10], [-70, 9], [-72, 8], [-74, 9],
            [-76, 8], [-78, 9], [-80, 12]
        ]
        
        # Africa
        africa = [
            [-17, 15], [-10, 20], [-5, 25], [0, 30], [5, 32],
            [10, 33], [12, 35], [10, 37], [8, 36], [5, 36],
            [0, 35], [-5, 35], [-7, 34], [-10, 32], [-12, 28],
            [-15, 27], [-17, 25], [-17, 20], [-16, 15], [-15, 10],
            [-12, 5], [-10, 0], [-8, -5], [-5, -10], [0, -15],
            [5, -20], [10, -25], [15, -28], [18, -30], [20, -32],
            [22, -34], [25, -34], [28, -32], [30, -30], [32, -28],
            [35, -25], [37, -20], [40, -15], [42, -10], [45, -5],
            [48, 0], [50, 5], [51, 10], [50, 12], [48, 15],
            [45, 18], [42, 20], [40, 23], [38, 25], [35, 28],
            [32, 30], [30, 32], [28, 30], [25, 28], [20, 25],
            [15, 23], [10, 20], [5, 18], [0, 15], [-5, 13],
            [-10, 12], [-15, 13], [-17, 15]
        ]
        
        # Europe
        europe = [
            [-10, 43], [-8, 44], [-5, 43], [-2, 43], [0, 44],
            [3, 43], [5, 44], [7, 45], [10, 46], [12, 45],
            [15, 45], [18, 46], [20, 45], [22, 44], [25, 45],
            [28, 46], [30, 48], [32, 50], [35, 52], [38, 54],
            [40, 56], [42, 58], [45, 60], [48, 62], [50, 64],
            [52, 66], [55, 68], [60, 70], [65, 71], [70, 71],
            [75, 70], [70, 68], [65, 66], [60, 64], [55, 62],
            [50, 60], [45, 58], [40, 55], [35, 53], [30, 51],
            [25, 50], [20, 52], [15, 54], [10, 55], [5, 57],
            [0, 58], [-5, 57], [-8, 55], [-10, 52], [-10, 48],
            [-9, 45], [-10, 43]
        ]
        
        # Asia (simplified)
        asia = [
            [25, 45], [30, 43], [35, 40], [40, 38], [45, 35],
            [50, 33], [55, 30], [60, 28], [65, 25], [70, 23],
            [75, 20], [80, 18], [85, 20], [90, 22], [95, 23],
            [100, 20], [105, 18], [110, 20], [115, 23], [120, 25],
            [125, 28], [130, 30], [135, 33], [140, 35], [145, 38],
            [150, 40], [155, 43], [160, 45], [165, 48], [170, 50],
            [175, 52], [180, 55], [175, 58], [170, 60], [165, 62],
            [160, 64], [155, 66], [150, 68], [145, 70], [140, 71],
            [135, 70], [130, 68], [125, 65], [120, 62], [115, 60],
            [110, 58], [105, 55], [100, 52], [95, 50], [90, 48],
            [85, 50], [80, 52], [75, 55], [70, 58], [65, 60],
            [60, 62], [55, 60], [50, 58], [45, 55], [40, 52],
            [35, 50], [30, 48], [25, 45]
        ]
        
        # Australia
        australia = [
            [113, -22], [115, -20], [118, -18], [122, -17], [125, -15],
            [128, -14], [132, -12], [135, -11], [138, -10], [142, -10],
            [145, -12], [148, -15], [150, -18], [152, -22], [153, -25],
            [152, -28], [150, -32], [148, -35], [145, -38], [142, -39],
            [138, -38], [135, -36], [132, -34], [128, -32], [125, -30],
            [122, -28], [118, -26], [115, -24], [113, -22]
        ]
        
        # Draw all continents with Earth-like colors
        continents = [
            (north_america, '#8fbc8f'),  # Dark sea green
            (central_america, '#9acd32'),  # Yellow green
            (south_america, '#8fbc8f'),  # Dark sea green
            (africa, '#daa520'),  # Goldenrod (desert-ish)
            (europe, '#90ee90'),  # Light green
            (asia, '#bdb76b'),  # Dark khaki
            (australia, '#cd853f'),  # Peru (desert-ish)
        ]
        
        for continent, color in continents:
            if len(continent) > 2:  # Ensure we have enough points
                poly = Polygon(continent, facecolor=color, edgecolor='darkgreen', 
                             alpha=0.6, linewidth=0.5, zorder=2)
                self.ax.add_patch(poly)
        
        # Add major islands
        # Greenland
        greenland = [[-55, 83], [-45, 83], [-35, 82], [-30, 78], [-25, 72],
                    [-30, 68], [-35, 65], [-40, 63], [-45, 60], [-50, 62],
                    [-55, 65], [-58, 70], [-60, 75], [-58, 80], [-55, 83]]
        poly = Polygon(greenland, facecolor='white', edgecolor='gray', 
                      alpha=0.8, linewidth=0.5, zorder=2)
        self.ax.add_patch(poly)
        
        # Madagascar
        madagascar = [[43, -12], [48, -13], [50, -16], [49, -20], [47, -24],
                     [44, -25], [43, -23], [42, -20], [42, -16], [43, -12]]
        poly = Polygon(madagascar, facecolor='#8fbc8f', edgecolor='darkgreen', 
                      alpha=0.6, linewidth=0.5, zorder=2)
        self.ax.add_patch(poly)
        
        # Japan (simplified)
        japan = [[130, 30], [132, 32], [135, 35], [138, 38], [141, 40],
                [142, 43], [141, 45], [139, 43], [136, 40], [133, 37],
                [131, 34], [130, 30]]
        poly = Polygon(japan, facecolor='#90ee90', edgecolor='darkgreen', 
                      alpha=0.6, linewidth=0.5, zorder=2)
        self.ax.add_patch(poly)
    
    
    def _draw_day_night_terminator(self):
        """Draw realistic day/night terminator based on current or simulation time"""
        # ALWAYS start with complete cleanup to prevent accumulation
        self._remove_all_day_night_elements()
        
        # Clear the night patches list to start fresh
        self.night_patches = []
        
        # Get time based on mode
        if self.use_simulation_time and hasattr(self, 'simulation_time'):
            # Convert simulation seconds to a datetime
            now = datetime.utcnow() + timedelta(seconds=self.simulation_time)
        else:
            # Use real UTC time
            now = datetime.utcnow()
        
        # Calculate solar position
        day_of_year = now.timetuple().tm_yday
        
        # Solar declination (simplified but more accurate)
        # Maximum declination is ±23.45 degrees
        P = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
        sun_lat = P  # Solar declination in degrees
        
        # Calculate sun longitude based on time of day
        # Sun moves 15 degrees per hour (360/24)
        hours_from_noon = now.hour + now.minute/60.0 + now.second/3600.0 - 12
        sun_lon = -hours_from_noon * 15  # Negative because sun moves west
        
        # Create the terminator line (great circle 90 degrees from sun position)
        lons = np.linspace(-180, 180, 360)
        terminator_lats = []
        
        for lon in lons:
            # Angular distance from sun position
            delta_lon = lon - sun_lon
            # Normalize to [-180, 180]
            while delta_lon > 180:
                delta_lon -= 360
            while delta_lon < -180:
                delta_lon += 360
            
            # Calculate terminator latitude at this longitude
            # This is the latitude where the sun is exactly on the horizon
            if abs(delta_lon) <= 90:
                # Daylight side
                term_lat = np.arctan(-np.cos(delta_lon * DEG_TO_RAD) / 
                                    np.tan(sun_lat * DEG_TO_RAD)) * RAD_TO_DEG
            else:
                # Night side
                term_lat = np.arctan(np.cos((delta_lon - 180) * DEG_TO_RAD) / 
                                   np.tan(sun_lat * DEG_TO_RAD)) * RAD_TO_DEG
            
            terminator_lats.append(term_lat)
        
        # Create night shading
        # Split into segments to handle the date line properly
        segments = []
        current_segment = {'lons': [], 'lats': []}
        
        for i, (lon, lat) in enumerate(zip(lons, terminator_lats)):
            if i > 0 and abs(lon - lons[i-1]) > 180:
                # Date line crossing
                if current_segment['lons']:
                    segments.append(current_segment)
                current_segment = {'lons': [], 'lats': []}
            current_segment['lons'].append(lon)
            current_segment['lats'].append(lat)
        
        if current_segment['lons']:
            segments.append(current_segment)
        
        # Draw night regions
        for segment in segments:
            if len(segment['lons']) > 1:
                # Determine which side is night
                mid_lon = segment['lons'][len(segment['lons'])//2]
                lon_diff = mid_lon - sun_lon
                while lon_diff > 180:
                    lon_diff -= 360
                while lon_diff < -180:
                    lon_diff += 360
                
                if abs(lon_diff) > 90:
                    # This segment is on the night side
                    # Create a polygon patch instead of fill_between for better control
                    if sun_lat > 0:
                        # Northern summer, fill to south
                        night_patch = mpatches.Polygon(
                            [(lon, lat) for lon, lat in zip(segment['lons'], segment['lats'])] +
                            [(segment['lons'][-1], -90), (segment['lons'][0], -90)],
                            facecolor='black', alpha=0.25, edgecolor='none', zorder=3
                        )
                    else:
                        # Southern summer, fill to north
                        night_patch = mpatches.Polygon(
                            [(lon, lat) for lon, lat in zip(segment['lons'], segment['lats'])] +
                            [(segment['lons'][-1], 90), (segment['lons'][0], 90)],
                            facecolor='black', alpha=0.25, edgecolor='none', zorder=3
                        )
                    self.ax.add_patch(night_patch)
                    # Store reference for later removal
                    self.night_patches.append(night_patch)
        
        # Draw the terminator line itself and store reference
        self.terminator_line = self.ax.plot(lons, terminator_lats, 'gray', linewidth=1, alpha=0.5, 
                                           linestyle='--', zorder=4)
        
        # Add sun position marker and store reference
        # The sun marker shows the subsolar point (where sun is directly overhead)
        self.sun_marker, = self.ax.plot(sun_lon, sun_lat, 'y*', markersize=15, 
                                       label='Subsolar Point', zorder=5)
        self.sun_marker._is_sun_marker = True  # Tag for easy removal
        
        # Add label near sun position and store reference
        self.sun_label = self.ax.text(sun_lon, sun_lat - 5, 'Sun overhead', 
                                     horizontalalignment='center', fontsize=7,
                                     color='orange', weight='bold',
                                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))
        
        # Add time label at bottom and store reference
        time_str = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        mode_str = "(Sim)" if self.use_simulation_time else "(Real)"
        self.time_label = self.ax.text(0, -85, f"Time: {time_str} {mode_str}", 
                                      horizontalalignment='center', fontsize=9, 
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    
    def update_position(self, lat, lon, altitude, time_seconds=None, is_sun_sync=False):
        """Update satellite position and track with polar orbit optimization"""
        # Handle longitude wrapping
        while lon > 180:
            lon -= 360
        while lon < -180:
            lon += 360
            
        # Store position and simulation time
        self.current_position = (lat, lon, altitude, time_seconds)
        if time_seconds is not None:
            self.simulation_time = time_seconds
        
        # Store sun-sync status for special visualization
        self.is_sun_sync_orbit = getattr(self, 'is_sun_sync_orbit', False) or is_sun_sync
        
        # Intelligent filtering for polar orbits to reduce visual artifacts
        should_add_point = True
        if self.track_points:
            last_lat, last_lon, _ = self.track_points[-1]
            
            # Calculate differences
            lat_diff = abs(lat - last_lat)
            lon_diff = abs(lon - last_lon)
            
            # More aggressive filtering for polar orbits
            if abs(lat) > 80 or abs(last_lat) > 80:  # Very close to poles
                # Be very selective near poles
                if lat_diff < 2.0 and lon_diff < 15.0:
                    should_add_point = False
            elif abs(lat) > 70 or abs(last_lat) > 70:  # High latitude
                if lat_diff < 1.0 and lon_diff < 5.0:
                    should_add_point = False
            elif lat_diff < 0.3 and lon_diff < 0.3:  # General filtering for close points
                should_add_point = False
        
        # Add to track only if point is significant
        if should_add_point:
            self.track_points.append((lat, lon, time_seconds))
        
        # Request update (throttled)
        self._request_update()
        
        # Update info immediately (lightweight operation)
        self._update_info(lat, lon, altitude, time_seconds)
    
    def _request_update(self):
        """Request a plot update (throttled to prevent overload)"""
        current_time = time.time() * 1000  # milliseconds
        
        # Only update if enough time has passed
        if current_time - self.last_update_time > self.update_interval:
            self.pending_update = True
    
    def _scheduled_update(self):
        """Scheduled update function called periodically"""
        # Always update day/night cycle if enabled
        if self.show_daynight.get():
            self._update_day_night_only()
            self.canvas.draw_idle()
        
        # Update plot if there's pending data
        if self.pending_update and not self.update_in_progress:
            self.update_in_progress = True
            try:
                self._update_plot()
                self.pending_update = False
                self.last_update_time = time.time() * 1000
            finally:
                self.update_in_progress = False
        
        # Schedule next update
        self.window.after(self.update_interval, self._scheduled_update)
    
    def _update_plot(self):
        """Update the ground track plot"""
        # Don't clear the track - only update what's necessary
        # First check if we need to redraw the base (only if settings changed)
        if not hasattr(self, '_last_settings') or self._settings_changed():
            self._draw_base_map()
            self._last_settings = self._get_current_settings()
        else:
            # Just update day/night if enabled
            if self.show_daynight.get():
                # Clear only the day/night elements and redraw them
                self._update_day_night_only()
        
        if not self.track_points:
            return
        
        # Extract coordinates
        points = list(self.track_points)
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        # Improved longitude wrapping handling for polar orbits
        segments = []
        current_segment = []
        
        # Threshold for detecting longitude wrapping (adjusted for polar orbits)
        wrap_threshold = 180  # Standard threshold for longitude wrapping
        
        for i in range(len(lons)):
            if i > 0:
                lon_diff = abs(lons[i] - lons[i-1])
                lat_diff = abs(lats[i] - lats[i-1])
                
                # More aggressive breaking for polar orbits
                should_break = False
                
                # Break on longitude wrapping (date line crossing)
                if lon_diff > wrap_threshold:
                    should_break = True
                
                # Break on pole crossing - much more restrictive
                if (abs(lats[i]) > 75 or abs(lats[i-1]) > 75):
                    # Near poles, break on smaller changes
                    if lat_diff > 30 or lon_diff > 90:
                        should_break = True
                
                # Break on very large latitude changes (pole flip)
                if lat_diff > 150:
                    should_break = True
                
                # Break on extreme longitude changes when far from poles
                if abs(lats[i]) < 60 and abs(lats[i-1]) < 60 and lon_diff > 90:
                    should_break = True
                
                if should_break:
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
            
            current_segment.append([lons[i], lats[i]])
        
        if current_segment:
            segments.append(current_segment)
        
        # Convert segments to LineCollection format with additional validation
        line_segments = []
        for segment in segments:
            if len(segment) > 1:
                for i in range(len(segment) - 1):
                    p1, p2 = segment[i], segment[i+1]
                    lon_diff = abs(p2[0] - p1[0])
                    lat_diff = abs(p2[1] - p1[1])
                    
                    # Skip line segments that are too long (prevent cross-map artifacts)
                    if lon_diff < 180 and lat_diff < 170:
                        line_segments.append([p1, p2])
        
        # Update LineCollection with gradient color
        if line_segments:
            colors = plt.cm.hot(np.linspace(0.8, 0.2, len(line_segments)))  # Red gradient
            self.track_collection.set_segments(line_segments)
            self.track_collection.set_colors(colors)
        
        # Update current position marker
        if self.current_position:
            # For sun-sync orbits, make marker slightly larger for visibility over subsolar point
            if getattr(self, 'is_sun_sync_orbit', False):
                self.current_pos_marker.set_markersize(12)  # Larger for sun-sync
                self.current_pos_marker.set_color('orange')  # Different color for distinction
            else:
                self.current_pos_marker.set_markersize(10)  # Normal size
                self.current_pos_marker.set_color('red')     # Normal color
            
            self.current_pos_marker.set_data([self.current_position[1]], 
                                            [self.current_position[0]])
            
            # Draw coverage circle if enabled
            if self.show_coverage.get():
                self._draw_coverage_circle(self.current_position[0], 
                                          self.current_position[1], 
                                          self.current_position[2])
        
        # Refresh canvas
        self.canvas.draw_idle()
    
    def _settings_changed(self):
        """Check if display settings have changed"""
        current = self._get_current_settings()
        if not hasattr(self, '_last_settings'):
            return True
        return current != self._last_settings
    
    def _get_current_settings(self):
        """Get current display settings as tuple"""
        return (
            self.show_grid.get(),
            self.show_daynight.get(),
            self.show_coverage.get(),
            self.use_simulation_time
        )
    
    def _update_day_night_only(self):
        """Update only the day/night terminator without redrawing everything"""
        # Complete cleanup of ALL day/night elements before redrawing
        self._remove_all_day_night_elements()
        
        # Redraw day/night if enabled
        if self.show_daynight.get():
            self._draw_day_night_terminator()
    
    
    def _remove_all_day_night_elements(self):
        """Completely remove ALL day/night related visual elements using stored references"""
        # Remove night patches using stored references
        for patch in self.night_patches:
            try:
                patch.remove()
            except:
                pass
        self.night_patches.clear()
        
        # Remove sun marker
        if self.sun_marker:
            try:
                self.sun_marker.remove()
            except:
                pass
            self.sun_marker = None
        
        # Remove terminator line
        if self.terminator_line:
            try:
                for line in self.terminator_line:
                    if line:
                        line.remove()
            except:
                try:
                    self.terminator_line.remove()
                except:
                    pass
            self.terminator_line = None
        
        # Remove sun label
        if self.sun_label:
            try:
                self.sun_label.remove()
            except:
                pass
            self.sun_label = None
        
        # Remove time label
        if self.time_label:
            try:
                self.time_label.remove()
            except:
                pass
            self.time_label = None
        
        # Also do a cleanup sweep for any missed items
        # Remove any text with time/sun references that we might have missed
        texts_to_remove = []
        for text in self.ax.texts:
            try:
                if any(keyword in text.get_text() for keyword in ['Time:', 'UTC', 'Sun overhead', '(Sim)', '(Real)']):
                    texts_to_remove.append(text)
            except:
                pass
        for text in texts_to_remove:
            try:
                text.remove()
            except:
                pass
    
    def _draw_coverage_circle(self, lat, lon, altitude):
        """Draw satellite coverage circle on ground"""
        # Remove ALL existing coverage circles to prevent accumulation
        circles_to_remove = []
        for patch in self.ax.patches:
            if isinstance(patch, Circle) and hasattr(patch, '_is_coverage_circle'):
                circles_to_remove.append(patch)
        
        for circle in circles_to_remove:
            try:
                circle.remove()
            except:
                pass
        
        # Also clean the stored reference
        self.coverage_circle = None
        
        # Remove old coverage text
        if hasattr(self, 'coverage_text') and self.coverage_text:
            self.coverage_text.remove()
            self.coverage_text = None
        
        # Also clean up any stray coverage texts that might have accumulated
        texts_to_remove = []
        for text in self.ax.texts:
            if 'Coverage:' in text.get_text():
                texts_to_remove.append(text)
        for text in texts_to_remove:
            text.remove()
        
        # Calculate coverage radius
        if altitude > 0:
            # Calculate horizon angle
            horizon_angle = np.arccos(EARTH_RADIUS / (EARTH_RADIUS + altitude))
            coverage_radius_deg = horizon_angle * RAD_TO_DEG
            
            # Draw circle with Earth curvature consideration
            self.coverage_circle = Circle((lon, lat), coverage_radius_deg, 
                                        fill=False, edgecolor='lime', 
                                        linewidth=2, alpha=0.7, linestyle='--',
                                        zorder=6)
            # Mark it as coverage circle for easy identification
            self.coverage_circle._is_coverage_circle = True
            self.ax.add_patch(self.coverage_circle)
            
            # Add coverage radius text and store reference
            self.coverage_text = self.ax.text(lon, lat + coverage_radius_deg + 2, 
                                             f"Coverage: {coverage_radius_deg:.1f}°",
                                             horizontalalignment='center', fontsize=8,
                                             color='green', weight='bold',
                                             bbox=dict(boxstyle="round,pad=0.2", 
                                                     facecolor="white", alpha=0.7))
    
    def _update_info(self, lat, lon, altitude, time_seconds):
        """Update position information labels"""
        if time_seconds is not None:
            hours = time_seconds / 3600
            days = time_seconds / 86400
            orbits = hours / 1.5  # Approximate for LEO
            
            info1 = (f"Position: Lat: {lat:7.3f}°  |  Lon: {lon:8.3f}°  |  "
                    f"Alt: {altitude:7.2f} km")
            info2 = (f"Mission Time: {hours:.2f} hrs ({days:.3f} days)  |  "
                    f"Orbits: {orbits:.1f}")
        else:
            info1 = "Waiting for satellite data..."
            info2 = ""
        
        self.info_label1.config(text=info1)
        self.info_label2.config(text=info2)
    
    def clear_track(self):
        """Clear the ground track"""
        self.track_points.clear()
        self.track_collection.set_segments([])
        self.current_pos_marker.set_data([], [])
        
        # Clear coverage circle and text
        if self.coverage_circle:
            try:
                self.coverage_circle.remove()
            except:
                try:
                    if self.coverage_circle in self.ax.patches:
                        self.ax.patches.remove(self.coverage_circle)
                except:
                    pass
            self.coverage_circle = None
        if hasattr(self, 'coverage_text') and self.coverage_text:
            try:
                self.coverage_text.remove()
            except:
                pass
            self.coverage_text = None
        
        # Clean up any stray coverage texts
        texts_to_remove = []
        for text in self.ax.texts:
            if 'Coverage:' in text.get_text():
                texts_to_remove.append(text)
        for text in texts_to_remove:
            text.remove()
        
        self._draw_base_map()
    
    def center_view(self):
        """Center view on current position"""
        if self.current_position:
            lat, lon, _, _ = self.current_position
            window_size = 60
            self.ax.set_xlim(lon - window_size, lon + window_size)
            self.ax.set_ylim(max(-90, lat - window_size/2), 
                            min(90, lat + window_size/2))
            self.canvas.draw_idle()
    
    def reset_zoom(self):
        """Reset to full world view"""
        self.ax.set_xlim(-180, 180)
        self.ax.set_ylim(-90, 90)
        self.canvas.draw_idle()
    
    def _update_display(self):
        """Update display based on checkbox selections"""
        # Force a complete redraw of the base map to apply new settings
        self._draw_base_map()
        # Then update the plot with current data
        self._update_plot()
    
    def _update_track_length(self, value):
        """Update maximum track length"""
        new_max = int(float(value))
        self.max_points = new_max
        self.track_length_label.config(text=str(self.max_points))
        
        # Update deque maxlen
        self.track_points = deque(list(self.track_points)[-self.max_points:], 
                                 maxlen=self.max_points)
        self._update_plot()


# Test function
if __name__ == "__main__":
    import math
    import threading
    
    # Create test window
    ground_track = EnhancedGroundTrackWindow()
    
    # Simulate satellite positions in a background thread
    def simulate():
        t = 0
        while True:
            try:
                # Simulate ISS-like orbit
                lat = 51.6 * math.sin(t * 0.1)
                lon = -180 + (t * 5) % 360
                altitude = 420 + 10 * math.sin(t * 0.05)
                
                ground_track.update_position(lat, lon, altitude, t * 60)
                
                t += 1
                time.sleep(0.1)  # Slower updates for testing
            except:
                break
    
    # Start simulation in background
    sim_thread = threading.Thread(target=simulate, daemon=True)
    sim_thread.start()
    
    # Run the main loop
    ground_track.window.mainloop()
