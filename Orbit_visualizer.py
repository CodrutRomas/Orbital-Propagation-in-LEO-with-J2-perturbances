"""
Optimized Real-time orbital animation app with complete perturbations
Performance improvements and crash prevention
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from collections import deque
import gc  # Garbage collector for memory management

from orbital_elements import OrbitalElements
from J2_perturbations import J2Perturbations
from dynamics import OrbitalDynamics
from atmospheric_density import AtmosphericModel
from constants import EARTH_RADIUS, RAD_TO_DEG, DEG_TO_RAD, EARTH_NU
from ground_track import GroundTrackWindow


# ---------- Helpers ----------

def set_axes_equal(ax, max_extent):
    """Set 3D plot axes to equal scale using a symmetric cube around origin."""
    ax.set_xlim3d([-max_extent, max_extent])
    ax.set_ylim3d([-max_extent, max_extent])
    ax.set_zlim3d([-max_extent, max_extent])


def draw_earth_optimized(ax, radius_km=EARTH_RADIUS, quality='low'):
    """Draw Earth with adjustable quality for performance"""
    if quality == 'low':
        # Very low poly for maximum performance
        u = np.linspace(0, 2 * np.pi, 20)  # Reduced from 40
        v = np.linspace(0, np.pi, 10)      # Reduced from 20
        wire_alpha = 0.25
    else:
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 15)
        wire_alpha = 0.35
    
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, alpha=wire_alpha, color='tab:blue', linewidth=0.5)
    
    # Simplified equator ring
    theta = np.linspace(0, 2*np.pi, 100)  # Reduced from 200
    ax.plot(radius_km*np.cos(theta), radius_km*np.sin(theta), 
            np.zeros_like(theta), 'b-', alpha=0.5, linewidth=1.5)


def generate_orbit_curve_optimized(orbit: OrbitalElements, num_points: int = 90) -> np.ndarray:
    """Generate orbit curve with optimized number of points"""
    # Use vectorized operations for better performance
    M_values = np.linspace(0, 2*np.pi, num_points)
    pts = np.zeros((num_points, 3))
    
    # Batch process points for better cache usage
    for k, M in enumerate(M_values):
        tmp = OrbitalElements(
            a=orbit.a,
            e=orbit.e,
            i=orbit.i * RAD_TO_DEG,
            raan=orbit.raan * RAD_TO_DEG,
            perigee=orbit.perigee * RAD_TO_DEG,
            mean_anomaly=M * RAD_TO_DEG,
        )
        pos, _ = tmp.to_cartesian()
        pts[k, :] = pos
    return pts


# ---------- Smooth positioner (J2-aware mean anomaly progression) ----------

class SmoothPositioner:
    def __init__(self, base_orbit: OrbitalElements, j2_enabled: bool = True):
        self.base_orbit = base_orbit
        self.j2_enabled = j2_enabled
        mu = 398600.4418  # km^3/s^2
        self.n_kepler = math.sqrt(mu / (base_orbit.a ** 3))
        self.M0 = base_orbit.mean_anomaly  # radians
        # Cache J2 rates
        self._cached_rates = None
        if j2_enabled:
            try:
                self._cached_rates = J2Perturbations(base_orbit).calculate_secular_rates()
            except:
                pass

    def pos_vel_at(self, t_seconds: float, angles_orbit: OrbitalElements):
        # Use cached J2 rates for better performance
        if self.j2_enabled and self._cached_rates:
            n = self._cached_rates['mean_anomaly_dot']
        else:
            n = self.n_kepler

        M = (self.M0 + n * t_seconds) % (2 * np.pi)
        smooth_orbit = OrbitalElements(
            a=angles_orbit.a,
            e=angles_orbit.e,
            i=angles_orbit.i * RAD_TO_DEG,
            raan=angles_orbit.raan * RAD_TO_DEG,
            perigee=angles_orbit.perigee * RAD_TO_DEG,
            mean_anomaly=M * RAD_TO_DEG,
        )
        return smooth_orbit.to_cartesian()


# ---------- Optimized Main App ----------

class OptimizedOrbitAnimationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Orbital Propagation Animation")
        self.root.geometry("1920x1200")  # Increased height for larger info panel

        # Performance mode flag
        self.performance_mode = False  # Start with performance mode OFF for better visual quality
        
        # Simulation state
        self.current_orbit: OrbitalElements | None = None
        self.initial_orbit: OrbitalElements | None = None
        self.initial_orbit_curve_pts: np.ndarray | None = None
        self.initial_angles_deg: dict | None = None
        self.j2_enabled = True
        self.smooth_enabled = True
        self.current_time = 0.0  # seconds since start
        self.time_step = 120.0   # seconds per frame at 1.0x
        self.speed_multiplier = 1.0
        self.j2_max_step = 600.0  # s, max J2 substep for stability
        self.playing = True
        
        # Perturbations state - ALL ENABLED BY DEFAULT
        self.drag_enabled = True   # Enabled by default
        self.srp_enabled = True    # Enabled by default
        self.third_body_enabled = True  # Enabled by default
        self.cartesian_state = None
        self.use_numerical_propagation = False  # Start with analytical
        self.dynamics = None

        # Optimized trails using deque with temporal gradient
        self.max_trail_points = 1000  # Reduced default for performance
        self.trail_points = deque(maxlen=self.max_trail_points)
        self.trail_timestamps = deque(maxlen=self.max_trail_points)  # For gradient coloring
        self.update_orbit_curve_every = 8  # Less frequent updates
        self._orbit_curve_tick = 0
        self.orbit_curve_resolution = 90  # Lower default resolution

        # Reference (no-J2) trail - disabled by default
        self.noj2_trail_points = deque(maxlen=600)  # Smaller trail
        self.noj2_trail_max = 600

        # Smooth positioner
        self.smoother: SmoothPositioner | None = None
        self.noj2_smoother: SmoothPositioner | None = None

        # Precomputed J2 rates
        self.j2_rates = None
        
        # Ground track window
        self.ground_track_window = None
        
        # Animation optimization
        self.animation_interval = 100  # Increased from 60ms for better performance
        self._skip_frames = 0  # Frame skipping for high speeds
        
        # Memory management
        self._gc_counter = 0
        self._gc_interval = 100  # Run garbage collection every N frames

        # Root layout frames
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Create a canvas and scrollbar for the left panel
        left_container = ttk.Frame(main, width=580)  # Increased width
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(8, 4), pady=8)
        left_container.pack_propagate(False)
        
        # Canvas for scrollable content
        left_canvas = tk.Canvas(left_container, width=560)  # Increased width
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure canvas
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        # Frame inside canvas for actual content
        left = ttk.Frame(left_canvas)
        left_canvas_window = left_canvas.create_window((0, 0), window=left, anchor="nw")
        
        # Update scroll region when frame changes size
        def configure_scroll_region(event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        
        left.bind('<Configure>', configure_scroll_region)
        
        # Enable mouse wheel scrolling for left panel only
        def on_left_mousewheel(event):
            # Check if we're over the info text widget first
            widget = self.root.winfo_containing(event.x_root, event.y_root)
            if widget == self.info_text:
                return  # Let the info text handle its own scrolling
            
            # Otherwise check if mouse is over the left panel area
            x = self.root.winfo_pointerx() - self.root.winfo_rootx()
            if x < 580:  # Only scroll if mouse is over left panel (updated width)
                left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                return "break"  # Prevent event propagation
        
        # Store reference for later use
        self.on_left_mousewheel = on_left_mousewheel
        self.left_canvas = left_canvas
        
        # Bind scroll to root initially (will be rebound after info_text is created)
        self.root.bind_all("<MouseWheel>", on_left_mousewheel)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 8), pady=8)

        # Left: inputs + info
        self._build_inputs(left)
        self._build_info_panel(left)
        self._build_controls(left)

        # Keyboard shortcuts
        self.root.bind('<space>', lambda e: self._toggle_play())
        self.root.bind('r', lambda e: self._reset())
        self.root.bind('R', lambda e: self._reset())
        self.root.bind('s', lambda e: self._screenshot())
        self.root.bind('S', lambda e: self._screenshot())
        self.root.bind('p', lambda e: self._toggle_performance_mode())
        self.root.bind('P', lambda e: self._toggle_performance_mode())

        # Right: 3D plot with optimized DPI
        dpi = 80 if self.performance_mode else 100
        self.fig = plt.Figure(figsize=(11, 9), dpi=dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Disable auto features for performance
        self.ax.set_autoscale_on(False)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar for zoom/pan
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

        # Plot artists - using different colors for better visibility
        self.sat_scatter = self.ax.scatter([], [], [], c='yellow', s=60, marker='o', 
                                          edgecolors='black', linewidths=1, label='Satellite')
        self.trail_line, = self.ax.plot([], [], [], color='orange', alpha=0.85, linewidth=2.0, label='Trail')
        self.full_orbit_line, = self.ax.plot([], [], [], color='lime', linestyle='--', 
                                             alpha=0.7, linewidth=1.5, label='Orbit (current)')
        self.initial_orbit_line, = self.ax.plot([], [], [], color='k', linestyle='-', 
                                                alpha=0.9, linewidth=2.0, label='Orbit (initial)')
        
        # Perigee/Apogee markers - re-enabled with better colors
        self.current_peri_marker = self.ax.scatter([], [], [], marker='v', s=40, c='red', 
                                                   edgecolors='darkred', linewidths=1, alpha=0.9, label='Perigee (current)')
        self.current_apo_marker = self.ax.scatter([], [], [], marker='^', s=40, c='blue', 
                                                  edgecolors='darkblue', linewidths=1, alpha=0.9, label='Apogee (current)')
        self.initial_peri_marker = self.ax.scatter([], [], [], marker='v', s=36, c='k', 
                                                   alpha=0.95, label='Perigee (initial)')
        self.initial_apo_marker = self.ax.scatter([], [], [], marker='^', s=36, c='k', 
                                                  alpha=0.95, label='Apogee (initial)')
        
        # No-J2 reference - create but don't show by default
        self.noj2_scatter = self.ax.scatter([], [], [], c='lightgreen', s=40, 
                                           marker='s', edgecolors='green', linewidths=1, alpha=0.8, label='Ref. sat (no J2)')
        self.noj2_trail_line, = self.ax.plot([], [], [], color='lightgreen', 
                                            alpha=0.6, linewidth=1.5, linestyle=':', label='Ref trail (no J2)')

        # Init with ISS
        self._set_defaults()
        self._update_orbit()
        self._draw_static()
        
        # Initialize perturbations after UI is set up
        self._initialize_perturbations_silently()

        # Animation with optimized interval
        self.animation = FuncAnimation(self.fig, self._animate_optimized, 
                                     interval=self.animation_interval, 
                                     blit=False, cache_frame_data=False)

    # ----- UI builders with performance defaults -----

    def _build_inputs(self, parent: ttk.Frame):
        box = ttk.LabelFrame(parent, text="Orbital Elements", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))

        self.entries: dict[str, ttk.Entry] = {}
        
        # Create two-column layout for orbital elements
        columns_frame = ttk.Frame(box)
        columns_frame.pack(fill=tk.X, pady=2)
        
        # Left column
        left_col = ttk.Frame(columns_frame)
        left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Right column  
        right_col = ttk.Frame(columns_frame)
        right_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Left column fields: a, e, i
        left_fields = [
            ('a', 'a (km)'),
            ('e', 'e (0-1)'),
            ('i', 'i (deg)')
        ]
        
        # Right column fields: RAAN, ω, M
        right_fields = [
            ('raan', 'RAAN (deg)'),
            ('perigee', 'ω (deg)'),
            ('mean_anomaly', 'M (deg)')
        ]
        
        # Create left column entries
        for key, label in left_fields:
            row = ttk.Frame(left_col)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=11).pack(side=tk.LEFT)
            ent = ttk.Entry(row, width=12)
            ent.pack(side=tk.LEFT)
            self.entries[key] = ent
        
        # Create right column entries
        for key, label in right_fields:
            row = ttk.Frame(right_col)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=11).pack(side=tk.LEFT)
            ent = ttk.Entry(row, width=12)
            ent.pack(side=tk.LEFT)
            self.entries[key] = ent

        # Options - compact layout
        opt_frame = ttk.Frame(box)
        opt_frame.pack(fill=tk.X, pady=(4, 4))
        
        # First compact row - main options
        opt1 = ttk.Frame(opt_frame)
        opt1.pack(fill=tk.X, pady=(0, 2))
        self.var_j2 = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt1, text="J2", variable=self.var_j2, 
                       command=self._toggle_j2).pack(side=tk.LEFT)
        self.var_smooth = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt1, text="Smooth", variable=self.var_smooth, 
                       command=self._toggle_smooth).pack(side=tk.LEFT, padx=(10, 0))
        self.var_lock_axes = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt1, text="Lock Axes", variable=self.var_lock_axes).pack(side=tk.LEFT, padx=(10, 0))
        self.var_performance = tk.BooleanVar(value=False)  # Performance mode off by default
        ttk.Checkbutton(opt1, text="Performance", variable=self.var_performance, 
                       command=self._toggle_performance_mode).pack(side=tk.LEFT, padx=(10, 0))
        
        # Second compact row - display options
        opt2 = ttk.Frame(opt_frame)
        opt2.pack(fill=tk.X, pady=(0, 0))
        self.var_full_orbit = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt2, text="Current Orbit", variable=self.var_full_orbit).pack(side=tk.LEFT)
        self.var_initial_orbit = tk.BooleanVar(value=True)  # ON by default
        ttk.Checkbutton(opt2, text="Initial Orbit", variable=self.var_initial_orbit, 
                       command=self._refresh_initial_orbit_line).pack(side=tk.LEFT, padx=(10, 0))

        # Compact buttons layout
        btns = ttk.Frame(box)
        btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Update", command=self._update_orbit).pack(side=tk.LEFT)
        ttk.Button(btns, text="ISS", width=6, command=self._preset_iss).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Polar", width=6, command=self._preset_polar).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Sun-sync", width=8, command=self._preset_sunsync).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="GEO", width=6, command=self._preset_geo).pack(side=tk.LEFT, padx=2)
        ground_track_btn = ttk.Button(btns, text="🌍 Track", command=self._toggle_ground_track)
        ground_track_btn.pack(side=tk.LEFT, padx=(8, 0))

        # Perturbations Section
        pert_box = ttk.LabelFrame(parent, text="Perturbations", padding=8)
        pert_box.pack(fill=tk.X, pady=(8, 8))

        # Compact perturbation checkboxes
        pert_checks = ttk.Frame(pert_box)
        pert_checks.pack(fill=tk.X, pady=(0, 4))
        
        self.var_drag = tk.BooleanVar(value=True)  # ON by default
        ttk.Checkbutton(pert_checks, text="Drag", variable=self.var_drag, 
                       command=self._toggle_perturbations).pack(side=tk.LEFT)
        
        self.var_srp = tk.BooleanVar(value=True)  # ON by default
        ttk.Checkbutton(pert_checks, text="SRP", variable=self.var_srp, 
                       command=self._toggle_perturbations).pack(side=tk.LEFT, padx=(10, 0))
        
        self.var_third_body = tk.BooleanVar(value=True)  # ON by default
        ttk.Checkbutton(pert_checks, text="Sun/Moon Gravity", variable=self.var_third_body, 
                       command=self._toggle_perturbations).pack(side=tk.LEFT, padx=(10, 0))

        # Satellite parameters - compact two-column layout
        sat_params = ttk.LabelFrame(pert_box, text="Satellite Parameters", padding=6)
        sat_params.pack(fill=tk.X, pady=(4, 0))
        
        # Two columns for satellite parameters
        sat_cols = ttk.Frame(sat_params)
        sat_cols.pack(fill=tk.X)
        
        sat_left = ttk.Frame(sat_cols)
        sat_left.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        
        sat_right = ttk.Frame(sat_cols)
        sat_right.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))
        
        self.sat_entries: dict[str, ttk.Entry] = {}
        
        # Left column: Mass, Area
        left_fields = [('mass', 'Mass (kg)', '400.0'), ('area', 'Area (m²)', '15.0')]
        for key, label, default in left_fields:
            row = ttk.Frame(sat_left)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            ent = ttk.Entry(row, width=8)
            ent.pack(side=tk.LEFT)
            ent.insert(0, default)
            self.sat_entries[key] = ent
        
        # Right column: Cd, Cr
        right_fields = [('cd', 'Drag Coef', '2.2'), ('cr', 'Refl Coef', '1.3')]
        for key, label, default in right_fields:
            row = ttk.Frame(sat_right)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=12).pack(side=tk.LEFT)
            ent = ttk.Entry(row, width=8)
            ent.pack(side=tk.LEFT)
            ent.insert(0, default)
            self.sat_entries[key] = ent

    def _build_info_panel(self, parent: ttk.Frame):
        box = ttk.LabelFrame(parent, text="Orbit Info", padding=6)
        box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        # Create frame for text widget and scrollbar
        text_frame = ttk.Frame(box)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        # Much larger text widget to show all info without scrolling
        self.info_text = tk.Text(text_frame, height=35, width=65, 
                                font=("Courier New", 9), wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar for info text
        info_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", 
                                      command=self.info_text.yview)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        # Make the text widget read-only
        self.info_text.bind("<Key>", lambda e: "break")
        
        # Enable mouse wheel scrolling specifically for info text
        def on_info_mousewheel(event):
            self.info_text.yview_scroll(int(-1*(event.delta/120)), "units")
            return "break"  # Stop propagation
        
        # Bind mouse wheel to info text with priority
        self.info_text.bind("<MouseWheel>", on_info_mousewheel)
        
        # Also bind Enter/Leave events to manage scrolling focus
        def on_info_enter(event):
            # When mouse enters info text, bind scrolling to it
            self.info_text.bind("<MouseWheel>", on_info_mousewheel)
        
        def on_info_leave(event):
            # When mouse leaves, check position for main panel scrolling
            pass  # Main panel scrolling handled by global binding
        
        self.info_text.bind("<Enter>", on_info_enter)
        self.info_text.bind("<Leave>", on_info_leave)
        
        # Re-bind the main scrolling function now that info_text exists
        def on_mousewheel_global(event):
            # Get the widget under the mouse
            x, y = self.root.winfo_pointerx(), self.root.winfo_pointery()
            widget = self.root.winfo_containing(x, y)
            
            # If we're over the info text, let it handle scrolling
            if widget == self.info_text:
                self.info_text.yview_scroll(int(-1*(event.delta/120)), "units")
                return "break"
            
            # Otherwise check if mouse is over the left panel
            x_rel = x - self.root.winfo_rootx()
            if x_rel < 580 and hasattr(self, 'left_canvas'):  # Updated for new width
                self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                return "break"
        
        # Replace the global binding
        self.root.unbind_all("<MouseWheel>")
        self.root.bind_all("<MouseWheel>", on_mousewheel_global)

    def _build_controls(self, parent: ttk.Frame):
        box = ttk.LabelFrame(parent, text="Controls", padding=8)
        box.pack(fill=tk.X)

        # Play / Reset / Screenshot
        top = ttk.Frame(box)
        top.pack(fill=tk.X)
        self.btn_play = ttk.Button(top, text="Pause", command=self._toggle_play)
        self.btn_play.pack(side=tk.LEFT)
        ttk.Button(top, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Reset View", command=self._reset_view).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Screenshot", command=self._screenshot).pack(side=tk.LEFT, padx=4)
        
        # Second row for additional buttons
        top2 = ttk.Frame(box)
        top2.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(top2, text="Clear Memory", command=self._clear_memory).pack(side=tk.LEFT)

        # Speed control - balanced range with optimized sampling
        spd = ttk.Frame(box)
        spd.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(spd, text="Speed").pack(side=tk.LEFT)
        self.var_speed = tk.DoubleVar(value=1.0)
        # Limited to 25x for optimal stability
        self.scale_speed = ttk.Scale(spd, from_=0.1, to=25.0, orient=tk.HORIZONTAL, 
                                    variable=self.var_speed, command=self._on_speed)
        self.scale_speed.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_speed = ttk.Label(spd, text="1.0x", width=8)
        self.lbl_speed.pack(side=tk.LEFT)
        
        # Speed preset buttons - optimized values
        speed_presets = ttk.Frame(box)
        speed_presets.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(speed_presets, text="Quick:").pack(side=tk.LEFT)
        ttk.Button(speed_presets, text="0.5x", width=5, 
                  command=lambda: self._set_speed(0.5)).pack(side=tk.LEFT, padx=2)
        ttk.Button(speed_presets, text="1x", width=5, 
                  command=lambda: self._set_speed(1.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(speed_presets, text="2x", width=5, 
                  command=lambda: self._set_speed(2.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(speed_presets, text="5x", width=5, 
                  command=lambda: self._set_speed(5.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(speed_presets, text="10x", width=5, 
                  command=lambda: self._set_speed(10.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(speed_presets, text="25x", width=5, 
                  command=lambda: self._set_speed(25.0)).pack(side=tk.LEFT, padx=2)

        # Trail length - optimized defaults
        trl = ttk.Frame(box)
        trl.pack(fill=tk.X)
        ttk.Label(trl, text="Trail length").pack(side=tk.LEFT)
        self.var_trail = tk.IntVar(value=self.max_trail_points)
        self.scale_trail = ttk.Scale(trl, from_=100, to=2000, orient=tk.HORIZONTAL, 
                                    command=self._on_trail)
        self.scale_trail.set(self.max_trail_points)
        self.scale_trail.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_trail = ttk.Label(trl, text=str(self.max_trail_points))
        self.lbl_trail.pack(side=tk.LEFT)

        # Orbit curve resolution - optimized defaults
        crv = ttk.Frame(box)
        crv.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(crv, text="Orbit curve res").pack(side=tk.LEFT)
        self.var_curve_res = tk.IntVar(value=self.orbit_curve_resolution)
        self.scale_curve = ttk.Scale(crv, from_=60, to=180, orient=tk.HORIZONTAL, 
                                    command=self._on_curve_res)
        self.scale_curve.set(self.orbit_curve_resolution)
        self.scale_curve.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_curve = ttk.Label(crv, text=str(self.orbit_curve_resolution))
        self.lbl_curve.pack(side=tk.LEFT)
        
        # Sampling Rate Multiplier for High-Speed Accuracy
        smp = ttk.Frame(box)
        smp.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(smp, text="Sampling quality", width=12).pack(side=tk.LEFT)
        self.var_sampling = tk.DoubleVar(value=1.0)
        self.scale_sampling = ttk.Scale(smp, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                                       variable=self.var_sampling, command=self._on_sampling)
        self.scale_sampling.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_sampling = ttk.Label(smp, text="1.0x", width=6)
        self.lbl_sampling.pack(side=tk.LEFT)

        # Time label
        tim = ttk.Frame(box)
        tim.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(tim, text="Sim time:").pack(side=tk.LEFT)
        self.lbl_time = ttk.Label(tim, text="0.000 d", font=("Courier New", 11))
        self.lbl_time.pack(side=tk.LEFT, padx=8)
        
        # Performance indicator - sync with actual state
        perf_text = "[Performance Mode ON]" if self.performance_mode else "[Performance Mode OFF]"
        perf_color = "green" if self.performance_mode else "gray"
        self.lbl_performance = ttk.Label(tim, text=perf_text, 
                                        font=("Courier New", 9), foreground=perf_color)
        self.lbl_performance.pack(side=tk.RIGHT, padx=8)

    # ----- Presets -----

    def _set_defaults(self):
        # Use ISS preset as default with realistic current orbital elements
        vals = {
            'a': '6793',        # ISS altitude
            'e': '0.0003',      # Low eccentricity
            'i': '51.6',        # ISS inclination
            'raan': '285.4',    # ISS RAAN changes ~4°/day due to J2
            'perigee': '120.8', # ISS argument of perigee varies
            'mean_anomaly': '45.2',  # Current position on orbit
        }
        for k, v in vals.items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, v)

    def _preset_iss(self):
        # ISS with realistic current orbital elements (RAAN precesses ~4°/day)
        vals = {'a': '6793', 'e': '0.0003', 'i': '51.6', 'raan': '285.4', 
                'perigee': '120.8', 'mean_anomaly': '45.2'}
        self._apply_values(vals)

    def _preset_polar(self):
        # Polar orbit - sun-synchronous at different altitude (frozen orbit)
        vals = {'a': '6871', 'e': '0.001', 'i': '90.0', 'raan': '98.7', 
                'perigee': '270.0', 'mean_anomaly': '0.0'}
        self._apply_values(vals)

    def _preset_sunsync(self):
        # Sun-synchronous: RAAN aligned for 10:30 LTAN (Local Time Ascending Node)
        vals = {'a': '7078', 'e': '0.001', 'i': '98.7', 'raan': '157.5', 
                'perigee': '90.0', 'mean_anomaly': '270.0'}
        self._apply_values(vals)

    def _preset_geo(self):
        # GEO positioned over 75°E longitude (realistic operational slot)
        vals = {'a': '42164', 'e': '0.0001', 'i': '0.0', 'raan': '75.0', 
                'perigee': '0.0', 'mean_anomaly': '0.0'}
        self._apply_values(vals)

    def _apply_values(self, values: dict[str, str]):
        for k, v in values.items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, v)
        self._update_orbit()

    # ----- Event handlers with optimizations -----
    
    def _toggle_performance_mode(self):
        """Toggle performance mode on/off"""
        self.performance_mode = self.var_performance.get()
        
        if self.performance_mode:
            # Enable performance optimizations
            self.animation_interval = 100  # Slower updates
            self.update_orbit_curve_every = 8
            self.orbit_curve_resolution = min(90, self.orbit_curve_resolution)
            self.max_trail_points = min(1000, self.max_trail_points)
            self.trail_points = deque(list(self.trail_points)[-self.max_trail_points:], 
                                     maxlen=self.max_trail_points)
            self.trail_timestamps = deque(list(self.trail_timestamps)[-self.max_trail_points:], 
                                         maxlen=self.max_trail_points)
            self.lbl_performance.config(text="[Performance Mode ON]", foreground="green")
            
            # Disable expensive features
            self.var_initial_orbit.set(False)
            self._refresh_initial_orbit_line()
        else:
            # Disable performance optimizations
            self.animation_interval = 60
            self.update_orbit_curve_every = 4
            self.lbl_performance.config(text="[Performance Mode OFF]", foreground="gray")
        
        # Redraw with new settings
        self._draw_static()
        self.canvas.draw_idle()
        
        # Restart animation with new interval
        if hasattr(self, 'animation'):
            self.animation.event_source.stop()
            self.animation = FuncAnimation(self.fig, self._animate_optimized, 
                                         interval=self.animation_interval, 
                                         blit=False, cache_frame_data=False)
            if self.playing:
                self.animation.event_source.start()
    
    def _clear_memory(self):
        """Manually clear memory and run garbage collection"""
        # Clear trails and timestamps
        if len(self.trail_points) > 100:
            recent_points = list(self.trail_points)[-100:]
            recent_timestamps = list(self.trail_timestamps)[-100:]
            self.trail_points = deque(recent_points, maxlen=self.max_trail_points)
            self.trail_timestamps = deque(recent_timestamps, maxlen=self.max_trail_points)
        
        if len(self.noj2_trail_points) > 100:
            recent = list(self.noj2_trail_points)[-100:]
            self.noj2_trail_points = deque(recent, maxlen=self.noj2_trail_max)
        
        # Force garbage collection
        gc.collect()
        
        messagebox.showinfo("Memory Cleared", "Memory has been optimized")

    def _toggle_j2(self):
        self.j2_enabled = self.var_j2.get()
        if self.current_orbit and self.smoother:
            self.smoother.j2_enabled = self.j2_enabled

    def _toggle_smooth(self):
        self.smooth_enabled = self.var_smooth.get()
    
    def _initialize_perturbations_silently(self):
        """Initialize perturbations at startup without user dialog"""
        self.drag_enabled = self.var_drag.get()
        self.srp_enabled = self.var_srp.get()
        self.third_body_enabled = self.var_third_body.get()
        
        # Check if any numerical perturbation is enabled
        needs_numerical = self.drag_enabled or self.srp_enabled or self.third_body_enabled
        
        if needs_numerical:
            # Switch to numerical integration mode without asking
            self.use_numerical_propagation = True
            # Initialize or update dynamics model
            self._update_dynamics()
            # Initialize Cartesian state from current orbital elements
            if self.current_orbit is not None:
                r, v = self.current_orbit.to_cartesian()
                self.cartesian_state = np.concatenate([r, v])
        else:
            # Switch back to analytical J2 mode
            self.use_numerical_propagation = False
            self.cartesian_state = None
            self.dynamics = None
    
    def _toggle_perturbations(self):
        """Toggle perturbations and switch between analytical/numerical propagation"""
        self.drag_enabled = self.var_drag.get()
        self.srp_enabled = self.var_srp.get()
        self.third_body_enabled = self.var_third_body.get()
        
        # Check if any numerical perturbation is enabled
        needs_numerical = self.drag_enabled or self.srp_enabled or self.third_body_enabled
        
        if needs_numerical:
            # Warn about performance impact
            if self.performance_mode:
                response = messagebox.askyesno("Performance Warning", 
                    "Enabling perturbations will impact performance.\n"
                    "Do you want to disable Performance Mode?")
                if response:
                    self.var_performance.set(False)
                    self._toggle_performance_mode()
            
            # Switch to numerical integration mode
            self.use_numerical_propagation = True
            # Initialize or update dynamics model
            self._update_dynamics()
            # Initialize Cartesian state from current orbital elements
            if self.current_orbit is not None:
                r, v = self.current_orbit.to_cartesian()
                self.cartesian_state = np.concatenate([r, v])
        else:
            # Switch back to analytical J2 mode
            self.use_numerical_propagation = False
            self.cartesian_state = None
            self.dynamics = None
    
    def _update_dynamics(self):
        """Create or update the dynamics model with current parameters"""
        if not self.use_numerical_propagation:
            return
        
        try:
            # Get satellite parameters
            mass = float(self.sat_entries['mass'].get())
            area = float(self.sat_entries['area'].get())
            cd = float(self.sat_entries['cd'].get())
            cr = float(self.sat_entries['cr'].get())
            
            # Create dynamics model
            self.dynamics = OrbitalDynamics(
                include_drag=self.drag_enabled,
                drag_coefficient=cd,
                satellite_mass=mass,
                cross_sectional_area=area,
                include_srp=self.srp_enabled,
                reflectivity_coefficient=cr,
                include_third_bodies=self.third_body_enabled
            )
                
        except Exception as e:
            print(f"Error updating dynamics: {e}")

    def _toggle_play(self):
        if not hasattr(self, 'animation') or self.animation is None:
            return
        if self.playing:
            try:
                self.animation.event_source.stop()
            except Exception:
                pass
            self.playing = False
            self.btn_play.config(text="Play")
        else:
            try:
                self.animation.event_source.start()
            except Exception:
                pass
            self.playing = True
            self.btn_play.config(text="Pause")

    def _on_speed(self, *_):
        self.speed_multiplier = float(self.var_speed.get())
        if self.speed_multiplier >= 10:
            self.lbl_speed.config(text=f"{self.speed_multiplier:.0f}x")
            # Enable frame skipping for high speeds
            self._skip_frames = int(self.speed_multiplier / 10)
        else:
            self.lbl_speed.config(text=f"{self.speed_multiplier:.1f}x")
            self._skip_frames = 0
    
    def _set_speed(self, value):
        """Set speed to a specific value from preset buttons"""
        self.var_speed.set(value)
        self._on_speed()
    
    def _on_sampling(self, *_):
        """Handle sampling rate changes"""
        sampling = float(self.var_sampling.get())
        self.lbl_sampling.config(text=f"{sampling:.1f}x")

    def _on_trail(self, *_):
        new_max = int(float(self.scale_trail.get()))
        if self.performance_mode:
            new_max = min(new_max, 1000)  # Limit in performance mode
        self.max_trail_points = new_max
        if hasattr(self, 'lbl_trail'):
            self.lbl_trail.config(text=str(self.max_trail_points))
        # Convert to deque with new maxlen for both points and timestamps
        self.trail_points = deque(list(self.trail_points)[-self.max_trail_points:], 
                                 maxlen=self.max_trail_points)
        self.trail_timestamps = deque(list(self.trail_timestamps)[-self.max_trail_points:], 
                                     maxlen=self.max_trail_points)

    def _on_curve_res(self, *_):
        try:
            new_res = int(float(self.scale_curve.get()))
            if self.performance_mode:
                new_res = min(new_res, 90)  # Limit in performance mode
            self.orbit_curve_resolution = new_res
        except Exception:
            self.orbit_curve_resolution = 90
        if hasattr(self, 'lbl_curve'):
            self.lbl_curve.config(text=str(self.orbit_curve_resolution))
        # Recompute initial curve
        if self.initial_orbit is not None:
            try:
                self.initial_orbit_curve_pts = generate_orbit_curve_optimized(
                    self.initial_orbit, num_points=self.orbit_curve_resolution)
            except Exception:
                pass
        # Refresh lines
        self._update_full_orbit_curve()
        self._refresh_initial_orbit_line()
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()

    def _screenshot(self):
        fname = "orbit_animation_optimized.png"
        self.fig.savefig(fname, dpi=150, bbox_inches='tight')  # Lower DPI for faster save
        messagebox.showinfo("Saved", f"Screenshot saved as {fname}")
    
    def _toggle_ground_track(self):
        """Toggle ground track window"""
        if self.ground_track_window is None:
            # Create new optimized ground track window
            self.ground_track_window = GroundTrackWindow(self.root)
            # Handle window close event
            self.ground_track_window.window.protocol("WM_DELETE_WINDOW", 
                                                    self._on_ground_track_close)
        else:
            # Close existing window
            self.ground_track_window.window.destroy()
            self.ground_track_window = None
    
    def _on_ground_track_close(self):
        """Handle ground track window close event"""
        if self.ground_track_window:
            self.ground_track_window.window.destroy()
            self.ground_track_window = None
    
    def _calculate_lat_lon_alt(self, position):
        """Calculate latitude, longitude and altitude from position vector"""
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)
        alt = r - EARTH_RADIUS
        
        # Improved latitude calculation with bounds checking
        z_r_ratio = np.clip(z / r, -1.0, 1.0)  # Prevent numerical errors
        lat = np.arcsin(z_r_ratio) * RAD_TO_DEG
        
        # Improved longitude calculation
        lon = np.arctan2(y, x) * RAD_TO_DEG
        
        # Account for Earth's rotation with better precision
        earth_rotation_rate = 360.0 / 86400.0  # degrees per second
        earth_rotation_correction = (self.current_time * earth_rotation_rate) % 360.0
        lon = lon - earth_rotation_correction
        
        # Special handling for sun-synchronous orbits - align with sun
        if self._is_sun_synchronous_orbit():
            lon = self._adjust_longitude_for_sun_sync(lon)
        
        # Normalize longitude more efficiently
        lon = ((lon + 180.0) % 360.0) - 180.0
        
        return lat, lon, alt
    
    def _is_sun_synchronous_orbit(self):
        """Detect if current orbit is sun-synchronous based on inclination and altitude"""
        if not self.current_orbit:
            return False
        
        # Sun-sync characteristics: inclination ~98-99°, altitude ~700-800km
        inclination_deg = self.current_orbit.i * RAD_TO_DEG
        altitude_km = self.current_orbit.a - EARTH_RADIUS
        
        # Check if orbit matches sun-sync characteristics
        is_retrograde_polar = 90 < inclination_deg < 110  # Retrograde polar
        is_typical_altitude = 600 < altitude_km < 1000     # Typical sun-sync altitude
        
        return is_retrograde_polar and is_typical_altitude
    
    def _adjust_longitude_for_sun_sync(self, original_lon):
        """Adjust longitude to make sun-synchronous satellite follow the subsolar point"""
        from datetime import datetime, timedelta
        import numpy as np
        
        # Calculate current sun position (same calculation as in ground_track_enhanced.py)
        if hasattr(self, 'current_time'):
            # Use simulation time
            now = datetime.utcnow() + timedelta(seconds=self.current_time)
        else:
            now = datetime.utcnow()
        
        # Calculate subsolar point longitude (exact same formula as ground track)
        hours_from_noon = now.hour + now.minute/60.0 + now.second/3600.0 - 12
        sun_lon = -hours_from_noon * 15  # Sun moves 15 degrees per hour westward
        
        # For true sun-synchronous behavior, satellite should be directly at subsolar point
        # This makes it always maintain the same local solar time (noon in this case)
        target_lon = sun_lon
        
        # Smooth transition to avoid jumps in visualization
        lon_diff = target_lon - original_lon
        
        # Handle longitude wrapping
        while lon_diff > 180:
            lon_diff -= 360
        while lon_diff < -180:
            lon_diff += 360
        
        # Apply more aggressive correction (50% per update) for better tracking
        corrected_lon = original_lon + lon_diff * 0.5
        
        return corrected_lon

    def _reset(self):
        self.current_time = 0.0
        self.trail_points.clear()
        self.noj2_trail_points.clear()
        self.lbl_time.config(text="0.000 d")
        # Reset smoother
        if self.current_orbit is not None:
            self.smoother = SmoothPositioner(self.current_orbit, self.j2_enabled)
        if self.initial_orbit is not None:
            self.noj2_smoother = SmoothPositioner(self.initial_orbit, j2_enabled=False)
        # Clear memory
        gc.collect()

    def _reset_view(self):
        max_extent = EARTH_RADIUS * 1.5
        try:
            if self.initial_orbit_curve_pts is not None and self.initial_orbit_curve_pts.size > 0:
                max_extent = max(max_extent, float(np.max(np.abs(self.initial_orbit_curve_pts))))
        except Exception:
            pass
        try:
            if self.current_orbit is not None:
                cur_pts = generate_orbit_curve_optimized(self.current_orbit, num_points=90)
                if cur_pts.size > 0:
                    max_extent = max(max_extent, float(np.max(np.abs(cur_pts))))
        except Exception:
            pass
        set_axes_equal(self.ax, max_extent * 1.12)
        self.canvas.draw_idle()

    # ----- Orbit and drawing -----

    def _read_orbit_from_inputs(self) -> OrbitalElements | None:
        try:
            vals = {k: float(self.entries[k].get()) for k in self.entries}
            return OrbitalElements(**vals)
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return None

    def _update_orbit(self):
        orb = self._read_orbit_from_inputs()
        if orb is None:
            return
        
        self.current_orbit = orb
        self.initial_orbit = OrbitalElements(
            a=orb.a,
            e=orb.e,
            i=orb.i * RAD_TO_DEG,
            raan=orb.raan * RAD_TO_DEG,
            perigee=orb.perigee * RAD_TO_DEG,
            mean_anomaly=orb.mean_anomaly * RAD_TO_DEG,
        )
        self.initial_angles_deg = {
            'raan': self.initial_orbit.raan * RAD_TO_DEG,
            'perigee': self.initial_orbit.perigee * RAD_TO_DEG,
        }
        
        # Precompute initial orbit curve points
        try:
            self.initial_orbit_curve_pts = generate_orbit_curve_optimized(
                self.initial_orbit, num_points=self.orbit_curve_resolution)
        except Exception:
            self.initial_orbit_curve_pts = None
        
        # Precompute J2 rates once
        try:
            self.j2_rates = J2Perturbations(self.initial_orbit).calculate_secular_rates()
        except Exception:
            self.j2_rates = None
        
        self.smoother = SmoothPositioner(orb, self.j2_enabled)
        self.noj2_smoother = SmoothPositioner(self.initial_orbit, j2_enabled=False)
        self.trail_points.clear()
        self.trail_timestamps.clear()
        self.noj2_trail_points.clear()
        self._orbit_curve_tick = 0
        
        # Reset simulation time when orbit changes
        self.current_time = 0.0
        self.lbl_time.config(text="0.000 d")
        
        # Update dynamics if using numerical propagation
        if self.use_numerical_propagation:
            self._update_dynamics()
            # CRITICAL: Reset Cartesian state when orbit changes
            if self.current_orbit is not None:
                try:
                    r, v = self.current_orbit.to_cartesian()
                    self.cartesian_state = np.concatenate([r, v])
                    print(f"Reset Cartesian state for new orbit: {self.current_orbit.a:.1f} km")
                except Exception as e:
                    print(f"Error resetting Cartesian state: {e}")
            
        # Clear and sync ground track when orbit changes
        if self.ground_track_window is not None:
            try:
                # Clear the old track completely
                self.ground_track_window.clear_track()
                # Reset ground track timing
                self.ground_track_window.simulation_time = 0.0
                self.ground_track_window.current_position = None
                self.ground_track_window.track_points.clear()
                print("Ground track cleared and synced with new orbit")
            except Exception as e:
                print(f"Error syncing ground track: {e}")
            if self.current_orbit is not None:
                r, v = self.current_orbit.to_cartesian()
                self.cartesian_state = np.concatenate([r, v])
        
        # Draw and update curves
        self._draw_static()
        self._refresh_initial_orbit_line()
        self._update_full_orbit_curve()
        self.canvas.draw_idle()

    def _draw_static(self):
        self.ax.clear()
        
        # Draw Earth with quality based on performance mode
        quality = 'low' if self.performance_mode else 'high'
        draw_earth_optimized(self.ax, radius_km=EARTH_RADIUS, quality=quality)
        
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.grid(True, alpha=0.15 if self.performance_mode else 0.25)
        self.ax.set_title("Orbital Propagation Animation")

        # Recreate artists after clear()
        self.sat_scatter = self.ax.scatter([], [], [], c='r', s=50, label='Satellite')
        
        # Simple gradient trail using overlapping lines with different alpha
        self.trail_lines = []  # Multiple line objects with different alpha
        # Create several line objects for layered gradient effect
        # Orange gradient - perfect contrast with cyan current orbit
        colors = ['#8B4513', '#FF6347', '#FF8C00', '#FFD700']  # Brown to bright orange to gold
        alphas = [0.5, 0.7, 0.85, 1.0]
        
        for i, (color, alpha) in enumerate(zip(colors, alphas)):
            line, = self.ax.plot([], [], [], color=color, alpha=alpha, 
                               linewidth=2.0 - i*0.3, label='Trail' if i == 0 else '')
            self.trail_lines.append(line)
        # Current Orbit: Dark Green - perfect contrast with orange trail
        self.full_orbit_line, = self.ax.plot([], [], [], color='#228B22', 
                                            linestyle='-', alpha=0.9, 
                                            linewidth=3.0, label='Current Orbit')
        self.initial_orbit_line, = self.ax.plot([], [], [], color='k', 
                                               linestyle='-', alpha=0.9, 
                                               linewidth=2.0, label='Orbit (initial)')
        
        # Perigee/Apogee markers - recreate after clear
        self.current_peri_marker = self.ax.scatter([], [], [], marker='v', s=30, c='tab:orange', 
                                                   alpha=0.9, label='Perigee (current)')
        self.current_apo_marker = self.ax.scatter([], [], [], marker='^', s=30, c='tab:cyan', 
                                                  alpha=0.9, label='Apogee (current)')
        self.initial_peri_marker = self.ax.scatter([], [], [], marker='v', s=36, c='k', 
                                                   alpha=0.95, label='Perigee (initial)')
        self.initial_apo_marker = self.ax.scatter([], [], [], marker='^', s=36, c='k', 
                                                  alpha=0.95, label='Apogee (initial)')
        
        # No-J2 reference (hidden)
        self.noj2_scatter = self.ax.scatter([], [], [], c='tab:green', s=35, 
                                           marker='o', alpha=0.8)
        self.noj2_trail_line, = self.ax.plot([], [], [], color='tab:green', 
                                            alpha=0.5, linewidth=1.2)
        
        # Simple legend
        if not self.performance_mode:
            self.ax.legend(loc='upper right', fontsize=8)

        # Axis extents
        if self.current_orbit and not self.var_lock_axes.get():
            a = self.current_orbit.a
            e = self.current_orbit.e
            r_max = a * (1 + e)
            margin = 800.0
            set_axes_equal(self.ax, max(EARTH_RADIUS * 1.5, r_max + margin))

    def _update_full_orbit_curve(self):
        if not self.current_orbit:
            return
        
        if self.use_numerical_propagation:
            orbit_now = self.current_orbit
        else:
            orbit_now = self._angles_for_time(self.current_time)
            self.current_orbit = orbit_now
        
        pts = generate_orbit_curve_optimized(orbit_now, num_points=self.orbit_curve_resolution)
        if self.var_full_orbit.get():
            self.full_orbit_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        else:
            self.full_orbit_line.set_data_3d([], [], [])
        
        # Update axes
        if not self.var_lock_axes.get():
            max_abs = float(np.max(np.abs(pts))) if pts.size > 0 else EARTH_RADIUS * 2
            if self.initial_orbit_curve_pts is not None and self.initial_orbit_curve_pts.size > 0:
                max_abs = max(max_abs, float(np.max(np.abs(self.initial_orbit_curve_pts))))
            set_axes_equal(self.ax, max(max_abs * 1.1, EARTH_RADIUS * 1.5))
        
        # Update perigee/apogee markers
        self._update_peri_apo_markers()

    def _animate_optimized(self, _frame_idx):
        """Optimized animation function with adaptive sampling"""
        if not self.current_orbit or not self.smoother:
            return tuple(self.trail_lines),
        if not self.playing:
            return tuple(self.trail_lines),
        
        # Periodic garbage collection
        self._gc_counter += 1
        if self._gc_counter >= self._gc_interval:
            gc.collect(0)  # Collect only generation 0 for speed
            self._gc_counter = 0

        # ADAPTIVE SAMPLING based on speed multiplier
        base_dt = self.time_step * self.speed_multiplier
        
        # Get user sampling quality multiplier
        sampling_quality = 1.0
        if hasattr(self, 'var_sampling'):
            try:
                sampling_quality = float(self.var_sampling.get())
            except:
                sampling_quality = 1.0
        
        # Calculate adaptive number of sub-steps for orbit accuracy
        # More conservative algorithm for better performance
        if self.speed_multiplier >= 20.0:
            # High speeds need moderate sub-steps
            base_substeps = max(4, int(self.speed_multiplier / 6))
        elif self.speed_multiplier >= 10.0:
            # Medium-high speeds need some sub-steps
            base_substeps = max(3, int(self.speed_multiplier / 5))
        elif self.speed_multiplier >= 5.0:
            # Medium speeds need few sub-steps
            base_substeps = max(2, int(self.speed_multiplier / 4))
        else:
            # Low speeds use single step
            base_substeps = 1
        
        # Apply sampling quality multiplier (0.5x to 3.0x)
        num_substeps = max(1, int(base_substeps * sampling_quality))
        
        # Limit maximum substeps to prevent performance issues
        num_substeps = min(num_substeps, 15)  # Reduced from 50
        
        # Subdivide the time step for better accuracy
        adaptive_dt = base_dt / num_substeps
        
        # Process multiple sub-steps for smooth orbital motion
        for sub_step in range(num_substeps):
            step_time = self.current_time + (sub_step + 1) * adaptive_dt
        
            # For each sub-step, compute position and update orbit
            if self.use_numerical_propagation and self.dynamics and self.cartesian_state is not None:
                # Numerical integration with adaptive sub-steps
                max_integration_step = 10.0 if self.performance_mode else 5.0
                integration_substeps = max(1, int(np.ceil(adaptive_dt / max_integration_step)))
                integration_dt = adaptive_dt / integration_substeps
                
                try:
                    # Multiple RK4 sub-steps for this animation sub-step
                    temp_state = self.cartesian_state.copy()
                    temp_time = self.current_time
                    
                    for i in range(integration_substeps):
                        temp_state = self.dynamics.rk4_step(
                            temp_state, integration_dt, temp_time
                        )
                        temp_time += integration_dt
                    
                    # Update state after all integration steps
                    if sub_step == num_substeps - 1:  # Only on final sub-step
                        self.cartesian_state = temp_state
                        last_pos = temp_state[:3]
                        vel = temp_state[3:6]
                    else:
                        last_pos = temp_state[:3]  # Use for trail but don't update state yet
                        vel = temp_state[3:6]
                    
                    # Update current orbit from Cartesian state
                    try:
                        if sub_step == num_substeps - 1:  # Only on final sub-step
                            self.current_orbit = OrbitalElements.from_cartesian(last_pos, vel)
                    except Exception as orbit_error:
                        print(f"Orbit conversion error: {orbit_error}")
                        # Fall back to analytical propagation if numerical fails
                        angles_at_t = self._angles_for_time(step_time)
                        if self.smooth_enabled:
                            last_pos, vel = self.smoother.pos_vel_at(step_time, angles_at_t)
                        else:
                            last_pos, vel = angles_at_t.to_cartesian()
                        if sub_step == num_substeps - 1:
                            self.current_orbit = angles_at_t
                        
                except Exception as e:
                    print(f"Integration error: {e}")
                    # Fall back to analytical propagation
                    angles_at_t = self._angles_for_time(step_time)
                    if self.smooth_enabled:
                        last_pos, vel = self.smoother.pos_vel_at(step_time, angles_at_t)
                    else:
                        last_pos, vel = angles_at_t.to_cartesian()
                    if sub_step == num_substeps - 1:
                        self.current_orbit = angles_at_t
                        
            else:
                # Analytical propagation with adaptive sampling
                angles_at_t = self._angles_for_time(step_time)
                if self.smooth_enabled:
                    last_pos, vel = self.smoother.pos_vel_at(step_time, angles_at_t)
                else:
                    last_pos, vel = angles_at_t.to_cartesian()
                
                if sub_step == num_substeps - 1:  # Only on final sub-step
                    self.current_orbit = angles_at_t
            
            # Add to trail (but not every sub-step to avoid overcrowding)
            if sub_step % max(1, num_substeps // 3) == 0 or sub_step == num_substeps - 1:
                self.trail_points.append(last_pos.copy())
                self.trail_timestamps.append(step_time)  # Add timestamp for gradient
        
        # Update time after all sub-steps
        self.current_time += base_dt
        
        # Get final position and velocity for display
        final_time = self.current_time
        
        # No-J2 reference - keep hidden (always disabled)
        self.noj2_trail_line.set_data_3d([], [], [])
        self.noj2_scatter._offsets3d = ([], [], [])

        # Update satellite position display
        if 'last_pos' in locals():
            self.sat_scatter._offsets3d = ([last_pos[0]], [last_pos[1]], [last_pos[2]])
            
            # Update trail display with temporal gradient effect
            if len(self.trail_points) >= 2:
                points = np.array(self.trail_points)
                timestamps = np.array(self.trail_timestamps)
                
                # Calculate different trail segments based on age
                total_points = len(points)
                if total_points > 4:  # Need enough points for gradient
                    # Calculate proportions for each trail layer
                    proportions = [1.0, 0.8, 0.6, 0.4]  # Newest to oldest
                    
                    for i, line in enumerate(self.trail_lines):
                        # Calculate how many points to show for this layer
                        points_to_show = max(2, int(total_points * proportions[i]))
                        start_idx = max(0, total_points - points_to_show)
                        
                        # Get subset of points
                        subset_points = points[start_idx:]
                        
                        # Update line with subset
                        if len(subset_points) >= 2:
                            line.set_data_3d(subset_points[:, 0], 
                                            subset_points[:, 1], 
                                            subset_points[:, 2])
                        else:
                            line.set_data_3d([], [], [])
                else:
                    # Not enough points, show simple trail on first line only
                    if total_points >= 2:
                        self.trail_lines[0].set_data_3d(points[:, 0], points[:, 1], points[:, 2])
                        # Clear other lines
                        for line in self.trail_lines[1:]:
                            line.set_data_3d([], [], [])
                    else:
                        # Clear all lines
                        for line in self.trail_lines:
                            line.set_data_3d([], [], [])
            else:
                # Clear all trail lines when no points
                for line in self.trail_lines:
                    line.set_data_3d([], [], [])
            
            # Update info less frequently for performance
            update_info = (not self.performance_mode) or (self._orbit_curve_tick % 5 == 0)
            if update_info and 'vel' in locals():
                self._update_info_optimized(last_pos, vel)
        
        # Update orbit curve less frequently
        self._orbit_curve_tick = (self._orbit_curve_tick + 1) % self.update_orbit_curve_every
        if self._orbit_curve_tick == 0:
            self._update_full_orbit_curve()
        
        # Update initial orbit line if enabled
        if self.var_initial_orbit.get():
            self._refresh_initial_orbit_line()
            self._update_peri_apo_markers()
        
        # Update time display
        days = self.current_time / 86400.0
        hours = self.current_time / 3600.0
        self.lbl_time.config(text=f"{days:.3f} d ({hours:.1f} h)")
        
        # Update ground track if window is open
        if self.ground_track_window is not None and 'last_pos' in locals():
            try:
                lat, lon, alt = self._calculate_lat_lon_alt(last_pos)
                is_sun_sync = self._is_sun_synchronous_orbit()
                self.ground_track_window.update_position(lat, lon, alt, self.current_time, is_sun_sync)
            except Exception as e:
                print(f"Ground track update error: {e}")
                pass

        return tuple(self.trail_lines),

    def _update_info_optimized(self, position: np.ndarray, velocity: np.ndarray):
        """Complete info update with J2 analysis"""
        if not self.current_orbit:
            return
        
        try:
            r_mag = float(np.linalg.norm(position))
            v_mag = float(np.linalg.norm(velocity))
            altitude = r_mag - EARTH_RADIUS
            
            # Calculate J2 effects
            j2_info = {}
            if self.j2_enabled:
                try:
                    j2_info = J2Perturbations(self.current_orbit).get_analysis_report()
                except Exception:
                    j2_info = {}
            
            def fmt(key, default="-"):
                return j2_info.get(key, default)
            
            # Calculate deltas from initial orbit
            dOmega_str = dOmega_rate_str = "-"
            domega_str = domega_rate_str = "-"
            if self.initial_angles_deg is not None:
                cur_Omega = self.current_orbit.raan * RAD_TO_DEG
                cur_omega = self.current_orbit.perigee * RAD_TO_DEG
                dOmega_val = cur_Omega - self.initial_angles_deg.get('raan', 0.0)
                domega_val = cur_omega - self.initial_angles_deg.get('perigee', 0.0)
                dOmega_str = f"{dOmega_val:.4f}"
                domega_str = f"{domega_val:.4f}"
                if j2_info:
                    dOmega_rate_str = f"{fmt('raan_precession_deg_day'):.5f}"
                    domega_rate_str = f"{fmt('perigee_precession_deg_day'):.5f}"
            
            # Apogee/Perigee calculations
            rp = self.current_orbit.a * (1.0 - self.current_orbit.e)
            ra = self.current_orbit.a * (1.0 + self.current_orbit.e)
            perigee_alt = rp - EARTH_RADIUS
            apogee_alt = ra - EARTH_RADIUS
            
            # Build complete info string
            info = (
                f"Current time: {self.current_time/86400.0:.3f} d ({self.current_time/3600.0:.1f} h)\n\n"
                f"Position (km):\n  X={position[0]:.2f}  Y={position[1]:.2f}  Z={position[2]:.2f}\n"
                f"Velocity (km/s):\n  Vx={velocity[0]:.3f}  Vy={velocity[1]:.3f}  Vz={velocity[2]:.3f}\n"
                f"Speed: {v_mag:.3f} km/s\n"
                f"Altitude: {altitude:.2f} km\n"
                f"Perigee: r={rp:.2f} km, h={perigee_alt:.2f} km\n"
                f"Apogee:  r={ra:.2f} km, h={apogee_alt:.2f} km\n\n"
                f"Orbit elements (deg):\n"
                f"  i={self.current_orbit.i*RAD_TO_DEG:.3f}  Ω={self.current_orbit.raan*RAD_TO_DEG:.3f}  "
                f"ω={self.current_orbit.perigee*RAD_TO_DEG:.3f}\n"
                f"  e={self.current_orbit.e:.5f}  a={self.current_orbit.a:.2f} km\n"
                f"  ΔΩ={dOmega_str} deg (rate: {dOmega_rate_str} deg/day)\n"
                f"  Δω={domega_str} deg (rate: {domega_rate_str} deg/day)\n\n"
            )
            
            # Add J2 effects if enabled
            if j2_info:
                # Format cycles properly
                raan_cycle = fmt('raan_period_days')
                omega_cycle = fmt('perigee_period_days')
                
                # Format large numbers properly
                if isinstance(raan_cycle, (int, float)) and raan_cycle != float('inf'):
                    if raan_cycle > 10000:
                        raan_cycle_str = f"{raan_cycle:.0f}"
                    else:
                        raan_cycle_str = f"{raan_cycle:.1f}"
                else:
                    raan_cycle_str = "∞" if raan_cycle == float('inf') else str(raan_cycle)
                
                if isinstance(omega_cycle, (int, float)) and omega_cycle != float('inf'):
                    if omega_cycle > 10000:
                        omega_cycle_str = f"{omega_cycle:.0f}"
                    else:
                        omega_cycle_str = f"{omega_cycle:.1f}"
                else:
                    omega_cycle_str = "∞" if omega_cycle == float('inf') else str(omega_cycle)
                
                info += (
                    "J2 effects:\n"
                    f"  RAAN rate: {fmt('raan_precession_deg_day'):.5f} deg/day\n"
                    f"  ω rate: {fmt('perigee_precession_deg_day'):.5f} deg/day\n"
                    f"  Period (Kepler): {fmt('kepler_period_hours'):.4f} h\n"
                    f"  Period (J2): {fmt('j2_modified_period_hours'):.4f} h\n"
                    f"  J2 correction: {fmt('j2_correction_percent'):.4f} %\n"
                    f"  RAAN cycle: {raan_cycle_str} days\n"
                    f"  ω cycle: {omega_cycle_str} days\n"
                    f"  Orbit type: {fmt('orbit_type')}\n"
                    f"  J2 impact: {fmt('j2_significance')}\n"
                )
            
            # Add performance mode and adaptive sampling indicators
            if self.performance_mode:
                info += "\n[Performance Mode Active]"
            
            # Add adaptive sampling info for very high speeds
            if hasattr(self, 'speed_multiplier') and self.speed_multiplier >= 15.0:
                info += "\n[High-Speed Trail Smoothing Active]"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
        except Exception as e:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, f"Info error: {e}")

    def _refresh_initial_orbit_line(self):
        """Update initial orbit line visibility"""
        try:
            if self.initial_orbit_curve_pts is not None and self.var_initial_orbit.get():
                pts = self.initial_orbit_curve_pts
                self.initial_orbit_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
            else:
                self.initial_orbit_line.set_data_3d([], [], [])
            self.canvas.draw_idle()
        except Exception:
            pass

    def _angles_for_time(self, t_seconds: float) -> OrbitalElements:
        """Build orbit with J2 progression at time t"""
        if self.initial_orbit is None:
            return self.current_orbit if self.current_orbit else OrbitalElements(6793,0.0003,51.6,0,0,0)
        
        if self.j2_enabled and self.j2_rates is not None:
            raan = (self.initial_orbit.raan + self.j2_rates['raan_dot'] * t_seconds) % (2*np.pi)
            perigee = (self.initial_orbit.perigee + self.j2_rates['perigee_dot'] * t_seconds) % (2*np.pi)
        else:
            raan = self.initial_orbit.raan
            perigee = self.initial_orbit.perigee
        
        return OrbitalElements(
            a=self.initial_orbit.a,
            e=self.initial_orbit.e,
            i=self.initial_orbit.i * RAD_TO_DEG,
            raan=raan * RAD_TO_DEG,
            perigee=perigee * RAD_TO_DEG,
            mean_anomaly=0.0,
        )
    
    def _update_peri_apo_markers(self):
        """Update perigee/apogee markers positions"""
        try:
            # Current orbit markers
            if self.var_full_orbit.get() and self.current_orbit:
                # For numerical propagation, use the current orbit directly
                if self.use_numerical_propagation:
                    cur = self.current_orbit
                else:
                    cur = self._angles_for_time(self.current_time)
                
                # Perigee = M=0 deg, Apogee = M=180 deg
                cur_peri = OrbitalElements(
                    a=cur.a,
                    e=cur.e,
                    i=cur.i * RAD_TO_DEG,
                    raan=cur.raan * RAD_TO_DEG,
                    perigee=cur.perigee * RAD_TO_DEG,
                    mean_anomaly=0.0,
                )
                ppos, _ = cur_peri.to_cartesian()
                
                cur_apo = OrbitalElements(
                    a=cur.a,
                    e=cur.e,
                    i=cur.i * RAD_TO_DEG,
                    raan=cur.raan * RAD_TO_DEG,
                    perigee=cur.perigee * RAD_TO_DEG,
                    mean_anomaly=180.0,
                )
                apos, _ = cur_apo.to_cartesian()
                
                self.current_peri_marker._offsets3d = ([ppos[0]], [ppos[1]], [ppos[2]])
                self.current_apo_marker._offsets3d = ([apos[0]], [apos[1]], [apos[2]])
            else:
                self.current_peri_marker._offsets3d = ([], [], [])
                self.current_apo_marker._offsets3d = ([], [], [])
            
            # Initial orbit markers
            if self.initial_orbit is not None and self.var_initial_orbit.get():
                ini_peri = OrbitalElements(
                    a=self.initial_orbit.a,
                    e=self.initial_orbit.e,
                    i=self.initial_orbit.i * RAD_TO_DEG,
                    raan=self.initial_orbit.raan * RAD_TO_DEG,
                    perigee=self.initial_orbit.perigee * RAD_TO_DEG,
                    mean_anomaly=0.0,
                )
                ipos, _ = ini_peri.to_cartesian()
                
                ini_apo = OrbitalElements(
                    a=self.initial_orbit.a,
                    e=self.initial_orbit.e,
                    i=self.initial_orbit.i * RAD_TO_DEG,
                    raan=self.initial_orbit.raan * RAD_TO_DEG,
                    perigee=self.initial_orbit.perigee * RAD_TO_DEG,
                    mean_anomaly=180.0,
                )
                iapos, _ = ini_apo.to_cartesian()
                
                self.initial_peri_marker._offsets3d = ([ipos[0]], [ipos[1]], [ipos[2]])
                self.initial_apo_marker._offsets3d = ([iapos[0]], [iapos[1]], [iapos[2]])
            else:
                self.initial_peri_marker._offsets3d = ([], [], [])
                self.initial_apo_marker._offsets3d = ([], [], [])
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = OptimizedOrbitAnimationApp(root)
    root.mainloop()
