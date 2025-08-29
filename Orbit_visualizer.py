"""
Real-time orbital animation app with J2 perturbations
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

from orbital_elements import OrbitalElements
from J2_perturbations import J2Perturbations
from constants import EARTH_RADIUS, RAD_TO_DEG, EARTH_NU


# ---------- Helpers ----------

def set_axes_equal(ax, max_extent):
    """Set 3D plot axes to equal scale using a symmetric cube around origin."""
    ax.set_xlim3d([-max_extent, max_extent])
    ax.set_ylim3d([-max_extent, max_extent])
    ax.set_zlim3d([-max_extent, max_extent])


def draw_earth(ax, radius_km=EARTH_RADIUS, wire_alpha=0.35):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, alpha=wire_alpha, color='tab:blue', linewidth=0.5)
    # Equator ring
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(radius_km*np.cos(theta), radius_km*np.sin(theta), np.zeros_like(theta), 'b-', alpha=0.5, linewidth=1.5)


def generate_orbit_curve(orbit: OrbitalElements, num_points: int = 180) -> np.ndarray:
    """Generate a full-orbit 3D curve for the given orbital elements (Nx3)."""
    pts = np.zeros((num_points, 3))
    for k in range(num_points):
        M = 2 * np.pi * (k / num_points)
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

    def pos_vel_at(self, t_seconds: float, angles_orbit: OrbitalElements):
        # Use J2-corrected mean motion if enabled
        try:
            if self.j2_enabled:
                rates = J2Perturbations(angles_orbit).calculate_secular_rates()
                n = rates['mean_anomaly_dot']  # rad/s
            else:
                n = self.n_kepler
        except Exception:
            n = self.n_kepler

        M = (self.M0 + n * t_seconds) % (2 * np.pi)  # radians
        smooth_orbit = OrbitalElements(
            a=angles_orbit.a,
            e=angles_orbit.e,
            i=angles_orbit.i * RAD_TO_DEG,
            raan=angles_orbit.raan * RAD_TO_DEG,
            perigee=angles_orbit.perigee * RAD_TO_DEG,
            mean_anomaly=M * RAD_TO_DEG,
        )
        return smooth_orbit.to_cartesian()


# ---------- Main App ----------

class OrbitAnimationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LEO J2 Orbit Animator")
        self.root.geometry("1900x1050")

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

        # Trails & plotting
        self.max_trail_points = 2000
        self.trail_points: list[list[float]] = []
        self.update_orbit_curve_every = 4  # frames
        self._orbit_curve_tick = 0
        self.orbit_curve_resolution = 180

        # Reference (no-J2) trail
        self.noj2_trail_points: list[list[float]] = []
        self.noj2_trail_max = 1200

        # Smooth positioner
        self.smoother: SmoothPositioner | None = None
        self.noj2_smoother: SmoothPositioner | None = None

        # Precomputed J2 rates for analytic angles progression
        self.j2_rates = None


        # Root layout frames
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)

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

        # Right: 3D plot
        self.fig = plt.Figure(figsize=(12, 9), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar for zoom/pan
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

        # Plot artists
        self.sat_scatter = self.ax.scatter([], [], [], c='r', s=50, label='Satellite')
        self.trail_line, = self.ax.plot([], [], [], 'r-', alpha=0.8, linewidth=2, label='Trail')
        self.full_orbit_line, = self.ax.plot([], [], [], color='tab:cyan', linestyle='--', alpha=0.7, linewidth=1.2, label='Orbit (current)')
        self.initial_orbit_line, = self.ax.plot([], [], [], color='k', linestyle='-', alpha=0.95, linewidth=2.4, label='Orbit (initial)')
        # Perigee/Apogee markers
        self.current_peri_marker = self.ax.scatter([], [], [], marker='v', s=30, c='tab:orange', alpha=0.9, label='Perigee (current)')
        self.current_apo_marker = self.ax.scatter([], [], [], marker='^', s=30, c='tab:cyan', alpha=0.9, label='Apogee (current)')
        self.initial_peri_marker = self.ax.scatter([], [], [], marker='v', s=36, c='k', alpha=0.95, label='Perigee (initial)')
        self.initial_apo_marker = self.ax.scatter([], [], [], marker='^', s=36, c='k', alpha=0.95, label='Apogee (initial)')
        # No-J2 reference sat + trail
        self.noj2_scatter = self.ax.scatter([], [], [], c='tab:green', s=35, marker='o', alpha=0.9, label='Ref. sat (no J2)')
        self.noj2_trail_line, = self.ax.plot([], [], [], color='tab:green', alpha=0.6, linewidth=1.6, label='Ref trail (no J2)')

        # Init with ISS
        self._set_defaults()
        self._update_orbit()
        self._draw_static()

        # Animation
        self.animation = FuncAnimation(self.fig, self._animate, interval=60, blit=False)

    # ----- UI builders -----

    def _build_inputs(self, parent: ttk.Frame):
        box = ttk.LabelFrame(parent, text="Orbital Elements", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))

        self.entries: dict[str, ttk.Entry] = {}
        fields = [
            ('a', 'a (km)'),
            ('e', 'e (0-1)'),
            ('i', 'i (deg)'),
            ('raan', 'RAAN (deg)'),
            ('perigee', 'ω (deg)'),
            ('mean_anomaly', 'M (deg)')
        ]
        for key, label in fields:
            row = ttk.Frame(box)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=13).pack(side=tk.LEFT)
            ent = ttk.Entry(row, width=12)
            ent.pack(side=tk.LEFT)
            self.entries[key] = ent

        # Options
        opt = ttk.Frame(box)
        opt.pack(fill=tk.X, pady=(4, 2))
        self.var_j2 = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Enable J2", variable=self.var_j2, command=self._toggle_j2).pack(side=tk.LEFT)
        self.var_smooth = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Smooth Motion", variable=self.var_smooth, command=self._toggle_smooth).pack(side=tk.LEFT, padx=(8, 0))
        self.var_full_orbit = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Show Current Orbit", variable=self.var_full_orbit).pack(side=tk.LEFT, padx=(8, 0))
        self.var_initial_orbit = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Show Initial Orbit", variable=self.var_initial_orbit, command=self._refresh_initial_orbit_line).pack(side=tk.LEFT, padx=(8, 0))
        self.var_noj2_ref = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Show No-J2 ref sat", variable=self.var_noj2_ref).pack(side=tk.LEFT, padx=(8, 0))
        self.var_lock_axes = tk.BooleanVar(value=False)
        ttk.Checkbutton(opt, text="Lock Axes", variable=self.var_lock_axes).pack(side=tk.LEFT, padx=(8, 0))

        # Buttons
        btns = ttk.Frame(box)
        btns.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(btns, text="Update", command=self._update_orbit).pack(side=tk.LEFT)
        ttk.Button(btns, text="ISS", command=self._preset_iss).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Polar", command=self._preset_polar).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Sun-sync", command=self._preset_sunsync).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="HEO", command=self._preset_heo).pack(side=tk.LEFT, padx=4)

    def _build_info_panel(self, parent: ttk.Frame):
        box = ttk.LabelFrame(parent, text="Orbit Info", padding=6)
        box.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        self.info_text = tk.Text(box, height=26, width=44, font=("Courier New", 9), wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True)

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

        # Speed
        spd = ttk.Frame(box)
        spd.pack(fill=tk.X, pady=(10, 4))
        ttk.Label(spd, text="Speed").pack(side=tk.LEFT)
        self.var_speed = tk.DoubleVar(value=1.0)
        self.scale_speed = ttk.Scale(spd, from_=0.1, to=50.0, orient=tk.HORIZONTAL, variable=self.var_speed, command=self._on_speed)
        self.scale_speed.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_speed = ttk.Label(spd, text="1.0x")
        self.lbl_speed.pack(side=tk.LEFT)

        # Trail length
        trl = ttk.Frame(box)
        trl.pack(fill=tk.X)
        ttk.Label(trl, text="Trail length").pack(side=tk.LEFT)
        self.var_trail = tk.IntVar(value=self.max_trail_points)
        self.scale_trail = ttk.Scale(trl, from_=200, to=10000, orient=tk.HORIZONTAL, command=self._on_trail)
        self.scale_trail.set(self.max_trail_points)
        self.scale_trail.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_trail = ttk.Label(trl, text=str(self.max_trail_points))
        self.lbl_trail.pack(side=tk.LEFT)

        # Orbit curve resolution
        crv = ttk.Frame(box)
        crv.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(crv, text="Orbit curve res").pack(side=tk.LEFT)
        self.var_curve_res = tk.IntVar(value=self.orbit_curve_resolution)
        self.scale_curve = ttk.Scale(crv, from_=90, to=360, orient=tk.HORIZONTAL, command=self._on_curve_res)
        self.scale_curve.set(self.orbit_curve_resolution)
        self.scale_curve.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.lbl_curve = ttk.Label(crv, text=str(self.orbit_curve_resolution))
        self.lbl_curve.pack(side=tk.LEFT)

        # Time label
        tim = ttk.Frame(box)
        tim.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(tim, text="Sim time:").pack(side=tk.LEFT)
        self.lbl_time = ttk.Label(tim, text="0.000 d", font=("Courier New", 11))
        self.lbl_time.pack(side=tk.LEFT, padx=8)

    # ----- Presets -----

    def _set_defaults(self):
        vals = {
            'a': '6793',    # km
            'e': '0.0003',
            'i': '51.6',
            'raan': '0.0',
            'perigee': '0.0',
            'mean_anomaly': '0.0',
        }
        for k, v in vals.items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, v)

    def _preset_iss(self):
        vals = {'a': '6793', 'e': '0.0003', 'i': '51.6', 'raan': '0.0', 'perigee': '0.0', 'mean_anomaly': '0.0'}
        self._apply_values(vals)

    def _preset_polar(self):
        vals = {'a': '6871', 'e': '0.001', 'i': '90.0', 'raan': '0.0', 'perigee': '0.0', 'mean_anomaly': '0.0'}
        self._apply_values(vals)

    def _preset_sunsync(self):
        vals = {'a': '7078', 'e': '0.001', 'i': '98.7', 'raan': '0.0', 'perigee': '0.0', 'mean_anomaly': '0.0'}
        self._apply_values(vals)

    def _preset_heo(self):
        vals = {'a': '12000', 'e': '0.3', 'i': '30.0', 'raan': '45.0', 'perigee': '90.0', 'mean_anomaly': '0.0'}
        self._apply_values(vals)

    def _apply_values(self, values: dict[str, str]):
        for k, v in values.items():
            self.entries[k].delete(0, tk.END)
            self.entries[k].insert(0, v)
        self._update_orbit()

    # ----- Event handlers -----

    def _toggle_j2(self):
        self.j2_enabled = self.var_j2.get()
        if self.current_orbit and self.smoother:
            self.smoother.j2_enabled = self.j2_enabled

    def _toggle_smooth(self):
        self.smooth_enabled = self.var_smooth.get()


    def _toggle_play(self):
        # Reliable play/pause regardless of backend
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
        self.lbl_speed.config(text=f"{self.speed_multiplier:.1f}x")

    def _on_trail(self, *_):
        self.max_trail_points = int(float(self.scale_trail.get()))
        self.lbl_trail.config(text=str(self.max_trail_points))
        # Trim if needed
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points = self.trail_points[-self.max_trail_points:]

    def _on_curve_res(self, *_):
        # Update orbit curve resolution and refresh curves/markers
        try:
            self.orbit_curve_resolution = int(float(self.scale_curve.get()))
        except Exception:
            self.orbit_curve_resolution = 180
        self.lbl_curve.config(text=str(self.orbit_curve_resolution))
        # Recompute initial curve
        if self.initial_orbit is not None:
            try:
                self.initial_orbit_curve_pts = generate_orbit_curve(self.initial_orbit, num_points=self.orbit_curve_resolution)
            except Exception:
                pass
        # Refresh lines and markers
        self._update_full_orbit_curve()
        self._refresh_initial_orbit_line()
        self._update_peri_apo_markers()
        self.canvas.draw_idle()

    def _screenshot(self):
        fname = "orbit_animation.png"
        self.fig.savefig(fname, dpi=300, bbox_inches='tight')
        messagebox.showinfo("Saved", f"Screenshot saved as {fname}")


    def _reset(self):
        self.current_time = 0.0
        self.trail_points.clear()
        self.noj2_trail_points.clear()
        self.lbl_time.config(text="0.000 d")
        # Reset smoother to respect new M0
        if self.current_orbit is not None:
            self.smoother = SmoothPositioner(self.current_orbit, self.j2_enabled)
        if self.initial_orbit is not None:
            self.noj2_smoother = SmoothPositioner(self.initial_orbit, j2_enabled=False)

    def _reset_view(self):
        # Recompute extents from initial and current orbit curves for robustness
        max_extent = EARTH_RADIUS * 1.5
        try:
            if self.initial_orbit_curve_pts is not None and self.initial_orbit_curve_pts.size > 0:
                max_extent = max(max_extent, float(np.max(np.abs(self.initial_orbit_curve_pts))))
        except Exception:
            pass
        try:
            if self.current_orbit is not None:
                cur_pts = generate_orbit_curve(self.current_orbit, num_points=180)
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
        # Set current and initial orbit (initial stays fixed for comparison)
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
            self.initial_orbit_curve_pts = generate_orbit_curve(self.initial_orbit, num_points=self.orbit_curve_resolution)
        except Exception:
            self.initial_orbit_curve_pts = None
        # Precompute J2 rates once from initial orbit
        try:
            self.j2_rates = J2Perturbations(self.initial_orbit).calculate_secular_rates()
        except Exception:
            self.j2_rates = None
        self.smoother = SmoothPositioner(orb, self.j2_enabled)
        self.noj2_smoother = SmoothPositioner(self.initial_orbit, j2_enabled=False)
        self.trail_points.clear()
        self.noj2_trail_points.clear()
        self._orbit_curve_tick = 0
        # Draw and update curves
        self._draw_static()
        self._refresh_initial_orbit_line()
        self._update_full_orbit_curve()
        self._update_peri_apo_markers()
        self.canvas.draw_idle()

    def _draw_static(self):
        self.ax.clear()
        draw_earth(self.ax, radius_km=EARTH_RADIUS)
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.grid(True, alpha=0.25)
        self.ax.set_title(f"LEO Animation ({'J2 ON' if self.j2_enabled else 'J2 OFF'})")

        # Recreate artists after clear()
        self.sat_scatter = self.ax.scatter([], [], [], c='r', s=50, label='Satellite')
        self.trail_line, = self.ax.plot([], [], [], 'r-', alpha=0.85, linewidth=2, label='Trail')
        self.full_orbit_line, = self.ax.plot([], [], [], color='tab:cyan', linestyle='--', alpha=0.7, linewidth=1.2, label='Orbit (current)')
        self.initial_orbit_line, = self.ax.plot([], [], [], color='k', linestyle='-', alpha=0.95, linewidth=2.4, label='Orbit (initial)')
        # Perigee/Apogee markers
        self.current_peri_marker = self.ax.scatter([], [], [], marker='v', s=30, c='tab:orange', alpha=0.9, label='Perigee (current)')
        self.current_apo_marker = self.ax.scatter([], [], [], marker='^', s=30, c='tab:cyan', alpha=0.9, label='Apogee (current)')
        self.initial_peri_marker = self.ax.scatter([], [], [], marker='v', s=36, c='k', alpha=0.95, label='Perigee (initial)')
        self.initial_apo_marker = self.ax.scatter([], [], [], marker='^', s=36, c='k', alpha=0.95, label='Apogee (initial)')
        # No-J2 reference sat + trail
        self.noj2_scatter = self.ax.scatter([], [], [], c='tab:green', s=35, marker='o', alpha=0.9, label='Ref. sat (no J2)')
        self.noj2_trail_line, = self.ax.plot([], [], [], color='tab:green', alpha=0.6, linewidth=1.6, label='Ref trail (no J2)')
        self.ax.legend(loc='upper right')

        # Axis extents from current orbit
        if self.current_orbit and not self.var_lock_axes.get():
            a = self.current_orbit.a
            e = self.current_orbit.e
            r_max = a * (1 + e)
            margin = 800.0
            set_axes_equal(self.ax, max(EARTH_RADIUS * 1.5, r_max + margin))

    def _update_full_orbit_curve(self):
        if not self.current_orbit:
            return
        # Recompute current orbit for the present time using analytic J2 progression
        orbit_now = self._angles_for_time(self.current_time)
        self.current_orbit = orbit_now
        pts = generate_orbit_curve(orbit_now, num_points=self.orbit_curve_resolution)
        if self.var_full_orbit.get():
            self.full_orbit_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        else:
            self.full_orbit_line.set_data_3d([], [], [])
        # Update axes to fully include the orbit curve unless axes are locked
        if not self.var_lock_axes.get():
            max_abs = float(np.max(np.abs(pts))) if pts.size > 0 else EARTH_RADIUS * 2
            # Consider initial orbit as well
            if self.initial_orbit_curve_pts is not None and self.initial_orbit_curve_pts.size > 0:
                max_abs = max(max_abs, float(np.max(np.abs(self.initial_orbit_curve_pts))))
            set_axes_equal(self.ax, max(max_abs * 1.1, EARTH_RADIUS * 1.5))
        # Update current perigee/apogee markers
        self._update_peri_apo_markers()

    def _animate(self, _frame_idx):
        if not self.current_orbit or not self.smoother:
            return self.trail_line,
        if not self.playing:
            return self.trail_line,

        # 1) Determine time increment and prepare sampling times
        dt = self.time_step * self.speed_multiplier
        time_for_pos = self.current_time + dt

        # 2) Multi-sample positions using analytic J2 angles at each sample time
        samples = 1
        if dt > 600.0:
            samples = min(int(dt // 600.0) + 1, 6)
        last_pos = None
        vel = None
        for m in range(1, samples + 1):
            t_samp = self.current_time + dt * (m / samples)
            try:
                angles_at_t = self._angles_for_time(t_samp)
                if self.smooth_enabled:
                    p_s, v_s = self.smoother.pos_vel_at(t_samp, angles_at_t)
                else:
                    p_s, v_s = angles_at_t.to_cartesian()
            except Exception:
                p_s, v_s = (last_pos if last_pos is not None else np.array([0,0,0])), vel
            self.trail_points.append([p_s[0], p_s[1], p_s[2]])
            last_pos, vel = p_s, v_s
        # Keep only the last max_trail_points
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points = self.trail_points[-self.max_trail_points:]
        # Keep only the last max_trail_points
        if len(self.trail_points) > self.max_trail_points:
            self.trail_points = self.trail_points[-self.max_trail_points:]

        # Update artists
        if last_pos is None:
            return self.trail_line,
        self.sat_scatter._offsets3d = ([last_pos[0]], [last_pos[1]], [last_pos[2]])
        if len(self.trail_points) >= 2:
            ta = np.asarray(self.trail_points)
            self.trail_line.set_data_3d(ta[:, 0], ta[:, 1], ta[:, 2])
        else:
            self.trail_line.set_data_3d([], [], [])

        # 4) Update full orbit curve every N frames for performance
        self._orbit_curve_tick = (self._orbit_curve_tick + 1) % self.update_orbit_curve_every
        if self._orbit_curve_tick == 0:
            self._update_full_orbit_curve()
        # Keep initial orbit line visible
        self._refresh_initial_orbit_line()

        # No-J2 reference satellite update
        if self.var_noj2_ref.get() and self.initial_orbit is not None and self.noj2_smoother is not None:
            # multi-sample similar to main trail
            ref_samples = samples
            for m in range(1, ref_samples + 1):
                t_samp = self.current_time + dt * (m / ref_samples)
                try:
                    p_ref, _ = self.noj2_smoother.pos_vel_at(t_samp, self.initial_orbit)
                except Exception:
                    p_ref = last_pos
                self.noj2_trail_points.append([p_ref[0], p_ref[1], p_ref[2]])
            if len(self.noj2_trail_points) > self.noj2_trail_max:
                self.noj2_trail_points = self.noj2_trail_points[-self.noj2_trail_max:]
            if self.noj2_trail_points:
                pr = np.asarray(self.noj2_trail_points)
                self.noj2_trail_line.set_data_3d(pr[:, 0], pr[:, 1], pr[:, 2])
                self.noj2_scatter._offsets3d = ([pr[-1, 0]], [pr[-1, 1]], [pr[-1, 2]])
        else:
            # hide reference
            self.noj2_trail_line.set_data_3d([], [], [])
            self.noj2_scatter._offsets3d = ([], [], [])

        # 5) Update info and time
        self.current_time = time_for_pos
        # Update current_orbit to analytic at time_for_pos for info
        self.current_orbit = self._angles_for_time(self.current_time)
        self._update_info(last_pos, vel)
        days = self.current_time / 86400.0
        self.lbl_time.config(text=f"{days:.3f} d")

        return self.trail_line,

    def _update_info(self, position: np.ndarray, velocity: np.ndarray):
        if not self.current_orbit:
            return
        try:
            r_mag = float(np.linalg.norm(position))
            v_mag = float(np.linalg.norm(velocity))
            altitude = r_mag - EARTH_RADIUS
            # J2 report
            j2_info = {}
            if self.j2_enabled:
                try:
                    j2_info = J2Perturbations(self.current_orbit).get_analysis_report()
                except Exception:
                    j2_info = {}

            def fmt(key, default="-"):
                return j2_info.get(key, default)

            # Deltas from initial
            dOmega = dOmega_str = dOmega_rate_str = "-"
            domega = domega_str = domega_rate_str = "-"
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

            # Apogee/Perigee radii and altitudes (max/min along orbit)
            rp = self.current_orbit.a * (1.0 - self.current_orbit.e)
            ra = self.current_orbit.a * (1.0 + self.current_orbit.e)
            perigee_alt = rp - EARTH_RADIUS
            apogee_alt = ra - EARTH_RADIUS

            info = (
                f"Current time: {self.current_time/86400.0:.3f} d\n\n"
                f"Position (km):\n  X={position[0]:.2f}  Y={position[1]:.2f}  Z={position[2]:.2f}\n"
                f"Velocity (km/s):\n  Vx={velocity[0]:.3f}  Vy={velocity[1]:.3f}  Vz={velocity[2]:.3f}\n"
                f"Speed: {v_mag:.3f} km/s\nAltitude: {altitude:.2f} km\n"
                f"Perigee: r={rp:.2f} km, h={perigee_alt:.2f} km\n"
                f"Apogee:  r={ra:.2f} km, h={apogee_alt:.2f} km\n\n"
                f"Orbit elements (deg):\n  i={self.current_orbit.i*RAD_TO_DEG:.3f}  Ω={self.current_orbit.raan*RAD_TO_DEG:.3f}  ω={self.current_orbit.perigee*RAD_TO_DEG:.3f}\n"
                f"  e={self.current_orbit.e:.5f}  a={self.current_orbit.a:.2f} km\n"
                f"  ΔΩ={dOmega_str} deg   (rate {dOmega_rate_str} deg/day)\n"
                f"  Δω={domega_str} deg   (rate {domega_rate_str} deg/day)\n\n"
            )

            if j2_info:
                info += (
                    "J2 effects:\n"
                    f"  RAAN rate: {fmt('raan_precession_deg_day'):.5f} deg/day\n"
                    f"  ω rate:    {fmt('perigee_precession_deg_day'):.5f} deg/day\n"
                    f"  Period (Kepler): {fmt('kepler_period_hours'):.4f} h\n"
                    f"  Period (J2):     {fmt('j2_modified_period_hours'):.4f} h\n"
                    f"  J2 correction:   {fmt('j2_correction_percent'):.4f} %\n"
                    f"  RAAN period: {fmt('raan_period_days')} d\n"
                    f"  ω period:    {fmt('perigee_period_days')} d\n"
                    f"  Orbit type:  {fmt('orbit_type')}\n"
                    f"  J2 impact:   {fmt('j2_significance')}\n"
                )

            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
        except Exception as e:
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, f"Info error: {e}")

    def _refresh_initial_orbit_line(self):
        # Update the static initial orbit line visibility/data
        try:
            if self.initial_orbit_curve_pts is not None and self.var_initial_orbit.get():
                pts = self.initial_orbit_curve_pts
                self.initial_orbit_line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
                # Update initial perigee/apogee markers
                self._update_peri_apo_markers()
            else:
                self.initial_orbit_line.set_data_3d([], [], [])
                self.initial_peri_marker._offsets3d = ([], [], [])
                self.initial_apo_marker._offsets3d = ([], [], [])
            self.canvas.draw_idle()
        except Exception:
            pass

    def _angles_for_time(self, t_seconds: float) -> OrbitalElements:
        """Build an orbit whose angles (RAAN, perigee) are the analytic J2 progression at time t.
        Mean anomaly is not used here (handled by SmoothPositioner); set 0 for clarity.
        """
        if self.initial_orbit is None:
            return self.current_orbit if self.current_orbit is not None else OrbitalElements(6793,0.0003,51.6,0,0,0)
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
        # Place perigee/apogee markers for current and initial orbits
        try:
            # Current orbit markers (re-evaluate analytic angles at current time)
            if self.var_full_orbit.get():
                cur = self._angles_for_time(self.current_time)
                # perigee = M=0 deg, apogee = M=180 deg
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
                    a=self.current_orbit.a,
                    e=self.current_orbit.e,
                    i=self.current_orbit.i * RAD_TO_DEG,
                    raan=self.current_orbit.raan * RAD_TO_DEG,
                    perigee=self.current_orbit.perigee * RAD_TO_DEG,
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
    app = OrbitAnimationApp(root)
    root.mainloop()

