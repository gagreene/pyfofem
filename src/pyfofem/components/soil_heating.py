"""
soil_heating.py – FOFEM mineral soil temperature prediction models.

Two models are implemented:

1. Campbell (1D equilibrium heat conduction, Campbell et al. 1995):
   Solves the heat equation with a surface flux boundary condition derived
   either from duff smoldering or from a supplied burnup intensity time series.
   Returns a DataFrame of temperature (°C) at each depth over time.

2. Massman HMV (non-equilibrium heat-moisture-vapor, Massman 2015):
   Extends the Campbell model with simplified moisture transport (Richards
   equation) and a BFD-shaped surface heat flux. Returns a dict containing
   temperature and volumetric moisture DataFrames.

Both models use a method-of-lines approach with scipy.integrate.solve_ivp
(Radau stiff solver) on a 15-node non-uniform grid.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Soil family defaults
# ---------------------------------------------------------------------------

_SOIL_FAMILY_DEFAULTS: dict = {
    "loamy-skeletal": dict(
        bulk_density=1230,
        particle_density=2650,
        k_mineral=2.00,
        vries_shape=0.100,
        extrap_water=0.16,
        cop_power=3.43,
    ),
    "fine-silty": dict(
        bulk_density=1100,
        particle_density=2650,
        k_mineral=1.80,
        vries_shape=0.105,
        extrap_water=0.14,
        cop_power=3.20,
    ),
    "fine": dict(
        bulk_density=1050,
        particle_density=2650,
        k_mineral=1.60,
        vries_shape=0.110,
        extrap_water=0.12,
        cop_power=3.10,
    ),
    "coarse-silty": dict(
        bulk_density=1230,
        particle_density=2350,
        k_mineral=2.53,
        vries_shape=0.103,
        extrap_water=0.16,
        cop_power=3.43,
    ),
    "coarse-loamy": dict(
        bulk_density=1350,
        particle_density=2650,
        k_mineral=2.20,
        vries_shape=0.098,
        extrap_water=0.18,
        cop_power=3.50,
    ),
}

# Physical constants
_K_WATER = 0.57       # W/m·K, thermal conductivity of liquid water
_K_AIR = 0.025        # W/m·K, thermal conductivity of air
_C_MINERAL = 870.0    # J/kg·K, specific heat of mineral
_C_WATER = 4180.0     # J/kg·K, specific heat of water
_RHO_WATER = 1000.0   # kg/m³, density of water
_L_V = 2.45e6         # J/kg, latent heat of vaporisation
_G = 9.81             # m/s², gravitational acceleration


# ---------------------------------------------------------------------------
# Private helpers – soil properties
# ---------------------------------------------------------------------------

def _build_soil_props(soil_params: dict) -> dict:
    """Return a unified soil-property dict merging family defaults with overrides."""
    family = soil_params.get("soil_family")
    if family not in _SOIL_FAMILY_DEFAULTS:
        raise ValueError(
            f"Unrecognised soil_family '{family}'. "
            f"Valid options: {list(_SOIL_FAMILY_DEFAULTS.keys())}"
        )
    props = dict(_SOIL_FAMILY_DEFAULTS[family])
    # Allow caller to override any key
    for key in (
        "bulk_density",
        "particle_density",
        "k_mineral",
        "vries_shape",
        "extrap_water",
        "cop_power",
        "start_water",
        "start_temp",
    ):
        if key in soil_params:
            props[key] = soil_params[key]
    props["start_water"] = soil_params["start_water"]
    props["start_temp"] = soil_params["start_temp"]
    return props


def _porosity(rho_b: float, rho_p: float) -> float:
    """Compute porosity φ = 1 - ρ_b / ρ_p."""
    return 1.0 - rho_b / rho_p


def _de_vries_k(
    theta_l: float,
    k_mineral: float,
    rho_b: float,
    rho_p: float,
    vries_shape: float,
) -> float:
    """
    Compute effective thermal conductivity (W/m·K) using the de Vries (1963) model.

    Parameters
    ----------
    theta_l : float
        Volumetric liquid water content (m³/m³).
    k_mineral : float
        Thermal conductivity of mineral grains (W/m·K).
    rho_b : float
        Bulk density (kg/m³).
    rho_p : float
        Particle density (kg/m³).
    vries_shape : float
        de Vries shape factor (used for both air and mineral phases).
    """
    phi = _porosity(rho_b, rho_p)
    f_m = 1.0 - phi                          # mineral volume fraction
    f_w = float(np.clip(theta_l, 0.0, phi))  # liquid water fraction
    f_a = max(phi - f_w, 0.0)               # air fraction

    k_m = k_mineral  # bulk mineral conductivity = the lookup value

    def _weighting(k_c: float) -> float:
        """de Vries weighting factor F_c for component with conductivity k_c."""
        ratio = k_c / k_m
        g = vries_shape
        term1 = 2.0 / (1.0 + (ratio - 1.0) * g)
        term2 = 1.0 / (1.0 + (ratio - 1.0) * (1.0 - 2.0 * g))
        return (1.0 / 3.0) * (term1 + term2)

    F_w = _weighting(_K_WATER)
    F_a = _weighting(_K_AIR)

    numerator = f_m * k_m + F_w * f_w * _K_WATER + F_a * f_a * _K_AIR
    denominator = f_m + F_w * f_w + F_a * f_a

    if denominator == 0.0:
        return 0.01
    k_eff = numerator / denominator
    return max(k_eff, 0.01)


def _volumetric_heat_capacity(rho_b: float, theta_l: float) -> float:
    """
    Volumetric heat capacity ρC (J/m³/K).

    ρC = ρ_b * C_mineral + θ_l * ρ_water * C_water
    """
    rho_c = rho_b * _C_MINERAL + theta_l * _RHO_WATER * _C_WATER
    return max(rho_c, 1e4)


# ---------------------------------------------------------------------------
# Private helpers – grid construction
# ---------------------------------------------------------------------------

def _build_grid(depth_layers: list) -> np.ndarray:
    """
    Build the 15-node depth grid (metres).

    Node 0  : z = 0 (surface)
    Nodes 1..13 : user-specified depths (cm → m)
    Node 14 : deep boundary at 2 * depth_layers[-1] cm (in m)
    """
    depths_m = np.array(depth_layers, dtype=float) / 100.0
    z = np.empty(15)
    z[0] = 0.0
    z[1:14] = depths_m
    z[14] = 2.0 * depths_m[-1]
    return z


def _column_names(depth_layers: list) -> list:
    """Return DataFrame column names: ['Surface', '1cm', '2cm', ...]."""
    cols = ["Surface"]
    for d in depth_layers:
        cols.append(f"{d:.0f}cm")
    return cols


# ---------------------------------------------------------------------------
# Private helpers – surface flux functions
# ---------------------------------------------------------------------------

def _duff_flux_and_duration(duff_params: dict) -> tuple:
    """
    Compute the constant duff smoldering surface heat flux (W/m²)
    and smoldering duration (s) for the duff model.

    Returns
    -------
    q_duff : float
        Heat flux during smoldering (W/m²).
    duration_s : float
        Duration of smoldering (s).
    """
    duff_moisture = duff_params["duff_moisture"]       # percent
    duff_depth_in = duff_params["duff_depth"]          # inches
    duff_depth_cm = duff_depth_in * 2.54               # → cm
    pct_consumed = duff_params["pct_consumed"]         # percent
    duff_heat_content = duff_params.get("duff_heat_content", 20.0)   # MJ/kg
    efficiency_duff = duff_params.get("efficiency_duff", 1.0)         # proportion

    # 1. Burning intensity  (kg/s/m²)
    i_d = max(7.5e-4 - 2.7e-4 * duff_moisture, 0.0)

    # 2. Proportion of heat directed to soil surface (%)
    d_d = duff_depth_cm
    h = -1.6996 + 32.7652 * np.exp(-7.4601 * d_d) + 68.9349 * np.exp(-0.6077 * d_d)
    h = float(np.clip(h, 0.0, 100.0))

    # 3. Smoldering duration
    burn_rate = max(0.05 * np.exp(-0.025 * duff_moisture), 1e-6)  # cm/min
    duration_s = (duff_depth_cm * pct_consumed / 100.0) / burn_rate * 60.0
    duration_s = float(np.clip(duration_s, 60.0, 48.0 * 3600.0))

    # 4. Surface heat flux (W/m²)
    q_duff = i_d * duff_heat_content * 1e6 * (h / 100.0) * efficiency_duff

    return q_duff, duration_s


def _make_duff_flux_fn(q_duff: float, duration_s: float):
    """Return a callable Q(t) → W/m² for the duff smoldering model."""

    def flux(t: float) -> float:
        return q_duff if t <= duration_s else 0.0

    return flux


def _make_nonduff_flux_fn(
    burnup_intensity: Optional[list],
    burnup_times: Optional[list],
    efficiency_wl: float,
    efficiency_hs: float,
) -> tuple:
    """
    Return (Q_fn, t_end_extra) for the non_duff model.

    Q_fn : callable t → W/m²
    last_time : float – last time in burnup_times (s)
    """
    if burnup_intensity is None or burnup_times is None:
        # Default: 20 kW/m² for 30 minutes (1800 s)
        default_q = 20.0 * (efficiency_wl + efficiency_hs) * 1000.0
        last_time = 1800.0

        def flux(t: float) -> float:
            return default_q if t <= last_time else 0.0

        return flux, last_time

    bt = np.asarray(burnup_times, dtype=float)
    bi = np.asarray(burnup_intensity, dtype=float)
    factor = (efficiency_wl + efficiency_hs) * 1000.0  # kW/m² → W/m²
    interp = interp1d(bt, bi, kind="linear", bounds_error=False, fill_value=(bi[0], 0.0))
    last_time = float(bt[-1])

    def flux(t: float) -> float:
        if t > last_time:
            return 0.0
        return float(interp(t)) * factor

    return flux, last_time


def _make_nonduff_flux_fn_split(
    burnup_intensity_wl: Optional[list],
    burnup_intensity_hs: Optional[list],
    burnup_times: Optional[list],
    efficiency_wl: float,
    efficiency_hs: float,
) -> tuple:
    """Like _make_nonduff_flux_fn, but accepts separate WL and HS series."""
    if burnup_intensity_wl is None:
        return _make_nonduff_flux_fn(
            burnup_intensity=None,
            burnup_times=burnup_times,
            efficiency_wl=efficiency_wl,
            efficiency_hs=efficiency_hs,
        )
    if burnup_intensity_hs is None:
        return _make_nonduff_flux_fn(
            burnup_intensity=burnup_intensity_wl,
            burnup_times=burnup_times,
            efficiency_wl=efficiency_wl,
            efficiency_hs=efficiency_hs,
        )
    if burnup_times is None:
        return _make_nonduff_flux_fn(
            burnup_intensity=None,
            burnup_times=None,
            efficiency_wl=efficiency_wl,
            efficiency_hs=efficiency_hs,
        )

    bt = np.asarray(burnup_times, dtype=float)
    bi_wl = np.asarray(burnup_intensity_wl, dtype=float)
    bi_hs = np.asarray(burnup_intensity_hs, dtype=float)
    if bi_wl.shape != bi_hs.shape:
        raise ValueError(
            "burnup_intensity_hs must have same shape as burnup_intensity_wl."
        )
    wl_interp = interp1d(
        bt, bi_wl, kind="linear", bounds_error=False, fill_value=(bi_wl[0], 0.0)
    )
    hs_interp = interp1d(
        bt, bi_hs, kind="linear", bounds_error=False, fill_value=(bi_hs[0], 0.0)
    )
    last_time = float(bt[-1])
    wl_factor = efficiency_wl * 1000.0
    hs_factor = efficiency_hs * 1000.0

    def flux(t: float) -> float:
        if t > last_time:
            return 0.0
        return float(wl_interp(t)) * wl_factor + float(hs_interp(t)) * hs_factor

    return flux, last_time


def _make_bfd_flux_fn(q_abs: float, t_m_s: float, t_d_s: float):
    """
    BFD (beta-function-derived) surface heat flux for Massman model.

    Q(t) = q_abs * 1000 * (t / t_m_s) * exp(1 - t / t_m_s)  for t ≤ t_d_s
    Q(t) = 0                                                    for t > t_d_s
    """
    peak_wm2 = q_abs * 1000.0  # kW/m² → W/m²

    def flux(t: float) -> float:
        if t > t_d_s or t_m_s <= 0.0:
            return 0.0
        tau = t / t_m_s
        return peak_wm2 * tau * np.exp(1.0 - tau)

    return flux


# ---------------------------------------------------------------------------
# Private helpers – ODE right-hand sides
# ---------------------------------------------------------------------------

def _campbell_rhs(
    t: float,
    state: np.ndarray,
    z: np.ndarray,
    rho_b: float,
    rho_p: float,
    k_mineral: float,
    vries_shape: float,
    theta_l: np.ndarray,
    start_temp: float,
    flux_fn,
) -> np.ndarray:
    """
    RHS of the Campbell heat-only ODE system.

    State: T[0..13] (T[14] = start_temp, fixed deep boundary).
    """
    T = state  # length 14 (nodes 0..13)
    n = 14     # number of state variables

    # Augment with fixed deep boundary
    T_full = np.empty(15)
    T_full[:14] = T
    T_full[14] = start_temp

    # Thermal conductivity at each node (use initial theta_l – static for Campbell)
    k = np.array(
        [
            _de_vries_k(theta_l[i], k_mineral, rho_b, rho_p, vries_shape)
            for i in range(15)
        ]
    )

    # Volumetric heat capacity at each node
    rho_c = np.array([_volumetric_heat_capacity(rho_b, theta_l[i]) for i in range(14)])

    dT = np.zeros(14)

    # Surface node (i=0): half-cell from z=0 to z[1]/2
    dz1 = z[1] - z[0]  # = z[1] since z[0]=0
    k_01 = 2.0 * k[0] * k[1] / (k[0] + k[1])  # harmonic mean
    Q_surf = flux_fn(t)
    dT[0] = (k_01 * (T_full[1] - T_full[0]) / dz1 + Q_surf) / (rho_c[0] * dz1 / 2.0)

    # Interior nodes 1..12
    for i in range(1, 13):
        dz_m = z[i] - z[i - 1]
        dz_p = z[i + 1] - z[i]
        dz_cv = (dz_m + dz_p) / 2.0
        k_m = 2.0 * k[i] * k[i - 1] / (k[i] + k[i - 1])
        k_p = 2.0 * k[i] * k[i + 1] / (k[i] + k[i + 1])
        flux_in = k_p * (T_full[i + 1] - T_full[i]) / dz_p
        flux_out = k_m * (T_full[i] - T_full[i - 1]) / dz_m
        dT[i] = (flux_in - flux_out) / (rho_c[i] * dz_cv)

    # Node 13 (deepest user layer) – uses T[14] = start_temp as boundary
    i = 13
    dz_m = z[13] - z[12]
    dz_p = z[14] - z[13]
    dz_cv = (dz_m + dz_p) / 2.0
    k_m = 2.0 * k[13] * k[12] / (k[13] + k[12])
    k_p = 2.0 * k[13] * k[14] / (k[13] + k[14])
    flux_in = k_p * (T_full[14] - T_full[13]) / dz_p
    flux_out = k_m * (T_full[13] - T_full[12]) / dz_m
    dT[13] = (flux_in - flux_out) / (rho_c[13] * dz_cv)

    return dT


def _massman_rhs(
    t: float,
    state: np.ndarray,
    z: np.ndarray,
    rho_b: float,
    rho_p: float,
    k_mineral: float,
    vries_shape: float,
    start_temp: float,
    extrap_water: float,
    cop_power: float,
    k_sat: float,
    flux_fn,
) -> np.ndarray:
    """
    RHS of the Massman HMV ODE system (simplified: E_v = 0).

    State layout: [T[0..13], theta_l[0..13]]  (28 values; nodes 14 fixed externally)
    """
    T = state[:14]
    theta_l = state[14:28]

    phi = _porosity(rho_b, rho_p)

    # Augment with fixed deep boundaries
    T_full = np.empty(15)
    T_full[:14] = T
    T_full[14] = start_temp

    theta_l_full = np.empty(15)
    theta_l_full[:14] = theta_l
    theta_l_full[14] = theta_l[13]  # keep bottom moisture fixed

    # Clip moisture to valid range
    theta_l_c = np.clip(theta_l_full, 1e-6, phi)

    # Thermal conductivity and heat capacity at each node
    k = np.array(
        [
            _de_vries_k(theta_l_c[i], k_mineral, rho_b, rho_p, vries_shape)
            for i in range(15)
        ]
    )
    rho_c = np.array(
        [_volumetric_heat_capacity(rho_b, theta_l_c[i]) for i in range(14)]
    )

    dT = np.zeros(14)
    dtheta = np.zeros(14)

    # --- Heat equation (same stencil as Campbell) ---
    Q_surf = flux_fn(t)
    dz1 = z[1] - z[0]
    k_01 = 2.0 * k[0] * k[1] / (k[0] + k[1])
    dT[0] = (k_01 * (T_full[1] - T_full[0]) / dz1 + Q_surf) / (rho_c[0] * dz1 / 2.0)

    for i in range(1, 13):
        dz_m = z[i] - z[i - 1]
        dz_p = z[i + 1] - z[i]
        dz_cv = (dz_m + dz_p) / 2.0
        k_m = 2.0 * k[i] * k[i - 1] / (k[i] + k[i - 1])
        k_p = 2.0 * k[i] * k[i + 1] / (k[i] + k[i + 1])
        flux_in = k_p * (T_full[i + 1] - T_full[i]) / dz_p
        flux_out = k_m * (T_full[i] - T_full[i - 1]) / dz_m
        dT[i] = (flux_in - flux_out) / (rho_c[i] * dz_cv)

    i = 13
    dz_m = z[13] - z[12]
    dz_p = z[14] - z[13]
    dz_cv = (dz_m + dz_p) / 2.0
    k_m = 2.0 * k[13] * k[12] / (k[13] + k[12])
    k_p = 2.0 * k[13] * k[14] / (k[13] + k[14])
    flux_in = k_p * (T_full[14] - T_full[13]) / dz_p
    flux_out = k_m * (T_full[13] - T_full[12]) / dz_m
    dT[13] = (flux_in - flux_out) / (rho_c[13] * dz_cv)

    # --- Moisture equation (simplified Richards) ---
    # Hydraulic conductivity
    def _K_l(theta: float) -> float:
        sat_ratio = float(np.clip(theta / phi, 0.0, 1.0))
        return k_sat * sat_ratio ** (2.0 * cop_power + 3.0)

    # Matric potential (J/kg)
    def _psi(theta: float) -> float:
        sat_ratio = float(np.clip(theta / phi, 1e-9, 1.0))
        return extrap_water * sat_ratio ** (-cop_power)

    # Compute water fluxes at inter-node interfaces (15 nodes → 14 interfaces)
    # Interface flux: q = -K_l * (dψ/dz + g)  (positive downward)
    # Use no-flux at surface (interface 0) and bottom (interface 13)
    q_water = np.zeros(15)  # q_water[i] = flux at interface between node i-1 and i

    # Interface 1..13 (between nodes 0-1, 1-2, ..., 12-13)
    for iface in range(1, 14):
        i_lo = iface - 1
        i_hi = iface
        dz_if = z[i_hi] - z[i_lo]
        K_avg = 0.5 * (_K_l(theta_l_c[i_lo]) + _K_l(theta_l_c[i_hi]))
        dpsi_dz = (_psi(theta_l_c[i_hi]) - _psi(theta_l_c[i_lo])) / dz_if
        q_water[iface] = -K_avg * (dpsi_dz + _G)

    # q_water[0] = 0 (no flux at surface)
    # q_water[14] = 0 (no flux at bottom)

    # dθ_l/dt for each node
    # Node 0: half-cell from 0 to z[1]/2, bottom interface at z[1]/2
    dz_cv0 = z[1] / 2.0
    dtheta[0] = -(q_water[1] - q_water[0]) / dz_cv0

    for i in range(1, 13):
        dz_m = z[i] - z[i - 1]
        dz_p = z[i + 1] - z[i]
        dz_cv = (dz_m + dz_p) / 2.0
        dtheta[i] = -(q_water[i + 1] - q_water[i]) / dz_cv

    # Node 13
    dz_m = z[13] - z[12]
    dz_p = z[14] - z[13]
    dz_cv = (dz_m + dz_p) / 2.0
    dtheta[13] = -(q_water[14] - q_water[13]) / dz_cv

    return np.concatenate([dT, dtheta])


# ---------------------------------------------------------------------------
# Private helper – build t_eval
# ---------------------------------------------------------------------------

def _build_t_eval(t_end: float) -> np.ndarray:
    """
    Build evaluation times every 30 s (0.5 min) from 0 to t_end (s).
    """
    return np.arange(0.0, t_end + 1.0, 30.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def soil_heat_campbell(
    model: str,
    duff_params: dict,
    soil_params: dict,
    depth_layers: list,
    burnup_intensity: Optional[list] = None,
    burnup_intensity_hs: Optional[list] = None,
    burnup_times: Optional[list] = None,
    efficiency_wl: float = 0.15,
    efficiency_hs: float = 0.10,
    timestep: float = 10.0,
) -> pd.DataFrame:
    """
    Predict mineral soil temperature using the Campbell (1995) heat-conduction model.

    Parameters
    ----------
    model : str
        ``'duff'`` – surface flux from duff smoldering.
        ``'non_duff'`` – surface flux from a burnup intensity time series.
    duff_params : dict
        Parameters describing the duff layer (used when *model* = ``'duff'``):

        - ``duff_load`` (T/ac) – pre-fire duff loading
        - ``duff_depth`` (in) – pre-fire duff depth
        - ``duff_moisture`` (%) – gravimetric moisture content
        - ``pct_consumed`` (%) – percent of duff consumed
        - ``duff_heat_content`` (MJ/kg, default 20) – heat of combustion
        - ``efficiency_duff`` (proportion, default 1.0) – fraction of heat to soil
    soil_params : dict
        Soil properties:

        - ``soil_family`` (str) – one of ``'loamy-skeletal'``, ``'fine-silty'``,
          ``'fine'``, ``'coarse-silty'``, ``'coarse-loamy'``
        - ``start_water`` (m³/m³) – initial volumetric water content
        - ``start_temp`` (°C) – initial uniform soil temperature
        - Optional overrides for any soil-family default key.
    depth_layers : list of float
        Exactly 13 depths (cm) at which to predict temperature.
    burnup_intensity : list of float, optional
        Fire intensity (kW/m²) at each time step; used when *model* = ``'non_duff'``.
    burnup_times : list of float, optional
        Times (s) corresponding to *burnup_intensity*.
    efficiency_wl : float
        Fraction of woody-litter intensity delivered to the soil surface (default 0.15).
    efficiency_hs : float
        Fraction of heavy-slash intensity delivered to the soil surface (default 0.10).
    timestep : float
        Integration time step (s, default 10). Currently informational; Radau adapts
        internally but this sets the maximum step.

    Returns
    -------
    pd.DataFrame
        Index: time in minutes (float).
        Columns: ``'Surface'``, then ``'<d>cm'`` for each depth in *depth_layers*.
    """
    model = model.strip().lower()
    if model not in ("duff", "non_duff"):
        raise ValueError(f"model must be 'duff' or 'non_duff', got '{model}'.")
    if len(depth_layers) != 13:
        raise ValueError(
            f"depth_layers must contain exactly 13 values, got {len(depth_layers)}."
        )

    props = _build_soil_props(soil_params)
    rho_b = props["bulk_density"]
    rho_p = props["particle_density"]
    k_mineral = props["k_mineral"]
    vries_shape = props["vries_shape"]
    start_water = props["start_water"]
    start_temp = props["start_temp"]

    z = _build_grid(depth_layers)

    # Initial moisture (static throughout Campbell simulation)
    theta_l = np.full(15, start_water)

    # Initial temperature
    T0 = np.full(14, start_temp, dtype=float)

    # Surface flux function and simulation end time
    if model == "duff":
        q_duff, duration_s = _duff_flux_and_duration(duff_params)
        flux_fn = _make_duff_flux_fn(q_duff, duration_s)
        t_end = max(duration_s + 2.0 * 3600.0, 4.0 * 3600.0)
    else:
        flux_fn, last_time = _make_nonduff_flux_fn_split(
            burnup_intensity, burnup_intensity_hs, burnup_times, efficiency_wl, efficiency_hs
        )
        t_end = last_time + 2.0 * 3600.0

    t_eval = _build_t_eval(t_end)

    def rhs(t, y):
        return _campbell_rhs(
            t, y, z, rho_b, rho_p, k_mineral, vries_shape, theta_l, start_temp, flux_fn
        )

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        T0,
        method="Radau",
        t_eval=t_eval,
        max_step=timestep,
        rtol=1e-4,
        atol=1e-6,
    )

    # Build DataFrame
    # sol.y shape: (14, n_times); we want nodes 0..13 → Surface + 13 depth columns
    times_min = sol.t / 60.0
    data = sol.y[:14, :].T  # (n_times, 14)
    cols = _column_names(depth_layers)  # 14 names: Surface + 13 depths
    df = pd.DataFrame(data, index=times_min, columns=cols)
    df.index.name = "time_min"
    return df


def soil_heat_massman(
    fire_type: str,
    bfd_params: dict,
    soil_params: dict,
    depth_layers: list,
    timestep: float = 10.0,
) -> dict:
    """
    Predict mineral soil temperature and moisture using the Massman (2015) HMV model.

    The implementation uses a simplified version (E_v = 0, isothermal evaporation
    assumption) that couples the Campbell heat-conduction model with a simplified
    Richards equation for moisture redistribution. The surface forcing uses a
    BFD (beta-function-derived) heat flux curve.

    Parameters
    ----------
    fire_type : str
        ``'wildfire'``, ``'prescribed_burn'``, or ``'pile_burn'``.
    bfd_params : dict
        BFD fire curve parameters:

        - ``q_abs`` (kW/m²) – peak heat rate
        - ``t_m`` (hr, default 4) – time to peak heat rate
        - ``t_d`` (hr) – fire duration (default 20 for wildfire, 8 for Rx, 40 for pile)
    soil_params : dict
        Same keys as for :func:`soil_heat_campbell`, plus optional Massman-specific
        overrides:

        - ``extrap_water`` (default 0.16) – water content extrapolated to −1 J/kg
        - ``vries_shape`` (default from soil family) – de Vries shape factor
        - ``cop_power`` (default from soil family) – power for liquid recirculation
    depth_layers : list of float
        Exactly 13 depths (cm).
    timestep : float
        Maximum integration step (s, default 10).

    Returns
    -------
    dict
        ``'temperature'``: pd.DataFrame – time (min) × depth (°C)
        ``'moisture'``: pd.DataFrame – time (min) × depth (m³/m³)
    """
    fire_type = fire_type.strip().lower()
    valid_fire_types = ("wildfire", "prescribed_burn", "pile_burn")
    if fire_type not in valid_fire_types:
        raise ValueError(
            f"fire_type must be one of {valid_fire_types}, got '{fire_type}'."
        )
    if len(depth_layers) != 13:
        raise ValueError(
            f"depth_layers must contain exactly 13 values, got {len(depth_layers)}."
        )

    # Default fire durations
    _default_t_d = {"wildfire": 20.0, "prescribed_burn": 8.0, "pile_burn": 40.0}
    t_m = float(bfd_params.get("t_m", 4.0))
    t_d = float(bfd_params.get("t_d", _default_t_d[fire_type]))
    q_abs = float(bfd_params["q_abs"])

    t_m_s = t_m * 3600.0
    t_d_s = t_d * 3600.0
    t_end = (t_d + 2.0) * 3600.0

    props = _build_soil_props(soil_params)
    # Allow Massman-specific overrides from soil_params
    for key in ("extrap_water", "vries_shape", "cop_power"):
        if key in soil_params:
            props[key] = soil_params[key]

    rho_b = props["bulk_density"]
    rho_p = props["particle_density"]
    k_mineral = props["k_mineral"]
    vries_shape = props["vries_shape"]
    start_water = props["start_water"]
    start_temp = props["start_temp"]
    extrap_water = props["extrap_water"]
    cop_power = props["cop_power"]

    # Saturated hydraulic conductivity (m/s) from bulk density
    k_sat = 0.0001 * np.exp(-3.0 * rho_b / 1000.0)

    z = _build_grid(depth_layers)

    # Initial state: [T[0..13], theta_l[0..13]]
    T_init = np.full(14, start_temp, dtype=float)
    theta_init = np.full(14, start_water, dtype=float)
    state0 = np.concatenate([T_init, theta_init])

    flux_fn = _make_bfd_flux_fn(q_abs, t_m_s, t_d_s)
    t_eval = _build_t_eval(t_end)

    def rhs(t, y):
        return _massman_rhs(
            t,
            y,
            z,
            rho_b,
            rho_p,
            k_mineral,
            vries_shape,
            start_temp,
            extrap_water,
            cop_power,
            k_sat,
            flux_fn,
        )

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        state0,
        method="Radau",
        t_eval=t_eval,
        max_step=timestep,
        rtol=1e-4,
        atol=1e-6,
    )

    times_min = sol.t / 60.0
    T_out = sol.y[:14, :].T        # (n_times, 14) temperature
    theta_out = sol.y[14:28, :].T  # (n_times, 14) moisture

    cols = _column_names(depth_layers)

    df_temp = pd.DataFrame(T_out, index=times_min, columns=cols)
    df_temp.index.name = "time_min"

    df_moist = pd.DataFrame(theta_out, index=times_min, columns=cols)
    df_moist.index.name = "time_min"

    return {"temperature": df_temp, "moisture": df_moist}
