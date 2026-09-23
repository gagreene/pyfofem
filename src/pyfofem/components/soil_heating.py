"""
soil_heating.py – FOFEM mineral soil temperature prediction models.

Two soil-heating approaches are represented here:

1. Campbell (1D equilibrium heat conduction, Campbell et al. 1995):
   Solves the heat equation with a surface flux boundary condition derived
   either from duff smoldering or from a supplied burnup intensity time series.
   Returns a DataFrame of temperature (°C) at each depth over time.

2. Massman HMV (Massman 2015):
   Reserved for a future, full non-equilibrium heat-moisture-vapor
   implementation. The previous simplified approximation is retained only
   as in-development source code and is deliberately unavailable to users.

Campbell uses a coupled nonlinear solve on a 15-node non-uniform grid.
"""

import math
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Soil family defaults
# ---------------------------------------------------------------------------

#: Soil-family physical constants used by the coupled Campbell model.
#: Densities remain in their native mass-per-volume scale because they are
#: used directly in the volumetric heat-capacity calculation. ``recirc_water``
#: is distinct from ``extrap_water``: it controls liquid-water recirculation
#: in the thermal-conductivity relation.
_SOIL_FAMILY_DEFAULTS: dict = {
    "loamy-skeletal": dict(
        bulk_density=0.8e6,
        particle_density=2.13e6,
        k_mineral=1.03,
        vries_shape=0.13,
        extrap_water=0.321,
        recirc_water=0.133,
        cop_power=6.08,
    ),
    "fine-silty": dict(
        bulk_density=1.3e6,
        particle_density=2.35e6,
        k_mineral=2.31,
        vries_shape=0.071,
        extrap_water=0.207,
        recirc_water=0.148,
        cop_power=4.14,
    ),
    "fine": dict(
        bulk_density=1.15e6,
        particle_density=2.35e6,
        k_mineral=2.21,
        vries_shape=0.084,
        extrap_water=0.202,
        recirc_water=0.152,
        cop_power=4.63,
    ),
    "coarse-silty": dict(
        bulk_density=1.23e6,
        particle_density=2.35e6,
        k_mineral=2.53,
        vries_shape=0.103,
        extrap_water=0.157,
        recirc_water=0.218,
        cop_power=3.43,
    ),
    "coarse-loamy": dict(         # C++ "Coarse-Loamy"
        bulk_density=1.3e6,
        particle_density=2.35e6,
        k_mineral=2.57,
        vries_shape=0.106,
        extrap_water=0.102,
        recirc_water=0.127,
        cop_power=2.93,
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
# Functions
# ---------------------------------------------------------------------------

def _build_grid(depth_layers: list) -> np.ndarray:
    """
    Build the 15-node depth grid (metres).

    Node 0 is the surface (z=0), nodes 1-13 are the user-specified depths
    (converted cm to m), and node 14 is a deep boundary at 2x the deepest
    user-specified depth.

    :param depth_layers: 13 depths (cm) at which temperature is predicted.
    :return: 1-D array of 15 node depths (m), surface to deep boundary.
    """
    depths_m = np.array(depth_layers, dtype=float) / 100.0
    z = np.empty(15)
    z[0] = 0.0
    z[1:14] = depths_m
    z[14] = 2.0 * depths_m[-1]
    return z


def _build_soil_props(soil_params: dict) -> dict:
    """
    Return a unified soil-property dict merging family defaults with overrides.

    :param soil_params: Dict with required key 'soil_family' (one of
        :data:`_SOIL_FAMILY_DEFAULTS`'s keys), required 'start_water' and
        'start_temp', and optional overrides for any family-default key
        ('bulk_density', 'particle_density', 'k_mineral', 'vries_shape',
        'extrap_water', 'recirc_water', 'cop_power').
    :return: Dict of resolved soil properties (family defaults with any
        supplied overrides applied, plus 'start_water'/'start_temp').
    :raises ValueError: If 'soil_family' is not a recognised family name.
    """
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
        "recirc_water",
        "cop_power",
        "start_water",
        "start_temp",
    ):
        if key in soil_params:
            props[key] = soil_params[key]
    props["start_water"] = soil_params["start_water"]
    props["start_temp"] = soil_params["start_temp"]
    return props


def _build_t_eval(t_end: float) -> np.ndarray:
    """
    Build evaluation times every 30 s (0.5 min) from 0 to t_end (s).

    :param t_end: Simulation end time (s).
    :return: 1-D array of evaluation times (s), spaced 30 s apart.
    """
    return np.arange(0.0, t_end + 1.0, 30.0)


def _column_names(depth_layers: list) -> list:
    """
    Return DataFrame column names: ['Surface', '1cm', '2cm', ...].

    :param depth_layers: 13 depths (cm) at which temperature is predicted.
    :return: List of 14 column name strings (surface plus one per depth).
    """
    cols = ["Surface"]
    for d in depth_layers:
        cols.append(f"{d:.0f}cm")
    return cols


def _de_vries_k(
        theta_l: float,
        k_mineral: float,
        rho_b: float,
        rho_p: float,
        vries_shape: float,
) -> float:
    """
    Compute effective thermal conductivity (W/m·K) using the de Vries (1963) model.

    :param theta_l: Volumetric liquid water content (m³/m³).
    :param k_mineral: Thermal conductivity of mineral grains (W/m·K).
    :param rho_b: Bulk density (kg/m³).
    :param rho_p: Particle density (kg/m³).
    :param vries_shape: de Vries shape factor (used for both air and mineral phases).
    :return: Effective thermal conductivity (W/m·K), floored at 0.01.
    """
    phi = _porosity(rho_b, rho_p)
    f_m = 1.0 - phi                          # mineral volume fraction
    f_w = float(np.clip(theta_l, 0.0, phi))  # liquid water fraction
    f_a = max(phi - f_w, 0.0)               # air fraction

    k_m = k_mineral  # bulk mineral conductivity = the lookup value

    def _weighting(k_c: float) -> float:
        """
        de Vries weighting factor F_c for a component with conductivity k_c.

        :param k_c: Thermal conductivity of the component (W/m·K).
        :return: Weighting factor F_c (dimensionless).
        """
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


def _make_bfd_flux_fn(q_abs: float, t_m_s: float, t_d_s: float):
    """
    Build a BFD (beta-function-derived) surface heat flux function for the Massman model.

    Q(t) = q_abs * 1000 * (t / t_m_s) * exp(1 - t / t_m_s) for t <= t_d_s, else 0.

    :param q_abs: Peak heat rate (kW/m²).
    :param t_m_s: Time to peak heat rate (s).
    :param t_d_s: Fire duration (s); flux is 0 after this time.
    :return: Callable t -> surface heat flux (W/m²).
    """
    peak_wm2 = q_abs * 1000.0  # kW/m² → W/m²

    def flux(t: float) -> float:
        """
        Evaluate the BFD surface heat flux at time t.

        :param t: Simulation time (s).
        :return: Surface heat flux (W/m²).
        """
        if t > t_d_s or t_m_s <= 0.0:
            return 0.0
        tau = t / t_m_s
        return peak_wm2 * tau * np.exp(1.0 - tau)

    return flux


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

    State layout is [T[0..13], theta_l[0..13]] (28 values); node 14 is
    fixed externally as the deep boundary for both temperature and moisture.

    :param t: Current simulation time (s).
    :param state: Current [temperature, moisture] state, length 28.
    :param z: Node depths (m), length 15, from :func:`_build_grid`.
    :param rho_b: Soil bulk density (kg/m³).
    :param rho_p: Soil particle density (kg/m³).
    :param k_mineral: Thermal conductivity of mineral grains (W/m·K).
    :param vries_shape: de Vries shape factor.
    :param start_temp: Deep-boundary (node 14) temperature (°C).
    :param extrap_water: Water content extrapolated to -1 J/kg matric potential.
    :param cop_power: Power exponent for liquid recirculation (Campbell/Cop model).
    :param k_sat: Saturated hydraulic conductivity (m/s).
    :param flux_fn: Callable t -> surface heat flux (W/m²).
    :return: Concatenated [dT/dt, dtheta_l/dt] for nodes 0-13, length 28.
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
        """
        Unsaturated hydraulic conductivity at a given moisture content.

        :param theta: Volumetric liquid water content (m³/m³).
        :return: Hydraulic conductivity (m/s).
        """
        sat_ratio = float(np.clip(theta / phi, 0.0, 1.0))
        return k_sat * sat_ratio ** (2.0 * cop_power + 3.0)

    # Matric potential (J/kg)
    def _psi(theta: float) -> float:
        """
        Matric potential at a given moisture content.

        :param theta: Volumetric liquid water content (m³/m³).
        :return: Matric potential (J/kg).
        """
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


def _porosity(rho_b: float, rho_p: float) -> float:
    """
    Compute porosity phi = 1 - rho_b / rho_p.

    :param rho_b: Bulk density (kg/m³).
    :param rho_p: Particle density (kg/m³).
    :return: Porosity (m³/m³, dimensionless).
    """
    return 1.0 - rho_b / rho_p


def _volumetric_heat_capacity(rho_b: float, theta_l: float) -> float:
    """
    Volumetric heat capacity rho*C (J/m³/K).

    rho*C = rho_b * C_mineral + theta_l * rho_water * C_water

    :param rho_b: Bulk density (kg/m³).
    :param theta_l: Volumetric liquid water content (m³/m³).
    :return: Volumetric heat capacity (J/m³/K), floored at 1e4.
    """
    rho_c = rho_b * _C_MINERAL + theta_l * _RHO_WATER * _C_WATER
    return max(rho_c, 1e4)


# ---------------------------------------------------------------------------
# Coupled Campbell heat/moisture/vapor solver
# ---------------------------------------------------------------------------

#: Number of solver-state array slots. Index 0 is the virtual air boundary;
#: indices 1–14 are soil nodes, with index 14 as the fixed deep boundary.
_CAMPBELL_NODE_COUNT = 15
_CAMPBELL_DEEP_NODE = 14
_CAMPBELL_INTERIOR_NODE_COUNT = _CAMPBELL_DEEP_NODE - 1

_CAMPBELL_ATMOSPHERIC_PRESSURE = 92000.0       # Atmospheric pressure at site (Pa).
_CAMPBELL_VAPOR_DIFFUSIVITY = 2.12e-5          # Vapor diffusivity in air (m²/s).
_CAMPBELL_STANDARD_PRESSURE = 101300.0         # Standard atmospheric pressure (Pa).
_CAMPBELL_GAS_CONSTANT = 8.3143                # Gas constant (J/mol/K).
_CAMPBELL_WATER_MOLAR_MASS = 0.018             # Molar mass of water (kg/mol).
_CAMPBELL_BOUNDARY_RESISTANCE = 20.0           # Surface boundary-layer resistance.
_CAMPBELL_ENERGY_ERROR_LIMIT = 100.0           # Energy-balance error limit (W/m²).
_CAMPBELL_WATER_ERROR_LIMIT = 1e-5             # Water mass-balance error limit (kg/(m² s)).
_CAMPBELL_WATER_DENSITY = 1000.0               # Water density (kg/m³).
_CAMPBELL_SOIL_TORTUOSITY = 0.66               # Soil tortuosity.
_CAMPBELL_AIR_VAPOR_PRESSURE = 1000.0          # Air vapor pressure (Pa).
_CAMPBELL_AIR_TEMPERATURE = 20.0               # Initial air temperature (°C).
_CAMPBELL_STANDARD_TEMPERATURE = 273.15        # Standard temperature (K).
_CAMPBELL_MAX_NEWTON_ITERATIONS = 500           # Newton-iteration failure limit.

#: Defensive outer-step cap for pathological non-terminating inputs.
_CAMPBELL_MAX_TIMESTEPS = 200_000

#: Fixed route timesteps (s). The public ``timestep`` parameter is accepted
#: for compatibility and does not change the coupled solver timestep.
_CAMPBELL_DUFF_TIMESTEP = 20.0
_CAMPBELL_NONDUFF_TIMESTEP = 10.0


class SoilSimulationError(RuntimeError):
    """
    Raised when the coupled soil solver fails to converge.

    Raised after :data:`_CAMPBELL_MAX_NEWTON_ITERATIONS` non-converging Newton
    iterations. The failure is immediate rather than retried because a
    converged timestep is required before advancing the coupled state.
    """


#: Tons per acre to kilograms per square metre for the duff-burn relation.
_TONS_ACRE_TO_KG_M2 = 1.0 / 4.46

#: Inches to centimetres for the duff heat-adjustment relation.
_INCH_TO_CM = 100.0 / 39.37

#: Non-burning duff-moisture threshold as a ratio, not a percentage.
_DUFF_BURN_MOISTURE_RATIO_MAX = 1.96


def _campbell_commit_timestep(state: dict, dt: float) -> None:
    """
    Post-convergence "commit" step for one coupled timestep: back-sweep
    the cumulative vapor flux and advance ``t``/``w`` to the newly
    Newton-converged ``tn``/``wn`` values.

    :param state: State dict from :func:`_soiltemp_initconsts` (and
        typically :func:`_soiltemp_initprofile`), after a converged
        Newton sub-iteration loop. Mutated in place: ``u``, ``w``,
        ``wn``, ``t``, ``tn`` for every real soil node.
    :param dt: Timestep (s), for the vapor-flux back-sweep.
    :return: None.
    """
    v = state["v"]; w = state["w"]; wn = state["wn"]
    t = state["t"]; tn = state["tn"]; u = state["u"]
    air_por = state["AirPor"]; m = state["m"]

    u[m] = 0.0
    for i in range(m, 0, -1):
        r_gvol = (_CAMPBELL_WATER_DENSITY * v[i] * _CAMPBELL_GAS_CONSTANT * (tn[i] + 273.0)
                  * (w[i] - wn[i])
                  / (dt * air_por[i] * _CAMPBELL_WATER_MOLAR_MASS * _CAMPBELL_ATMOSPHERIC_PRESSURE))
        if r_gvol < 0.0:
            r_gvol = 0.0
        u[i - 1] = u[i] + r_gvol
    for i in range(1, _CAMPBELL_DEEP_NODE + 1):
        r_ch = wn[i] - w[i]
        w[i] = wn[i]
        wn[i] = w[i] + r_ch
        r_ch = tn[i] - t[i]
        t[i] = tn[i]
        tn[i] = t[i] + r_ch


def _campbell_newton_boundary_init(state: dict) -> tuple:
    """
    One Newton sub-iteration's boundary/node-1 preparation: derive node 0
    (ambient air)/node 1 (surface) quantities needed before the per-node
    residual/Jacobian sweep, and snapshot node 1's pre-update state for
    the surface-update diagnostic hook. A direct extraction of
    ``soiltemp_step()``'s own per-sub-iteration boundary setup
    (``fof_soi.cpp:96-108``), moved verbatim (same statements, same
    order) out of :func:`_soiltemp_step` -- not re-derived.

    :param state: State dict from :func:`_soiltemp_initconsts` (and
        typically :func:`_soiltemp_initprofile`). Mutated in place: sets
        ``ke[0]``, ``kev[0]``, ``psat[0]``, ``h[0]``, ``psat[1]``,
        ``s[1]``, ``hvap[1]``, ``kh[1]``, ``enh[1]``, ``air_por[1]``,
        ``kv[1]``.
    :return: ``(surf_old_tn1, surf_old_p1, surf_old_wn1, surf_old_h1)`` --
        node 1's temperature/matric-potential/water-content/humidity
        BEFORE this sub-iteration's own update, forwarded to
        :func:`_campbell_newton_node_update`'s diagnostic hook.
    """
    tn = state["tn"]; wn = state["wn"]; p = state["p"]; h = state["h"]
    psat = state["psat"]; s = state["s"]; hvap = state["Hvap"]
    kh = state["kh"]; enh = state["enh"]; air_por = state["AirPor"]
    kv = state["kv"]; kev = state["kev"]; ke = state["ke"]
    xs = state["xs"]; xws = state["xws"]
    ls = state["ls"]; ga = state["ga"]; xwo = state["xwo"]; cop = state["cop"]
    t = state["t"]

    ke[0] = 0.0
    kev[0] = 6.2e-9 * _CAMPBELL_BOUNDARY_RESISTANCE
    psat[0] = _campbell_vapor_pressure(tn[0])
    h[0] = _CAMPBELL_AIR_VAPOR_PRESSURE / psat[0]
    psat[1] = _campbell_vapor_pressure(tn[1])
    r_wav = 0.5 * (wn[1] + wn[2])
    s[1] = _campbell_vapor_pressure_slope(tn[1], psat[1])
    hvap[1] = _campbell_latent_heat(tn[1])
    kh[1], enh[1] = _campbell_thermal_conductivity(
        tn[1], r_wav, xs, ls, ga, xwo, cop, h[1] * psat[1], s[1],
    )
    air_por[1] = xws - r_wav
    kv[1] = enh[1] * air_por[1] * _CAMPBELL_SOIL_TORTUOSITY * _campbell_vapor_conductivity(t[1], psat[1] * h[1])

    # Snapshot node 1 before this sub-iteration's update, matching the C++
    # overlay's SoiSurfaceUpdateDiag fields (fof_soi_instr.cpp,
    # `diag_old_tn1` et al.). Nothing above this point writes
    # tn[1]/p[1]/wn[1]/h[1], so this is a pure read.
    return float(tn[1]), float(p[1]), float(wn[1]), float(h[1])


def _campbell_newton_node_update(
        state: dict, i: int, r_rabs: float, dt: float, n_bug: int,
        on_surface_update, surf_old: tuple,
) -> tuple:
    """
    One node's residual/Jacobian construction, boundary (Stefan-Boltzmann)
    correction, and Newton temperature/matric-potential update for one
    Newton sub-iteration. A direct extraction of ``soiltemp_step()``'s
    own per-node loop body (``fof_soi.cpp:109-174``), moved verbatim
    (same statements, same order) out of :func:`_soiltemp_step` -- not
    re-derived. Mutates *state* in place: node *i*'s ``tn``/``p``/
    ``wn``/``dwdp``/``h``/``dhdp``, and every node-derived conductivity/
    Jacobian array this node and its neighbor touch (``cp``, ``psat``,
    ``conv``, ``vcon``, ``s``, ``hvap``, ``kh``, ``enh``, ``ke``,
    ``air_por``, ``kv``, ``kev``).

    ``r_sev_running``/``r_seh_running`` in the diagnostic payload equal
    this node's OWN ``abs(d_v)``/``abs(d_c)`` exactly, not a true running
    sum across nodes: the hook only ever fires for ``i == 1``, the first
    node the per-timestep loop visits, so the caller's accumulator has
    not yet added any other node's contribution at that point --
    unchanged from :func:`_soiltemp_step`'s original single-function
    behavior, just derived locally instead of read from the caller.

    :param state: State dict from :func:`_soiltemp_initconsts`.
    :param i: Node index, ``1..state['m']``.
    :param r_rabs: Total absorbed surface radiation (W/m^2) for this
        timestep -- only used in the ``i == 1`` boundary correction.
    :param dt: Timestep (s).
    :param n_bug: Sub-iteration count BEFORE this sub-iteration completes
        (0-based) -- forwarded to *on_surface_update* as ``n_bug + 1``,
        matching :func:`_soiltemp_step`'s own convention.
    :param on_surface_update: Diagnostic-only observer (see
        :func:`_soiltemp_step`'s docstring); called only when ``i == 1``
        and not ``None``.
    :param surf_old: ``(surf_old_tn1, surf_old_p1, surf_old_wn1,
        surf_old_h1)`` from :func:`_campbell_newton_boundary_init`.
    :return: ``(abs(d_v), abs(d_c))`` -- this node's water/energy
        residual magnitudes, for the caller's ``r_sev``/``r_seh``
        accumulation.
    """
    surf_old_tn1, surf_old_p1, surf_old_wn1, surf_old_h1 = surf_old

    z = state["z"]; v = state["v"]; w = state["w"]; wn = state["wn"]
    t = state["t"]; tn = state["tn"]; p = state["p"]; dwdp = state["dwdp"]
    h = state["h"]; dhdp = state["dhdp"]; psat = state["psat"]
    kev = state["kev"]; hvap = state["Hvap"]
    s = state["s"]; ke = state["ke"]; kh = state["kh"]; kv = state["kv"]
    cp = state["cp"]; conv = state["conv"]; vcon = state["vcon"]
    enh = state["enh"]; air_por = state["AirPor"]
    xs = state["xs"]; xws = state["xws"]; bd = state["bd"]
    ls = state["ls"]; ga = state["ga"]; xwo = state["xwo"]
    cop = state["cop"]; xo = state["xo"]; m = state["m"]; u = state["u"]

    cp[i] = v[i] * (0.87 * bd + 4.18e6 * wn[i]) / dt
    psat[i + 1] = _campbell_vapor_pressure(tn[i + 1])
    if i < m:
        r_wav = 0.5 * (wn[i + 1] + wn[i + 2])
        r_tav = 0.5 * (tn[i + 1] + tn[i + 2]) + 273.0
    else:
        r_wav = wn[_CAMPBELL_DEEP_NODE]
        r_tav = tn[i + 1] + 273.0
    conv[i] = 0.5 * (u[i - 1] + u[i]) * 1200.0 * 293.0 / r_tav
    vcon[i] = conv[i] * _CAMPBELL_WATER_MOLAR_MASS / (_CAMPBELL_GAS_CONSTANT * 1200.0 * 293.0)
    s[i + 1] = _campbell_vapor_pressure_slope(tn[i + 1], psat[i + 1])
    hvap[i + 1] = _campbell_latent_heat(tn[i + 1])
    kh[i + 1], enh[i + 1] = _campbell_thermal_conductivity(
        tn[i + 1], r_wav, xs, ls, ga, xwo, cop,
        h[i + 1] * psat[i + 1], s[i + 1],
    )
    ke[i] = kh[i] / (z[i + 1] - z[i]) + conv[i]
    air_por[i + 1] = xws - r_wav
    kv[i + 1] = (enh[i + 1] * air_por[i + 1] * _CAMPBELL_SOIL_TORTUOSITY
                 * _campbell_vapor_conductivity(t[i + 1], psat[i + 1] * h[i + 1]))
    kev[i] = (kv[i] + kv[i + 1]) / (2.0 * (z[i + 1] - z[i])) + vcon[i]

    d_jv = (kev[i - 1] * (psat[i] * h[i] - psat[i - 1] * h[i - 1])
            - kev[i] * (psat[i + 1] * h[i + 1] - psat[i] * h[i]))
    d_jvdt = s[i] * h[i] * (kev[i - 1] + kev[i])
    d_jvdp = psat[i] * (kev[i - 1] + kev[i]) * dhdp[i]

    d_c = (ke[i - 1] * (tn[i] - tn[i - 1])
           - ke[i] * (tn[i + 1] - tn[i])
           + cp[i] * (tn[i] - t[i])
           - hvap[i] * _CAMPBELL_WATER_DENSITY * v[i] * (wn[i] - w[i]) / dt)
    d_v = d_jv + _CAMPBELL_WATER_DENSITY * v[i] * (wn[i] - w[i]) / dt
    d_cdp = -hvap[i] * _CAMPBELL_WATER_DENSITY * v[i] * dwdp[i] / dt
    d_vdp = d_jvdp + _CAMPBELL_WATER_DENSITY * v[i] * dwdp[i] / dt
    d_vdt = d_jvdt
    d_cdt = ke[i] + ke[i - 1] + cp[i]

    # Snapshot the residual/Jacobian pair before the boundary correction,
    # matching the C++ overlay's diag_dC_before/diag_dCdt_before exactly.
    surf_dC_before = d_c
    surf_dCdt_before = d_cdt
    surf_stefan_term = 0.0
    surf_tk_old = 0.0
    if i == 1:
        r_tk = tn[1] + 273.0
        r_tk3 = r_tk * r_tk * r_tk
        surf_stefan_term = 5.67e-8 * r_tk * r_tk3
        surf_tk_old = r_tk
        d_c = d_c - r_rabs + 5.67e-8 * r_tk * r_tk3
        d_cdt = d_cdt + 4.0 * 5.67e-8 * r_tk3

    d_v_abs = abs(d_v)
    d_c_abs = abs(d_c)

    d_tn = (d_v * d_cdp - d_c * d_vdp) / (d_cdp * d_vdt - d_cdt * d_vdp)
    # Preserve the raw Newton temperature increment before the <-100 clamp,
    # matching diag_dtn_temp_raw/_clamped.
    surf_dtn_temp_raw = d_tn
    surf_dtn_temp_clamped = 1.0 if d_tn < -100.0 else 0.0
    if d_tn < -100.0:
        d_tn = -100.0
    tn[i] = tn[i] - d_tn
    d_tn = (d_v - d_vdt * d_tn) / d_vdp
    surf_dtn_matric_raw = d_tn
    p[i] = p[i] - d_tn
    surf_p_before_range_clamp = p[i]
    surf_p_clamp_branch = 0.0
    if p[i] > 0.0:
        p[i] = (p[i] + d_tn) * 0.5
        surf_p_clamp_branch = 1.0
    if p[i] < -1e20:
        p[i] = -1e20
        surf_p_clamp_branch = 2.0
    wn[i], dwdp[i] = _campbell_water_content(p[i], xo)
    h[i], dhdp[i] = _campbell_humidity(p[i], tn[i])

    if i == 1 and on_surface_update is not None:
        # Fire once per Newton sub-iteration, matching the C++ overlay's
        # SoiDiagRecordSurfaceUpdate hook (fof_soi_instr.cpp)
        # field-for-field -- keys match its CSV column names
        # exactly (SOIL_SURFUP_DIAG_COLUMNS in test_harness.cpp).
        on_surface_update(n_bug + 1, {
            "old_tn1": surf_old_tn1, "new_tn1": float(tn[1]),
            "old_p1": surf_old_p1, "new_p1": float(p[1]),
            "old_wn1": surf_old_wn1, "new_wn1": float(wn[1]),
            "old_h1": surf_old_h1, "new_h1": float(h[1]),
            "psat0": float(psat[0]), "h0": float(h[0]),
            "psat1": float(psat[1]),
            "psat2": float(psat[2]), "h2": float(h[2]),
            "s1": float(s[1]), "hvap1": float(hvap[1]),
            "kh1": float(kh[1]), "enh1": float(enh[1]), "kv1": float(kv[1]),
            "kh2": float(kh[2]), "kv2": float(kv[2]),
            "ke0": float(ke[0]), "ke1": float(ke[1]),
            "kev0": float(kev[0]), "kev1": float(kev[1]),
            "conv1": float(conv[1]), "vcon1": float(vcon[1]), "cp1": float(cp[1]),
            "d_jv": float(d_jv), "d_jvdt": float(d_jvdt), "d_jvdp": float(d_jvdp),
            "dC_before_boundary": float(surf_dC_before),
            "dC_after_boundary": float(d_c),
            "dv": float(d_v),
            "dCdp": float(d_cdp), "dvdp": float(d_vdp),
            "dCdt_before_boundary": float(surf_dCdt_before),
            "dCdt_after_boundary": float(d_cdt),
            "dvdt": float(d_vdt),
            "r_rabs_in": float(r_rabs),
            "stefan_term": float(surf_stefan_term),
            "tk_old": float(surf_tk_old),
            "dtn_temperature_raw": float(surf_dtn_temp_raw),
            "dtn_temperature_clamped": float(surf_dtn_temp_clamped),
            "dtn_matric_raw": float(surf_dtn_matric_raw),
            "p1_before_range_clamp": float(surf_p_before_range_clamp),
            "p1_clamp_branch": float(surf_p_clamp_branch),
            "r_sev_running": float(d_v_abs), "r_seh_running": float(d_c_abs),
        })

    return d_v_abs, d_c_abs


def _campbell_ambient_radiation(start_temp: float) -> float:
    """
    Constant ambient radiative floor added to absorbed radiation for BOTH
    soil routes (``fof_sd.cpp:76``, ``fof_se.cpp:82``): a Stefan-Boltzmann
    term evaluated ONCE at the starting temperature and held fixed for
    the whole simulation. This is NOT re-evaluated at the current surface
    temperature — the corresponding OUTGOING loss term (based on the
    CURRENT, time-varying surface temperature) is computed separately,
    inside :func:`_soiltemp_step` itself.

    :param start_temp: Starting soil temperature (degC).
    :return: Ambient radiative term (W/m^2).
    """
    tk = start_temp + 273.0   # NOT 273.15 -- matches the pinned literal exactly
    return 5.67e-8 * (tk ** 4)


def _campbell_fire_intensity(clock_sec: float, series, inc_s: float = 15.0) -> float:
    """
    Port of C++ ``_Get_FirInt()`` (``fof_se.cpp:234-246``): a ZERO-ORDER
    HOLD sample from a fixed-increment array via INTEGER index division —
    NOT linear interpolation (a real divergence source the prior
    ``interp1d``-based non-duff flux construction did not replicate).

    :param clock_sec: Current simulation clock (s).
    :param series: Fire-intensity samples (kW/m^2), *inc_s* apart.
    :param inc_s: Seconds between samples. ALWAYS 15.0 in the pinned
        C++ — hardcoded by ``SH_Mngr``'s own call into
        ``SE_Mngr_Array`` (``fof_sh.cpp:67``, literal ``15``), never
        derived from the caller's own times array.
    :return: The sampled intensity (kW/m^2), or -1.0 past the array end
        (matching C++'s own end-of-array sentinel).
    """
    i = int(clock_sec // inc_s)
    if i >= len(series):
        return -1.0
    return float(series[i])


def _campbell_humidity(p: float, t: float) -> tuple:
    """
    Port of C++ ``humidity()`` (``fof_soi.cpp:346-354``): relative
    humidity of soil pore air as a function of matric potential *p* and
    temperature *t* (Kelvin equation).

    :param p: Matric potential (J/kg).
    :param t: Temperature (degC).
    :return: ``(relative_humidity, dhdp)``.
    """
    tk = t + _CAMPBELL_STANDARD_TEMPERATURE
    h = math.exp(_CAMPBELL_WATER_MOLAR_MASS * p / (_CAMPBELL_GAS_CONSTANT * tk))
    dhdp = _CAMPBELL_WATER_MOLAR_MASS * h / (_CAMPBELL_GAS_CONSTANT * tk)
    return h, dhdp


def _campbell_latent_heat(t: float) -> float:
    """
    Port of C++ ``Hv()`` (``fof_soi.cpp:394-397``): latent heat of
    vaporization (J/kg).

    :param t: Temperature (degC).
    :return: Latent heat of vaporization (J/kg).
    """
    return 2.508e6 - 2670.0 * t


def _campbell_vapor_conductivity(t: float, p: float) -> float:
    """
    Port of C++ ``Kvap()`` (``fof_soi.cpp:403-415``): vapor conductivity
    (kg/(m s Pa)).

    :param t: Temperature (degC).
    :param p: Pressure (Pa).
    :return: Vapor conductivity (kg/(m s Pa)).
    """
    tk = t + _CAMPBELL_STANDARD_TEMPERATURE
    f = _campbell_signed_power(tk / _CAMPBELL_STANDARD_TEMPERATURE, 1.75)
    g = _CAMPBELL_STANDARD_PRESSURE / _CAMPBELL_ATMOSPHERIC_PRESSURE
    dv = _CAMPBELL_VAPOR_DIFFUSIVITY * g * f
    stcor = 1.0 - p / _CAMPBELL_ATMOSPHERIC_PRESSURE
    if stcor < 0.3:
        stcor = 0.3
    return _CAMPBELL_WATER_MOLAR_MASS * dv / (_CAMPBELL_GAS_CONSTANT * tk * stcor)


def _campbell_vapor_pressure_slope(t: float, p: float) -> float:
    """
    Port of C++ ``slope()`` (``fof_soi.cpp:380-388``): d(vapor
    pressure)/dT at temperature *t*, vapor pressure *p*.

    :param t: Temperature (degC).
    :param p: Vapor pressure (Pa).
    :return: Slope (Pa/degC).
    """
    tk = t + _CAMPBELL_STANDARD_TEMPERATURE
    tt = 1.0 - 373.15 / tk
    dydt = 373.15 / (tk * tk)
    return p * dydt * (13.3015 + tt * (-4.082 + tt * (0.78 + tt * 10.76)))


def _campbell_soil_depths_mm(depth_layers: list) -> list:
    """
    Build the 15-node depth array (mm), matching C++'s fixed
    ``rr_Lay``/``SH_Init_LayDis`` layer scheme (``fof_sh.cpp:162-163``)
    EXACTLY when *depth_layers* == ``[1, 2, ..., 13]`` (cm) — the only
    depth scheme the pinned C++ ever actually uses; it is not
    user-configurable in C++ at all. Index 0 and 1 are both 0 mm (index 0
    is the virtual air/BC node; index 1 is the true first solved soil
    node, the surface); indices 2-14 are *depth_layers* converted cm ->
    mm.

    :param depth_layers: Exactly 13 depths (cm).
    :return: 15 depths (mm).
    """
    return [0.0, 0.0] + [float(d) * 10.0 for d in depth_layers]


def _campbell_signed_power(x: float, y: float) -> float:
    """
    Port of C++ ``sPOW()`` (``fof_soi.cpp:425-439``): ``x`` raised to
    ``y``, treating ``x`` as its absolute value (the Pascal-derived
    original), returning 0.0 for ``x == 0`` rather than raising.

    :param x: Base.
    :param y: Exponent.
    :return: ``abs(x) ** y``, or 0.0 if ``x == 0``.
    """
    x = abs(x)
    if x == 0.0:
        return 0.0
    return math.exp(y * math.log(x))


def _campbell_thermal_conductivity(
        t: float, xw: float, xs: float, ls: float, ga: float,
        xwo: float, cop: float, p: float, s: float,
) -> tuple:
    """
    Port of C++ ``tcond()`` (``fof_soi.cpp:218-255``): effective thermal
    conductivity of the soil (de Vries-type mixing model with a
    moisture-dependent liquid-recirculation enhancement), plus the vapor
    conductivity enhancement factor.

    :param t: Temperature (degC).
    :param xw: Volumetric liquid water content (m^3/m^3).
    :param xs: Solid (mineral) volume fraction.
    :param ls: Thermal conductivity of the mineral fraction (W/m/K).
    :param ga: de Vries shape factor.
    :param xwo: Water content for liquid recirculation (m^3/m^3).
    :param cop: Power for the liquid-recirculation function.
    :param p: Vapor pressure of the pore air (Pa) — C++'s
        ``h[i]*psat[i]``.
    :param s: Slope of the vapor-pressure curve (Pa/degC), from
        :func:`_campbell_vapor_pressure_slope`.
    :return: ``(thermal_conductivity, vapor_enhancement_factor)``.
    """
    xws = 1.0 - xs
    xa = xws - xw
    if t < 100.0:
        lw = 0.554 + t * (2.24e-3 - 9.87e-6 * t)
        tc = ((t + 273.0) / 303.0) ** 4
    else:
        lw = 0.68
        tc = 2.3
    lda = 0.024 + t * (7.73e-5 - 2.6e-8 * t)
    if xw < 0.01 * xwo:
        wf = 0.0
    else:
        wf = 1.0 / (1.0 + _campbell_signed_power(xw / xwo, -cop * tc))
    a = _campbell_vapor_conductivity(t, p)
    b = _campbell_latent_heat(t)
    c = wf * b * s * a
    la = lda + c
    gc = 1.0 - 2.0 * ga
    lf = la + (lw - la) * wf
    ka = (2.0 / (1.0 + (la / lf - 1.0) * ga) + 1.0 / (1.0 + (la / lf - 1.0) * gc)) / 3.0
    kw = (2.0 / (1.0 + (lw / lf - 1.0) * ga) + 1.0 / (1.0 + (lw / lf - 1.0) * gc)) / 3.0
    ks = (2.0 / (1.0 + (ls / lf - 1.0) * ga) + 1.0 / (1.0 + (ls / lf - 1.0) * gc)) / 3.0
    enh = (1.0 + 2.0 * wf) * ka
    tc_out = (kw * lw * xw + ka * la * xa + ks * ls * xs) / (kw * xw + ka * xa + ks * xs)
    if 0.0 < tc_out < 5.0:
        return tc_out, enh
    return 1.0, enh


def _campbell_vapor_pressure(tn: float) -> float:
    """
    Port of C++ ``vaporpressure()`` (``fof_soi.cpp:361-374``).

    :param tn: Temperature (degC, but scaled by 1000 per the source's own
        ``r_t = r_tin * 1000`` — a Pascal-derived idiosyncrasy preserved
        exactly, not corrected).
    :return: Vapor pressure (Pa).
    """
    t = tn * 1000.0
    r0 = t + 273150.0
    r1 = 373150.0 / r0
    t2 = 1.0 - r1
    r1b = t2 * (13.3016 + t2 * (-2.042 + t2 * (0.26 + t2 * 2.69)))
    return 101325.0 * math.exp(r1b)


def _campbell_water_content(p: float, xo: float) -> tuple:
    """
    Port of C++ ``watercontent()`` (``fof_soi.cpp:325-340``): volumetric
    water content as a function of matric potential *p*.

    :param p: Matric potential (J/kg, always negative in practice — a
        non-negative input is clamped to -0.001, matching C++ exactly).
    :param xo: Extrapolated water content at -1 J/kg (m^3/m^3).
    :return: ``(water_content, dwdp)``.
    """
    if p >= 0.0:
        p = -0.001
    dwdp = -xo / (13.82 * p)
    w = xo * (1.0 - math.log(-p) / 13.82)
    return w, dwdp


def _duff_burn_profile(duff_params: dict) -> dict:
    """
    Build the Campbell duff-forcing time profile from C++'s
    ``DuffBurn()``/``SD_HeatAdj()`` contract (``bur_brn.cpp:1950-1986``,
    ``fof_sd.cpp:98-129,294-313``).

    Unlike the pre-correction implementation, *duff_load* is a genuine,
    required input (matching ``TPA_To_KiSq(duff_load)`` -> C++'s ``wdf``),
    and the duff-to-soil heat-transmission fraction is evaluated at the
    TIME-VARYING remaining duff depth — linearly interpolated from the
    pre-fire depth down to the implied post-fire depth over the burn
    duration (matching C++'s own ``slice``-based trajectory,
    ``fof_sd.cpp:109-110,125-126``) — never a single static pre-fire-depth
    value held constant for the whole simulation.

    ``duff_heat_content`` is intentionally NOT used: C++'s intensity
    formula (``bur_brn.cpp:1963``) is a self-contained empirical function of
    moisture alone, with no heat-content input at all — multiplying by a
    heat-content term (as the pre-correction implementation did) has no C++
    counterpart. A caller may still pass it (accepted, silently unused) for
    backward compatibility with existing call sites.

    :param duff_params: Dict with ``'duff_load'`` (T/ac, REQUIRED),
        ``'duff_depth'`` (in), ``'duff_moisture'`` (%, whole-percent
        convention), ``'pct_consumed'`` (%, required in ``[0, 100]``), and
        optional ``'efficiency_duff'`` (proportion, default 1.0).
    :return: Dict with:

        * ``'duration_s'`` — burn duration (s); ``0.0`` for a zero-forcing
          case.
        * ``'intensity_w'`` — surface fire intensity (W/m²), C++'s
          ``f_Watts = d_kW * 1000``.
        * ``'consumed_rate_kgm2_s'`` — duff dry-mass consumption rate
          (kg/m²/s) while burning.
        * ``'pre_depth_cm'`` / ``'post_depth_cm'`` — duff depth (cm) at
          ``t=0`` and at (or after) ``t=duration_s``.
        * ``'remaining_depth_fn'`` — callable ``t (s) -> cm``, linear from
          ``pre_depth_cm`` to ``post_depth_cm`` over ``[0, duration_s]``,
          constant at ``post_depth_cm`` afterward.
        * ``'heat_fraction_fn'`` — callable ``t (s) -> fraction [0, 1]``,
          :func:`_duff_heat_fraction` evaluated at
          ``remaining_depth_fn(t)``.
        * ``'flux_fn'`` — callable ``t (s) -> W/m²``, the Campbell surface-
          flux boundary condition: ``intensity_w * heat_fraction_fn(t) *
          efficiency_duff`` while ``0 <= t < duration_s``, else ``0.0``.
    :raises ValueError: If ``duff_load`` is missing or non-finite (a
        present, finite value ``<= 0`` — including negative — is valid and
        yields zero forcing, matching C++); if ``duff_depth`` or
        ``duff_moisture`` is missing/non-finite/negative; or if
        ``pct_consumed`` is missing, non-finite, or outside ``[0, 100]``
        (C++'s own fallback for an out-of-range percent,
        ``bur_brn.cpp:1974-1975``, exists only for a standalone-Burnup-
        without-FOFEM code path this Campbell contract has no equivalent
        of, so it is rejected rather than silently ported).
    """
    if duff_params.get("duff_load") is None:
        raise ValueError(
            "duff_params['duff_load'] (T/ac) is required for "
            "soil_heat_campbell(model='duff')."
        )
    duff_load_tac = float(duff_params["duff_load"])
    if not np.isfinite(duff_load_tac):
        raise ValueError(
            f"duff_params['duff_load'] must be finite, got {duff_load_tac!r}."
        )

    duff_depth_in = float(duff_params["duff_depth"])
    if not np.isfinite(duff_depth_in) or duff_depth_in < 0.0:
        raise ValueError(
            "duff_params['duff_depth'] must be finite and >= 0, got "
            f"{duff_depth_in!r}."
        )

    duff_moisture_pct = float(duff_params["duff_moisture"])
    if not np.isfinite(duff_moisture_pct) or duff_moisture_pct < 0.0:
        raise ValueError(
            "duff_params['duff_moisture'] must be finite and >= 0, got "
            f"{duff_moisture_pct!r}."
        )

    pct_consumed = float(duff_params["pct_consumed"])
    if not np.isfinite(pct_consumed) or not (0.0 <= pct_consumed <= 100.0):
        raise ValueError(
            "duff_params['pct_consumed'] must be finite and in [0, 100], "
            f"got {pct_consumed!r}. C++'s own fallback for an out-of-range "
            "value (bur_brn.cpp:1974-1975) exists only for a standalone-"
            "Burnup-without-FOFEM code path with no Campbell equivalent "
            "here."
        )

    efficiency_duff = duff_params.get("efficiency_duff", 1.0)

    wdf_kgm2 = duff_load_tac * _TONS_ACRE_TO_KG_M2
    dfm_ratio = duff_moisture_pct / 100.0  # Convert percent to ratio once at this boundary.

    intensity_kw, duration_s, consumed_rate = _duff_burn_rate(
        wdf_kgm2, dfm_ratio, pct_consumed,
    )
    intensity_w = intensity_kw * 1000.0

    pre_depth_cm = duff_depth_in * _INCH_TO_CM
    if duration_s > 0.0:
        post_depth_cm = pre_depth_cm * (1.0 - pct_consumed / 100.0)
    else:
        post_depth_cm = pre_depth_cm  # nothing burns; depth never changes

    def remaining_depth_fn(t: float) -> float:
        """
        Remaining (currently standing) duff depth at time *t*.

        :param t: Simulation time (s).
        :return: Duff depth (cm), linearly interpolated from
            *pre_depth_cm* to *post_depth_cm* over ``[0, duration_s]``.
        """
        if duration_s <= 0.0:
            return pre_depth_cm
        frac = min(max(t, 0.0), duration_s) / duration_s
        return pre_depth_cm + (post_depth_cm - pre_depth_cm) * frac

    def heat_fraction_fn(t: float) -> float:
        """
        Fraction of surface heat transmitted through the duff at time *t*.

        :param t: Simulation time (s).
        :return: Fraction (0.0-1.0).
        """
        return _duff_heat_fraction(remaining_depth_fn(t))

    def flux_fn(t: float) -> float:
        """
        Campbell surface heat flux boundary condition at time *t*.

        :param t: Simulation time (s).
        :return: Surface heat flux (W/m²); ``0.0`` when no burn occurs or
            once ``t >= duration_s``.
        """
        # SD_Mngr_New() applies duff heat only while i_ClockSec < i_BurnTime
        # (fof_sd.cpp:123-131). The strict boundary also ensures a 0%-
        # consumed case (duration_s == 0) has no t=0 impulse in solve_ivp.
        if duration_s <= 0.0 or t >= duration_s:
            return 0.0
        return intensity_w * heat_fraction_fn(t) * efficiency_duff

    return {
        "duration_s": duration_s,
        "intensity_w": intensity_w,
        "consumed_rate_kgm2_s": consumed_rate,
        "pre_depth_cm": pre_depth_cm,
        "post_depth_cm": post_depth_cm,
        "remaining_depth_fn": remaining_depth_fn,
        "heat_fraction_fn": heat_fraction_fn,
        "flux_fn": flux_fn,
    }


def _duff_burn_rate(
        wdf_kgm2: float, dfm_ratio: float, pct_consumed: float,
) -> tuple:
    """
    Direct port of C++ ``DuffBurn()`` (``bur_brn.cpp:1950-1986``).

    :param wdf_kgm2: Duff dry load (kg/m²) — C++ ``wdf``. Any value ``<= 0``
        (including negative) yields all-zero output, matching C++'s own
        ``if (wdf <= 0.0 ...) return;`` guard exactly — not an error case.
    :param dfm_ratio: Duff moisture as a ratio (0 to ~1.96, not a percent)
        — C++ ``dfm``.
    :param pct_consumed: Duff consumed, percent, REQUIRED to already be in
        ``[0, 100]`` (validated by the caller, :func:`_duff_burn_profile`;
        this function does not re-validate it).
    :return: ``(intensity_kw, duration_s, consumed_rate_kgm2_s)`` — C++
        ``dfi``, ``tdf``, ``ad_Duf_CPTS``. All three are ``0.0`` when
        ``wdf_kgm2 <= 0`` or ``dfm_ratio >= 1.96``
        (``bur_brn.cpp:1960-1961``).
    """
    if wdf_kgm2 <= 0.0 or dfm_ratio >= _DUFF_BURN_MOISTURE_RATIO_MAX:
        return 0.0, 0.0, 0.0

    intensity_kw = 11.25 - 4.05 * dfm_ratio  # bur_brn.cpp:1963

    consumed_fraction = pct_consumed / 100.0  # bur_brn.cpp:1972-1973
    denom = 7.5 - 2.7 * dfm_ratio
    duration_s = (
        1.0e4 * consumed_fraction * wdf_kgm2 / denom if denom != 0.0 else 0.0
    )  # bur_brn.cpp:1978

    if duration_s == 0.0:
        consumed_rate = 0.0
    else:
        consumed_rate = (consumed_fraction * wdf_kgm2) / duration_s  # bur_brn.cpp:1982-1985

    return intensity_kw, duration_s, consumed_rate


def _duff_heat_fraction(remaining_depth_cm: float) -> float:
    """
    Direct port of C++ ``SD_HeatAdj()`` (``fof_sd.cpp:294-313``): the
    fraction of surface heat transmitted through a duff layer of the given
    remaining depth.

    :param remaining_depth_cm: Remaining (currently standing) duff depth,
        centimetres.
    :return: Fraction (0.0-1.0) of surface heat transmitted through the
        duff to the mineral soil.
    """
    y0, a, b, c, d = -1.6996, 32.7652, 7.4601, 68.9349, 0.6077
    r = (
        y0 + a * np.exp(-b * remaining_depth_cm)
        + c * np.exp(-d * remaining_depth_cm)
    )
    r = float(np.clip(r, 0.0, 100.0))
    return r * 0.01


def _make_duff_forcing_fn(duff_profile: dict, ambient_rabs: float):
    """
    Build the duff-route ``forcing_fn`` for :func:`_run_coupled_soil_sim`,
    matching C++ ``SD_Mngr_New``'s per-tick absorbed-radiation computation
    (``fof_sd.cpp:122-139``): total absorbed radiation is the constant
    ambient term PLUS the (already time-varying, efficiency-scaled, and
    zero-past-duration) duff-forcing flux from :func:`_duff_burn_profile`.

    :param duff_profile: Dict from :func:`_duff_burn_profile`.
    :param ambient_rabs: Constant ambient radiative term (W/m^2), from
        :func:`_campbell_ambient_radiation`.
    :return: Callable ``clock_sec -> (r_rabsub, is_still_burning)``.
    """
    duration_s = duff_profile["duration_s"]
    flux_fn = duff_profile["flux_fn"]

    def forcing_fn(clock_sec: float):
        """
        Evaluate total absorbed radiation and burn status at *clock_sec*.

        :param clock_sec: Current simulation clock (s).
        :return: ``(r_rabsub, is_still_burning)``.
        """
        still_burning = clock_sec < duration_s
        return ambient_rabs + flux_fn(clock_sec), still_burning

    return forcing_fn


def _make_nonduff_forcing_fn(
        wl_series, hs_series, wl_eff: float, hs_eff: float, ambient_rabs: float,
):
    """
    Build the non-duff-route ``forcing_fn`` for
    :func:`_run_coupled_soil_sim`, matching C++ ``SE_Mngr_Array``'s
    per-tick absorbed-radiation computation (``fof_se.cpp:107-121``): a
    zero-order-hold sample (:func:`_campbell_fire_intensity`) of each of
    the wood-litter and herb-shrub fire-intensity series, scaled by
    their respective efficiencies, plus the constant ambient term.

    :param wl_series: Wood-litter fire intensity (kW/m^2), 15 s apart.
    :param hs_series: Herb-shrub fire intensity (kW/m^2), 15 s apart.
    :param wl_eff: Wood-litter delivery efficiency (proportion).
    :param hs_eff: Herb-shrub delivery efficiency (proportion).
    :param ambient_rabs: Constant ambient radiative term (W/m^2), from
        :func:`_campbell_ambient_radiation`.
    :return: Callable ``clock_sec -> (r_rabsub, fi_now)`` — *fi_now* is
        the combined WL+HS intensity EXCLUDING the ambient term, for
        :func:`_soi_done_nonduff`.
    """
    def forcing_fn(clock_sec: float):
        """
        Evaluate total absorbed radiation and combined fire intensity at
        *clock_sec*.

        :param clock_sec: Current simulation clock (s).
        :return: ``(r_rabsub, fi_now)``.
        """
        f_wl = _campbell_fire_intensity(clock_sec, wl_series)
        if f_wl < 0.0:
            f_wl = 0.0
        f_wl = f_wl * 1000.0 * wl_eff
        f_hs = _campbell_fire_intensity(clock_sec, hs_series)
        if f_hs < 0.0:
            f_hs = 0.0
        f_hs = f_hs * 1000.0 * hs_eff
        fi_now = f_wl + f_hs
        return fi_now + ambient_rabs, fi_now

    return forcing_fn


def _make_guide_herb_shrub_intensity(
        consumed_load: float,
        heat_content: float = 1.86e7,
) -> list[float]:
    """Build the FOFEM 6.9.4 herb/shrub intensity profile.

    The guide specifies a maximum first-minute consumption of 5 T/ac.  Its
    four 15-second intervals consume 10, 20, 30, and 40 percent of that
    first-minute amount; any remaining consumed herb/shrub fuel then burns at
    a constant 5 T/ac/min rate in 15-second intervals.

    :param consumed_load: Total consumed herb plus shrub load (kg/m^2).
    :param heat_content: Low heat of combustion (J/kg).
    :returns: Herb/shrub fire-intensity samples (kW/m^2), 15 seconds apart.
    :raises ValueError: If either input is non-finite, or if *consumed_load*
        is negative or *heat_content* is not positive.
    """
    if not np.isfinite(consumed_load) or consumed_load < 0.0:
        raise ValueError("consumed_load must be a finite value >= 0.")
    if not np.isfinite(heat_content) or heat_content <= 0.0:
        raise ValueError("heat_content must be a finite value > 0.")
    if consumed_load == 0.0:
        return []

    interval_s = 15.0
    rate_kg_m2_s = 5.0 / 4.4609 / 60.0
    first_minute_load = min(consumed_load, rate_kg_m2_s * 60.0)
    interval_loads = [first_minute_load * fraction for fraction in (0.10, 0.20, 0.30, 0.40)]

    remaining_load = consumed_load - first_minute_load
    while remaining_load > 0.0:
        interval_load = min(rate_kg_m2_s * interval_s, remaining_load)
        interval_loads.append(interval_load)
        remaining_load -= interval_load

    return [heat_content * load / interval_s * 1.0e-3 for load in interval_loads]


def _run_coupled_soil_sim(state: dict, dt: float, forcing_fn, done_fn, start_temp: float) -> list:
    """
    Shared outer clock-driven loop — a direct port of the shared skeleton
    in C++'s ``SD_Mngr_New`` (``fof_sd.cpp:121-182``) and
    ``SE_Mngr_Array`` (``fof_se.cpp:104-163``): call *forcing_fn* for the
    current absorbed radiation, advance one Newton-converged timestep via
    :func:`_soiltemp_step`, record every real soil node's temperature,
    and repeat until *done_fn* signals completion.

    C++'s own per-step timestep-halving retry branch is unreachable:
    ``soiltemp_step()``'s own ``*ai_success`` output
    is unconditionally 1 on every non-hard-failure return, because the
    ``i_its`` variable that would trigger the ``0`` branch is never
    incremented anywhere in the function. So a hard failure (the Newton
    iteration not converging within
    :data:`_CAMPBELL_MAX_NEWTON_ITERATIONS` sub-iterations, or ``dt <= 0``) is
    propagated here exactly as C++ propagates it: as an immediate,
    fatal, non-retried simulation failure — see
    :class:`SoilSimulationError`.

    :param state: Solver state from :func:`_soiltemp_initconsts` /
        :func:`_soiltemp_initprofile`.
    :param dt: Fixed per-route timestep (s) — 20.0 for the duff route,
        10.0 for the non-duff route, matching the pinned C++ family-table
        defaults exactly (never user-configurable in C++).
    :param forcing_fn: Callable ``clock_sec -> (r_rabsub, done_signal)``.
    :param done_fn: Callable ``(temps, start_temp, clock_sec,
        done_signal) -> bool``.
    :param start_temp: Starting soil temperature (degC), forwarded to
        *done_fn* unchanged.
    :return: List of ``(time_s, temps)`` tuples, one per recorded step —
        *time_s* is ``step_index * dt`` (0-based), matching the
        established harness/golden ``time_index * SHA_GetInc()``
        convention exactly; *temps* is a length-14 array (index 0 =
        surface = C++ node 1, index 13 = deepest depth = C++ node 14).
    :raises SoilSimulationError: If the Newton iteration fails to
        converge for any timestep.
    """
    records = []
    clock_sec = 0.0
    step_index = 0
    for _ in range(_CAMPBELL_MAX_TIMESTEPS):
        r_rabsub, done_signal = forcing_fn(clock_sec)
        ok = _soiltemp_step(state, r_rabsub, dt)
        if not ok:
            raise SoilSimulationError(
                "Soil Simulation Failed: the coupled Newton iteration did "
                f"not converge within {_CAMPBELL_MAX_NEWTON_ITERATIONS} sub-iterations "
                f"at clock_sec={clock_sec}, dt={dt} -- matches the pinned "
                "C++ soiltemp_step()'s own fatal e_SoiSimFail path."
            )
        temps = np.array(state["tn"][1:_CAMPBELL_DEEP_NODE + 1], dtype=float)
        records.append((step_index * dt, temps))
        step_index += 1
        if done_fn(state["tn"], start_temp, clock_sec, done_signal):
            break
        clock_sec += dt
    return records


def _soi_done_duff(t_arr, start_temp: float, clock_sec: float, burn_time_s: float) -> bool:
    """
    Port of C++ ``SD_Mngr_New``'s ``_Done()`` (``fof_sd.cpp:201-217``):
    the duff-route simulation is complete once the burn has ended AND
    the top 5 layers have cooled back within 0.5 degC of the starting
    temperature.

    :param t_arr: Current temperature array (index 1-14 are the real
        soil nodes).
    :param start_temp: Starting soil temperature (degC).
    :param clock_sec: Current simulation clock (s), BEFORE this step's
        own increment (matching C++'s check ordering exactly).
    :param burn_time_s: Duff burn duration (s), from
        :func:`_duff_burn_profile`.
    :return: ``True`` if the simulation should stop.
    """
    if clock_sec <= burn_time_s:
        return False
    thresh = start_temp + 0.5
    for i in range(1, 6):
        if t_arr[i] > thresh:
            return False
    return True


def _soi_done_nonduff(t_arr, start_temp: float, clock_sec: float, fi_now: float) -> bool:
    """
    Port of C++ ``SE_Mngr_Array``'s ``_Done()`` (``fof_se.cpp:200-220``):
    the non-duff-route simulation is complete once at least 20 minutes
    have elapsed AND the combined wood-litter/herb-shrub fire intensity
    has dropped to zero AND the top 5 layers have cooled back within 0.5
    degC of the starting temperature.

    :param t_arr: Current temperature array (index 1-14 are the real
        soil nodes).
    :param start_temp: Starting soil temperature (degC).
    :param clock_sec: Current simulation clock (s), BEFORE this step's
        own increment.
    :param fi_now: Combined wood-litter + herb-shrub fire intensity
        (W/m^2) at *clock_sec*, EXCLUDING the ambient radiative floor —
        matches C++'s own ``f_FI`` (computed before ``r_Rabs`` is added).
    :return: ``True`` if the simulation should stop.
    """
    if clock_sec < 60.0 * 20.0:
        return False
    if fi_now > 0.0:
        return False
    thresh = start_temp + 0.5
    for i in range(1, 6):
        if t_arr[i] > thresh:
            return False
    return True


def _soiltemp_initconsts(
        bd: float, pd: float, ls: float, ga: float, xwo: float,
        cop: float, xo: float, z_mm: list,
) -> dict:
    """
    Port of C++ ``soiltemp_initconsts()`` (``fof_soi.cpp:264-295``):
    build a fresh solver-state dict for one simulation run. C++ uses
    persistent module-global arrays reused across calls within one run;
    this dict gives each Python call its own isolated state instead, so
    concurrent/repeated calls cannot interfere with each other.

    :param bd: Soil bulk density, in the SAME raw units as the pinned
        C++ ``sr_SD``/``sr_SE`` table literal (its own comment calls it
        "g/m^3", but the pinned solver never converts it — pass the
        table literal unconverted; only the ``xs = bd/pd`` ratio is
        scale-invariant, ``soiltemp_step``'s own ``cp[i]`` term is not).
    :param pd: Soil particle density, same raw/unconverted convention as *bd*.
    :param ls: Thermal conductivity of the mineral fraction (W/m/K).
    :param ga: de Vries shape factor.
    :param xwo: Water content for liquid recirculation (m^3/m^3).
    :param cop: Power for the liquid-recirculation function.
    :param xo: Extrapolated water content at -1 J/kg (m^3/m^3).
    :param z_mm: 15 node depths (mm), from :func:`_campbell_soil_depths_mm`.
    :return: A fresh state dict for :func:`_soiltemp_step`.
    """
    n = _CAMPBELL_NODE_COUNT
    z = np.asarray(z_mm, dtype=float) / 1000.0   # mm -> m, fof_soi.cpp:284-286
    v = np.zeros(n, dtype=float)
    for i in range(1, _CAMPBELL_INTERIOR_NODE_COUNT + 1):
        v[i] = 0.5 * (z[i + 1] - z[i - 1])
    air_por = np.zeros(n, dtype=float)
    air_por[0] = 1.0   # fof_soi.cpp:294
    xs = bd / pd
    return dict(
        bd=float(bd), pd=float(pd), ls=float(ls), ga=float(ga),
        xwo=float(xwo), cop=float(cop), xo=float(xo),
        xs=xs, xws=1.0 - xs, m=_CAMPBELL_INTERIOR_NODE_COUNT,
        z=z, v=v,
        w=np.zeros(n), wn=np.zeros(n), t=np.zeros(n), tn=np.zeros(n),
        p=np.zeros(n), dwdp=np.zeros(n), h=np.zeros(n), dhdp=np.zeros(n),
        psat=np.zeros(n), kev=np.zeros(n), u=np.zeros(n), Hvap=np.zeros(n),
        s=np.zeros(n), ke=np.zeros(n), kh=np.zeros(n), kv=np.zeros(n),
        cp=np.zeros(n), conv=np.zeros(n), vcon=np.zeros(n), enh=np.zeros(n),
        AirPor=air_por,
    )


def _soiltemp_initprofile(state: dict, w_init: float, t_init: float) -> None:
    """
    Port of C++ ``soiltemp_initprofile()`` (``fof_soi.cpp:302-317``):
    (re-)initialize the profile from a uniform starting water content and
    temperature. Mutates *state* in place — this is also called to RESET
    the profile after any state mutation the caller wants to discard
    (C++'s own usage pattern, though the retry path that would trigger
    this is unreachable because its iteration counter is never incremented).

    :param state: State dict from :func:`_soiltemp_initconsts`.
    :param w_init: Starting volumetric water content (m^3/m^3), uniform
        across all 15 nodes.
    :param t_init: Starting temperature (degC), uniform across all 15
        nodes.
    """
    n = _CAMPBELL_NODE_COUNT
    xo = state["xo"]
    state["w"][:] = w_init
    state["wn"][:] = w_init
    state["t"][:] = t_init
    state["tn"][:] = t_init
    for i in range(n):
        state["p"][i] = -math.exp(13.82 * (1.0 - state["w"][i] / xo))
        w, dwdp = _campbell_water_content(state["p"][i], xo)
        state["w"][i] = w
        state["dwdp"][i] = dwdp
        h, dhdp = _campbell_humidity(state["p"][i], state["t"][i])
        state["h"][i] = h
        state["dhdp"][i] = dhdp
        state["kev"][i] = 0.0
        state["u"][i] = 0.0
        state["enh"][i] = 0.0


def _soiltemp_step(state: dict, r_rabs: float, dt: float, on_subiter=None,
                    on_surface_update=None) -> bool:
    """
    Port of C++ ``soiltemp_step()`` (``fof_soi.cpp:87-211``): advance the
    coupled temperature/matric-potential Newton iteration by one
    timestep, converging via a repeated linearized update until the
    energy and water mass-balance residuals fall below
    :data:`_CAMPBELL_ENERGY_ERROR_LIMIT`/:data:`_CAMPBELL_WATER_ERROR_LIMIT`,
    or fail after :data:`_CAMPBELL_MAX_NEWTON_ITERATIONS` sub-iterations.

    This function coordinates one coupled timestep. Each Newton
    sub-iteration delegates boundary setup to
    :func:`_campbell_newton_boundary_init`, its per-node residual/
    Jacobian/Newton update to :func:`_campbell_newton_node_update`, and
    the post-convergence state advance to
    :func:`_campbell_commit_timestep`. The helpers preserve C++'s operation
    order across boundary preparation, residual/Jacobian construction,
    Newton updates, and the final state commit.

    Mutates *state* in place — on success, ``state['t']``/``state['tn']``
    and ``state['w']``/``state['wn']`` are both updated to the newly
    converged values (matching C++'s own "commit" step,
    ``fof_soi.cpp:195-206``).

    IEEE-754 divide-by-zero/invalid-value semantics (silent inf/nan,
    never raising) are used throughout, via ``np.errstate`` — matching
    C's own float arithmetic exactly for any degenerate denominator,
    rather than Python's default zero-division exception. This applies
    transitively to :func:`_campbell_newton_boundary_init` and
    :func:`_campbell_newton_node_update` too, since they only ever run
    while called from inside this function's own ``with`` block.

    :param state: State dict from :func:`_soiltemp_initconsts` (and
        typically :func:`_soiltemp_initprofile`).
    :param r_rabs: Total absorbed surface radiation (W/m^2) for this
        timestep.
    :param dt: Timestep (s).
    :param on_subiter: Diagnostic-only observer mirroring the overlay
        C++ diagnostic build's ``SoiDiagRecordSubIteration`` (see
        ``reference/fofem_cpp_overlay/source/FOF_UNIX/fof_soi_instr.cpp``)
        exactly: if given, called once per Newton sub-iteration as
        ``on_subiter(n_subiter, tn1, p1, r_sev, r_seh)`` at the SAME
        point in the loop (after the sub-iteration counter increments,
        before the convergence break-check) — surface-node (index 1)
        values only. ``None`` by default: zero behavior/performance
        change for every existing caller. Never used for control flow —
        purely an observer.
    :param on_surface_update: Diagnostic-only observer mirroring the
        overlay C++ diagnostic build's ``SoiDiagRecordSurfaceUpdate``
        exactly: if given, called once per Newton sub-iteration as
        ``on_surface_update(n_subiter, fields_dict)``, where
        ``fields_dict`` carries every named quantity the surface-node
        (i=1) update itself reads or writes this sub-iteration (old/new
        temperature and matric potential, water content, humidity,
        vapor pressure, every conductivity term, the boundary
        Stefan-Boltzmann correction before/after, both residuals, the
        Jacobian terms, both Newton increments, and any clamp branch
        taken) — keys match the C++ hook's CSV column names exactly.
        ``None`` by default: zero behavior/performance change. Never
        used for control flow — purely an observer. Forwarded to
        :func:`_campbell_newton_node_update` unchanged.
    :return: ``True`` on convergence; ``False`` if ``dt <= 0`` or the
        Newton iteration did not converge within
        :data:`_CAMPBELL_MAX_NEWTON_ITERATIONS` sub-iterations (both hard-failure
        conditions in C++, propagated identically here).
    """
    if dt <= 0.0:
        return False

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tn = state["tn"]
        p = state["p"]
        m = state["m"]

        tn[0] = _CAMPBELL_AIR_TEMPERATURE
        n_bug = 0

        while True:
            r_seh = 0.0
            r_sev = 0.0
            surf_old = _campbell_newton_boundary_init(state)

            for i in range(1, m + 1):
                d_v_abs, d_c_abs = _campbell_newton_node_update(
                    state, i, r_rabs, dt, n_bug, on_surface_update, surf_old,
                )
                r_sev += d_v_abs
                r_seh += d_c_abs

            n_bug += 1
            if on_subiter is not None:
                on_subiter(n_bug, float(tn[1]), float(p[1]), float(r_sev), float(r_seh))
            if n_bug >= _CAMPBELL_MAX_NEWTON_ITERATIONS:
                return False
            if r_sev < _CAMPBELL_WATER_ERROR_LIMIT and r_seh < _CAMPBELL_ENERGY_ERROR_LIMIT:
                break

        # Converged -- commit state (fof_soi.cpp:187-206).
        _campbell_commit_timestep(state, dt)

        return True


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
    Predict mineral soil temperature using a direct port of the pinned
    C++ coupled heat/moisture/vapor soil solver (``fof_soi.cpp``'s
    ``soiltemp_step``, driven by ``fof_sd.cpp``'s ``SD_Mngr_New`` for the
    duff route and ``fof_se.cpp``'s ``SE_Mngr_Array`` for the non-duff
    route — collectively C++'s ``SH_Mngr``).

    The function solves the same coupled temperature/matric-potential/vapor
    system as C++, with the same per-timestep Newton iteration, fixed
    per-route timestep, and termination logic. C++'s nominal timestep-
    halving retry branch is unreachable because its iteration counter is
    never incremented; a failed Newton solve therefore raises
    :class:`SoilSimulationError`, matching C++'s fatal ``e_SoiSimFail``
    path.

    Exact bit-for-bit parity is NOT expected or claimed: the pinned C++
    computes entirely in 32-bit ``float``, this port in 64-bit Python
    ``float``/``numpy.float64`` — see the function's own tolerance
    evidence in ``tests/cpp_parity_live/tolerance_policy.json``
    (``soil_campbell_p5.duff``/``nonduff``) for the measured, evidence-
    derived bound this is validated against.

    :param model: 'duff' for surface flux from duff smoldering, or
        'non_duff' for surface flux from a burnup intensity time series.
    :param duff_params: Parameters describing the duff layer, used when
        *model* is 'duff'. Ported directly from C++'s ``DuffBurn()``/
        ``SD_HeatAdj()`` contract (``bur_brn.cpp:1950-1986``,
        ``fof_sd.cpp:98-129,294-313``) — see :func:`_duff_burn_profile` for
        the exact citations and validation rules:

        * ``'duff_load'`` (T/ac) — REQUIRED. Missing or non-finite raises
          ``ValueError``; a present value ``<= 0`` (including negative) is
          valid and yields zero surface forcing, matching C++.
        * ``'duff_depth'`` (in) — pre-fire depth. Required, finite, ``>= 0``.
        * ``'duff_moisture'`` (%) — a WHOLE PERCENT (e.g. ``45.0`` for 45%),
          exactly as FOFEM's own duff-moisture convention documents it
          everywhere else in this package. Converted to C++'s internal
          ratio convention (divide by 100) exactly once, at this function's
          own forcing boundary — the caller never supplies a ratio
          directly. Required, finite, ``>= 0``.
        * ``'pct_consumed'`` (%) — REQUIRED, and must be in ``[0, 100]``
          inclusive; an out-of-range value raises ``ValueError`` rather
          than silently applying C++'s own moisture-only fallback formula
          (which exists only for a standalone-Burnup-without-FOFEM code
          path this Campbell contract has no equivalent of).
        * ``'efficiency_duff'`` (proportion, default 1.0) — matches C++'s
          ``f_EffFF * 0.01`` duff-efficiency scaling.
        * ``'duff_heat_content'`` — accepted (for backward compatibility)
          but NOT used: C++'s intensity formula is a self-contained
          function of moisture alone, with no heat-content input.

        The duff-to-soil heat-transmission fraction (C++'s
        ``SD_HeatAdj()``) is evaluated at the TIME-VARYING remaining duff
        depth (linearly interpolated from the pre-fire depth down to the
        implied post-fire depth over the burn duration), not held constant
        at the static pre-fire depth for the whole simulation.
    :param soil_params: Soil properties: 'soil_family' (one of
        'loamy-skeletal', 'fine-silty', 'fine', 'coarse-silty',
        'coarse-loamy'), 'start_water' (m³/m³), 'start_temp' (°C), plus
        optional overrides for any soil-family default key, including
        'recirc_water' (C++ ``xwo``). Every family's constants match the
        pinned C++ ``sr_SD``/``sr_SE`` tables bit-for-bit.
    :param depth_layers: Exactly 13 depths (cm) at which to predict
        temperature. For a true C++-equivalent comparison, this MUST be
        ``[1, 2, ..., 13]`` — the only depth scheme the pinned C++ harness
        ever actually exercises (``SH_Init_LayDis`` is not
        user-configurable); other depth lists are still accepted for API
        generality but have no C++ counterpart to compare against.
    :param burnup_intensity: Required wood/litter fire-intensity samples
        (kW/m²), 15 seconds apart; used when *model* is 'non_duff'.
    :param burnup_intensity_hs: Heavy-slash fire intensity (kW/m²) at each
        time step; used when *model* is 'non_duff'.
    :param burnup_times: Accepted for signature compatibility only. C++'s
        ``_Get_FirInt()`` samples *burnup_intensity*/*burnup_intensity_hs*
        by a FIXED 15-second-apart zero-order hold
        (``fof_sh.cpp:67``'s hardcoded literal ``15``), never by looking
        up this array's own values — matching the real production
        Burnup-array spacing. This parameter's actual values are not
        read.
    :param efficiency_wl: Fraction of woody-litter intensity delivered to the
        soil surface (default 0.15).
    :param efficiency_hs: Fraction of heavy-slash intensity delivered to the
        soil surface (default 0.10).
    :param timestep: Accepted for signature compatibility only and has NO
        EFFECT: the coupled solver's timestep is fixed by the pinned C++
        contract (20 s for the duff route, 10 s for the non-duff route),
        never user-configurable in C++.
    :raises ValueError: If *model* is not 'duff'/'non_duff', or *depth_layers*
        doesn't contain exactly 13 values; or (for *model* ``'duff'``) if
        *duff_params* is missing/invalid — see :func:`_duff_burn_profile`.
    :raises SoilSimulationError: If the coupled Newton iteration fails to
        converge for any timestep — matches C++'s own fatal
        ``e_SoiSimFail`` path exactly.
    :return: DataFrame indexed by time (minutes), columns 'Surface' then
        '<d>cm' for each depth in *depth_layers*. The first recorded row
        (``time_min == 0``) is the state after the FIRST Newton-converged
        timestep, not the raw pre-simulation initial condition — this
        matches the pinned C++ harness's own ``time_index == 0`` output
        convention exactly. In a representative non-duff reference output,
        the golden ``time_index=0`` row reads 38.849651 degC against a
        21.0 degC harness ambient start temperature, not the ambient value
        itself. A column may therefore differ from
        ``soil_params['start_temp']`` at
        ``time_min == 0`` — most visibly at the surface, which has already
        received one timestep of forcing; deeper layers converge back to
        ``start_temp`` within that same first step because diffusion has
        not yet had time to reach them.
    """
    model = model.strip().lower()
    if model not in ("duff", "non_duff"):
        raise ValueError(f"model must be 'duff' or 'non_duff', got '{model}'.")
    if len(depth_layers) != 13:
        raise ValueError(
            f"depth_layers must contain exactly 13 values, got {len(depth_layers)}."
        )

    props = _build_soil_props(soil_params)
    start_water = props["start_water"]
    start_temp = props["start_temp"]

    z_mm = _campbell_soil_depths_mm(depth_layers)
    ambient_rabs = _campbell_ambient_radiation(start_temp)

    state = _soiltemp_initconsts(
        props["bulk_density"], props["particle_density"], props["k_mineral"],
        props["vries_shape"], props["recirc_water"], props["cop_power"],
        props["extrap_water"], z_mm,
    )
    _soiltemp_initprofile(state, start_water, start_temp)

    if model == "duff":
        duff_profile = _duff_burn_profile(duff_params)
        duration_s = duff_profile["duration_s"]
        forcing_fn = _make_duff_forcing_fn(duff_profile, ambient_rabs)

        def done_fn(temps, st_temp, clock_sec, still_burning):
            """
            Duff-route termination check, bound to this call's burn duration.

            :param temps: Current temperature array.
            :param st_temp: Starting soil temperature (degC).
            :param clock_sec: Current simulation clock (s).
            :param still_burning: Unused — :func:`_soi_done_duff` derives
                its own burn/no-burn state from *clock_sec* vs
                *duration_s* directly, matching C++'s own ``_Done()``
                signature exactly.
            :return: ``True`` if the simulation should stop.
            """
            return _soi_done_duff(temps, st_temp, clock_sec, duration_s)

        dt = _CAMPBELL_DUFF_TIMESTEP
    else:
        if burnup_intensity is None:
            raise ValueError(
                "model='non_duff' requires burnup_intensity. Supply explicit "
                "wood/litter intensity samples or use soil_heat_from_consumption() "
                "to construct guide-defined forcing from consumed fuels."
            )
        else:
            wl_series = list(burnup_intensity)
            hs_series = (
                list(burnup_intensity_hs)
                if burnup_intensity_hs is not None
                else [0.0]
            )
        forcing_fn = _make_nonduff_forcing_fn(
            wl_series, hs_series, efficiency_wl, efficiency_hs, ambient_rabs,
        )

        def done_fn(temps, st_temp, clock_sec, fi_now):
            """
            Non-duff-route termination check.

            :param temps: Current temperature array.
            :param st_temp: Starting soil temperature (degC).
            :param clock_sec: Current simulation clock (s).
            :param fi_now: Combined WL+HS fire intensity (W/m^2, excluding
                the ambient term) at *clock_sec*.
            :return: ``True`` if the simulation should stop.
            """
            return _soi_done_nonduff(temps, st_temp, clock_sec, fi_now)

        dt = _CAMPBELL_NONDUFF_TIMESTEP

    records = _run_coupled_soil_sim(state, dt, forcing_fn, done_fn, start_temp)

    times_s = np.array([r[0] for r in records], dtype=float)
    temps = np.array([r[1] for r in records], dtype=float)  # (n_times, 14)
    times_min = times_s / 60.0
    cols = _column_names(depth_layers)  # 14 names: Surface + 13 depths
    df = pd.DataFrame(temps, index=times_min, columns=cols)
    df.index.name = "time_min"
    return df


def soil_heat_from_consumption(
        soil_params: dict,
        depth_layers: list,
        herb_shrub_consumed: float,
        woody_litter_intensity: Optional[Sequence[float]] = None,
        burnup_results: Optional[Sequence] = None,
        burnup_params: Optional[dict] = None,
        efficiency_wl: float = 0.15,
        efficiency_hs: float = 0.10,
        heat_content: float = 1.86e7,
) -> pd.DataFrame:
    """Predict non-duff soil heating from guide-defined fuel forcing.

    Woody/litter intensity is taken from supplied BURNUP results or from a
    BURNUP run configured by *burnup_params*. Herb/shrub intensity follows
    the FOFEM 6.9.4 guide: up to 5 T/ac is consumed in the first minute with
    10/20/30/40 percent allocated to its four 15-second intervals, followed
    by a uniform 5 T/ac/min rate until all consumed herb/shrub fuel is used.
    A BURNUP run created here excludes herb/shrub and branch/foliage forcing,
    so each guide-defined source is represented exactly once.

    :param soil_params: Soil-family and initial-condition parameters accepted
        by :func:`soil_heat_campbell`.
    :param depth_layers: Exactly 13 requested soil depths (cm).
    :param herb_shrub_consumed: Total consumed herb plus shrub load (kg/m^2).
    :param woody_litter_intensity: Optional 15-second wood/litter intensity
        samples (kW/m^2), such as the W/L series produced by Burnup.
    :param burnup_results: Optional BURNUP result sequence whose ``fi_wl``
        values provide 15-second woody/litter intensity samples (kW/m^2).
    :param burnup_params: Optional keyword arguments for
        :func:`pyfofem.components.burnup_calcs.run_burnup`; used to create
        *burnup_results* when they are not supplied.
    :param efficiency_wl: Woody/litter heat-delivery proportion.
    :param efficiency_hs: Herb/shrub heat-delivery proportion.
    :param heat_content: Low heat of combustion for guide-derived herb/shrub
        forcing (J/kg).
    :returns: Soil-temperature trajectory indexed by elapsed minutes.
    :raises ValueError: If more than one wood/litter input is supplied, if
        *burnup_params* does not use 15-second samples, or if a BURNUP result
        lacks woody/litter intensity.
    """
    woody_inputs = (woody_litter_intensity, burnup_results, burnup_params)
    if sum(value is not None for value in woody_inputs) > 1:
        raise ValueError(
            "Supply only one of woody_litter_intensity, burnup_results, or "
            "burnup_params."
        )

    if burnup_params is not None:
        if not isinstance(burnup_params, dict):
            raise ValueError("burnup_params must be a dictionary when supplied.")
        sample_interval = burnup_params.get("timestep", 15.0)
        if not np.isclose(sample_interval, 15.0):
            raise ValueError(
                "burnup_params['timestep'] must be 15 seconds for the "
                "FOFEM non-duff soil-forcing contract."
            )
        from .burnup_calcs import run_burnup

        woody_burnup_params = dict(burnup_params)
        woody_burnup_params["hsf_consumed"] = 0.0
        woody_burnup_params["brafol_consumed"] = 0.0
        burnup_results, _, _ = run_burnup(**woody_burnup_params)

    if woody_litter_intensity is not None:
        woody_litter_intensity = list(woody_litter_intensity)
        if not all(np.isfinite(intensity) for intensity in woody_litter_intensity):
            raise ValueError("Each wood/litter intensity must be finite.")
    elif burnup_results is None:
        woody_litter_intensity = []
    else:
        woody_litter_intensity = []
        for result in burnup_results:
            intensity = getattr(result, "fi_wl", None)
            if intensity is None or not np.isfinite(intensity):
                raise ValueError(
                    "Each burnup result must provide a finite fi_wl intensity."
                )
            woody_litter_intensity.append(float(intensity))

    herb_shrub_intensity = _make_guide_herb_shrub_intensity(
        herb_shrub_consumed, heat_content,
    )
    return soil_heat_campbell(
        model="non_duff",
        duff_params={},
        soil_params=soil_params,
        depth_layers=depth_layers,
        burnup_intensity=woody_litter_intensity,
        burnup_intensity_hs=herb_shrub_intensity,
        efficiency_wl=efficiency_wl,
        efficiency_hs=efficiency_hs,
    )


def soil_heat_massman(
        fire_type: str,
        bfd_params: dict,
        soil_params: dict,
        depth_layers: list,
        timestep: float = 10.0,
) -> dict:
    """
    Reserve the Massman HMV API until a full, validated implementation is available.

    The former implementation was a simplified heat/moisture approximation,
    not the three-state non-equilibrium Massman (2015) HMV model. It is not
    scientifically validated and must not be used for analysis or publication.

    :param fire_type: 'wildfire', 'prescribed_burn', or 'pile_burn'.
    :param bfd_params: BFD fire curve parameters: 'q_abs' (kW/m², peak heat
        rate), optional 't_m' (hr, time to peak heat rate, default 4),
        optional 't_d' (hr, fire duration; default 20 for wildfire, 8 for
        prescribed_burn, 40 for pile_burn).
    :param soil_params: Same keys as :func:`soil_heat_campbell`, plus optional
        Massman-specific overrides: 'extrap_water' (default from soil
        family, water content extrapolated to -1 J/kg), 'vries_shape'
        (default from soil family), 'cop_power' (default from soil family,
        power for liquid recirculation).
    :param depth_layers: Exactly 13 depths (cm).
    :param timestep: Maximum integration step (s, default 10).
    :returns: Never returns; this model is unavailable pending implementation
        and validation against published Massman HMV cases.
    :raises NotImplementedError: Always. Use :func:`soil_heat_campbell` for
        supported soil-heating simulations.
    """
    raise NotImplementedError(
        "soil_heat_massman() is in development and non-functional. "
        "Use soil_heat_campbell() for supported soil-heating simulations."
    )

    # The retained implementation below is intentionally unreachable while
    # the full three-state Massman HMV model is designed and validated.
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
        """
        Wrap :func:`_massman_rhs` with this call's fixed parameters for solve_ivp.

        :param t: Current simulation time (s).
        :param y: Current [temperature, moisture] state, length 28.
        :return: Concatenated [dT/dt, dtheta_l/dt], length 28.
        """
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
