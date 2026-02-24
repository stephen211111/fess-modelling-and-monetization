"""
linearized_physics.py
=====================
Linearized FESS physics for use in Linear Programming (LP) / day-ahead
energy trading optimization.

Non-linear relationships in the physics model and their LP treatments
---------------------------------------------------------------------

1. COMBINED EFFICIENCY (η_charge, η_discharge)
   Non-linear: η = f(power_pu, speed_ratio)  — varies with both power and speed.
   LP treatment: Find the single shaft power level that maximises combined
   round-trip efficiency (averaged over the operating speed window), then fix
   η_charge and η_discharge as constants for the LP.
   → scalar η_charge_lp, η_discharge_lp

2. STANDBY (SELF-DISCHARGE) LOSS
   Non-linear: mechanical losses ∝ ω² or ω³ (speed-dependent).
   LP treatment: Integrate total standby power (mechanical + auxiliary) over
   the full operational speed window [min_speed_ratio, max_speed_ratio],
   weighted uniformly (assuming the flywheel spends equal time at all SoC
   levels on average). Express as a constant kWh-per-hour drain.
   → scalar standby_loss_kw (constant)

3. AVAILABLE DISCHARGE POWER vs SoC
   Non-linear: P_available = P_rated × speed_ratio = P_rated × √SoC_frac
   LP treatment: In an LP the power limit must be linear in the state
   variable (SoC). We linearise P_available as a constant equal to the
   average available power over the operational SoC range. The LP then uses
   a single rated power constraint for both charge and discharge.
   → scalar p_max_lp (effective constant power limit)

4. SoC–ENERGY RELATIONSHIP
   Already linear: E_stored = SoC_kwh (energy in kWh is the LP state variable
   directly). No linearization needed — the LP operates entirely in kWh.

5. RAMP RATE
   Already linear: ΔP ≤ ramp_rate_kw_per_sec × Δt. Kept as-is.

6. TRANSFORMER / GRID INTERFACE LOSSES
   Non-linear: η_transformer (constant fraction, but applied multiplicatively).
   LP treatment: Combined into the charge/discharge efficiency constants as
   an additional multiplier. The auxiliary load is kept as a constant kW
   offset on every timestep.

Output — LinearizedFESSParams dataclass
---------------------------------------
A flat set of LP-ready constants that fully describe one FESS unit for
scheduling purposes:

    rated_energy_kwh      : Maximum stored energy (LP variable upper bound)
    min_energy_kwh        : Minimum stored energy (LP variable lower bound)
    p_max_kw              : Max charge/discharge power (constant)
    eta_charge            : One-way charge efficiency (constant)
    eta_discharge         : One-way discharge efficiency (constant)
    eta_roundtrip         : Round-trip efficiency = eta_charge * eta_discharge
    standby_loss_kw       : Constant self-discharge power (kW), drains SoC
    auxiliary_load_kw     : Constant auxiliary draw from grid (kW), does NOT
                            drain SoC — affects net grid power only
    opt_power_pu          : Per-unit power at which efficiency was optimised
    opt_speed_ratio       : Speed ratio at which efficiency was evaluated for plot

Usage in an LP
--------------
For each 15-min interval t with duration dt_h = 0.25 h:

    Variables:
        e[t]      : SoC in kWh at end of interval t
        p_c[t]    : Charge power (kW), ≥ 0
        p_d[t]    : Discharge power (kW), ≥ 0

    SoC update:
        e[t] = e[t-1]
               + p_c[t] * eta_charge * dt_h
               - p_d[t] / eta_discharge * dt_h   ← shaft energy needed
               - standby_loss_kw * dt_h

        (Equivalently for discharge: grid delivery = p_d[t]; shaft draw =
         p_d[t] / eta_discharge)

    Bounds:
        min_energy_kwh ≤ e[t] ≤ rated_energy_kwh
        0 ≤ p_c[t] ≤ p_max_kw
        0 ≤ p_d[t] ≤ p_max_kw
        p_c[t] + p_d[t] ≤ p_max_kw   (cannot charge and discharge simultaneously)

    Objective contribution per interval:
        revenue[t]  = p_d[t] * price[t] * dt_h       (discharge revenue)
        cost[t]     = (p_c[t] + auxiliary_load_kw) * price[t] * dt_h  (charge + aux cost)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from scipy.optimize import minimize_scalar

from efficiency_models import (
    MachineParams, MachineEfficiency,
    InverterParams, InverterEfficiency,
    FESSEfficiencyModel,
)
from standby_losses import StandbyLossParams
from fess_unit import FESSParams


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LinearizedFESSParams:
    """
    Flat set of LP-ready constants for one FESS unit.
    All non-linear physics relationships have been collapsed to scalars.
    """
    # Energy bounds (kWh) — LP variable bounds
    rated_energy_kwh:   float   # = FESSParams.rated_energy_kwh × max_speed_ratio²
    min_energy_kwh:     float   # = FESSParams.rated_energy_kwh × min_speed_ratio²

    # Power limit (kW) — linearised via least-squares regression over SoC window
    # LP constraint: p_discharge[t] <= p_max_slope * e[t] + p_max_intercept
    # where e[t] is SoC in kWh.  Both coefficients are constants, so the
    # constraint remains linear in the LP state variable.
    p_max_kw:           float   # Average (simple constant fallback)
    p_max_slope:        float   # kW per kWh — regression slope
    p_max_intercept:    float   # kW — regression intercept

    # Efficiency constants
    eta_charge:         float   # Grid → flywheel shaft (one-way)
    eta_discharge:      float   # Flywheel shaft → grid (one-way)
    eta_roundtrip:      float   # = eta_charge × eta_discharge

    # Standby losses — constant per hour
    standby_loss_kw:    float   # Drains SoC (mechanical, flywheel kinetic energy)
    auxiliary_load_kw:  float   # Grid draw (does not drain SoC)

    # Diagnostics — where in the efficiency map the constants were evaluated
    opt_power_pu:       float   # Per-unit power at peak speed-averaged round-trip η
    opt_power_kw:       float   # Absolute power at peak speed-averaged round-trip η
    opt_speed_ratio:    float   # Speed ratio at which eta_charge/discharge were sampled
    p_max_r2:           float   # R² of the linear regression for p_max vs SoC

    def summary(self) -> dict:
        return {
            "rated_energy_kwh":  round(self.rated_energy_kwh,  3),
            "min_energy_kwh":    round(self.min_energy_kwh,    3),
            "usable_energy_kwh": round(self.rated_energy_kwh - self.min_energy_kwh, 3),
            "p_max_kw (avg)":    round(self.p_max_kw,          2),
            "p_max_slope (kW/kWh)":  round(self.p_max_slope,   6),
            "p_max_intercept (kW)":  round(self.p_max_intercept, 4),
            "p_max_regression_R2":   round(self.p_max_r2,       6),
            "eta_charge":        f"{self.eta_charge:.4%}",
            "eta_discharge":     f"{self.eta_discharge:.4%}",
            "eta_roundtrip":     f"{self.eta_roundtrip:.4%}",
            "standby_loss_kw":   round(self.standby_loss_kw,   4),
            "auxiliary_load_kw": round(self.auxiliary_load_kw, 4),
            "opt_power_pu":      round(self.opt_power_pu,      4),
            "opt_power_kw":      round(self.opt_power_kw,      2),
            "opt_speed_ratio":   round(self.opt_speed_ratio,   4),
        }


# ---------------------------------------------------------------------------
# Core linearization function
# ---------------------------------------------------------------------------

def linearize_fess(
    params: FESSParams,
    n_speed_points: int = 200,
    n_power_points: int = 200,
) -> LinearizedFESSParams:
    """
    Derive LP-ready constants from a full-physics FESSParams object.

    Steps
    -----
    1. Build efficiency model from MachineParams + InverterParams.
    2. Find the shaft power_pu that maximises round-trip efficiency averaged
       across the full operational speed window [min_speed_ratio, max_speed_ratio].
       For each candidate power_pu, compute RT efficiency at every speed in the
       grid and take the mean. The power_pu with the highest mean is the optimum.
    3. Compute eta_charge and eta_discharge at (opt_power_pu, sr_mid) where
       sr_mid is the centre of the speed window, to give a representative
       single-point efficiency consistent with the speed-averaged optimum.
    4. Compute average total standby loss (mechanical + auxiliary) integrated
       uniformly over the speed window [min_speed_ratio, max_speed_ratio].
    5. Linearise available discharge power P_available = P_rated * speed_ratio
       (non-linear in SoC since SoC = speed_ratio^2) using least-squares linear
       regression of P_available vs SoC_kwh over the operational window.
       Gives p_max_slope (kW/kWh) and p_max_intercept (kW), which together
       define a tighter, LP-compatible constraint than a single average.

    Parameters
    ----------
    params : FESSParams
        Full-physics unit parameters.
    n_speed_points : int
        Resolution for numerical integration over the speed window.
    n_power_points : int
        Resolution for the power sweep when finding optimal power.

    Returns
    -------
    LinearizedFESSParams
    """
    eff_model = params.build_efficiency_model()
    sb_params = params.standby_params

    sr_min = params.min_speed_ratio
    sr_max = params.max_speed_ratio
    # Energy midpoint: SoC_mid = (SoC_min + SoC_max) / 2 = (sr_min² + sr_max²) / 2
    # Convert back to speed: sr_energy_mid = sqrt(SoC_mid)
    # This is more representative than the speed midpoint because the LP state
    # variable is energy (kWh), not speed, so equal weighting in energy space is correct.
    sr_energy_mid = float(np.sqrt(0.5 * (sr_min**2 + sr_max**2)))
    P_rat  = params.rated_power_kw

    speed_grid = np.linspace(sr_min, sr_max, n_speed_points)
    power_grid = np.linspace(0.05, 1.0, n_power_points)

    # ------------------------------------------------------------------
    # Step 1: Find optimal shaft power over the full speed window.
    # For each candidate power_pu, compute RT efficiency at every speed
    # in speed_grid and average — the power_pu with the highest mean RT
    # efficiency across all speeds is the LP operating point.
    # ------------------------------------------------------------------
    mean_rt_by_power = np.array([
        np.mean([eff_model.eta_roundtrip(p_pu, sr) for sr in speed_grid])
        for p_pu in power_grid
    ])
    opt_idx      = int(np.argmax(mean_rt_by_power))
    opt_power_pu = float(power_grid[opt_idx])
    opt_power_kw = opt_power_pu * P_rat

    # ------------------------------------------------------------------
    # Step 2: Efficiency constants at (opt_power_pu, sr_energy_mid).
    # sr_energy_mid corresponds to the midpoint of the SoC (energy) window,
    # which is the correct representative point since the LP state variable
    # is energy in kWh, not speed ratio.
    # ------------------------------------------------------------------
    eta_c  = eff_model.eta_charge(opt_power_pu, sr_energy_mid)
    eta_d  = eff_model.eta_discharge(opt_power_pu, sr_energy_mid)
    eta_rt = eta_c * eta_d

    # ------------------------------------------------------------------
    # Step 3: Average standby loss over the operational speed window.
    # Mechanical losses drain SoC; auxiliary losses draw from grid.
    # ------------------------------------------------------------------
    vac_duty  = sb_params.vacuum_on_duration_s / sb_params.vacuum_cycle_period_s
    vac_avg_w = sb_params.p_vacuum_pump_w * vac_duty

    mech_w_arr = (
        sb_params.k_aero_w        * speed_grid**3
        + sb_params.k_tmb_eddy_w      * speed_grid**2
        + sb_params.k_rmb_eddy_sync_w * speed_grid**2
    )
    aux_w_arr = (
        sb_params.p_tmb_bias_w
        + sb_params.k_rmb_bias_w    * speed_grid**2
        + sb_params.p_rmb_eddy_pwm_w
        + sb_params.p_cooling_w
        + vac_avg_w
    )

    avg_mech_kw = float(np.mean(mech_w_arr)) / 1000.0
    avg_aux_kw  = float(np.mean(aux_w_arr))  / 1000.0

    # ------------------------------------------------------------------
    # Step 4: Linearise P_available vs SoC via least-squares regression.
    #
    # Physics:  P_available = P_rated * speed_ratio
    #           SoC_kwh     = rated_energy_kwh * speed_ratio^2
    #   =>      P_available = P_rated * sqrt(SoC_kwh / rated_energy_kwh)
    #
    # This is non-linear (square-root) in the LP state variable SoC_kwh.
    # Least-squares linear fit over the operational SoC window:
    #   P_available ≈ p_max_slope * SoC_kwh + p_max_intercept
    # Both coefficients are constants → LP constraint stays linear.
    # ------------------------------------------------------------------
    rated_energy_kwh = params.rated_energy_kwh * sr_max**2
    min_energy_kwh   = params.rated_energy_kwh * sr_min**2

    soc_kwh_grid  = params.rated_energy_kwh * speed_grid**2
    p_avail_grid  = P_rat * speed_grid

    # Least-squares: [SoC_kwh | 1] @ [slope, intercept]^T = P_available
    A = np.column_stack([soc_kwh_grid, np.ones_like(soc_kwh_grid)])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, p_avail_grid, rcond=None)
    p_max_slope     = float(coeffs[0])   # kW per kWh
    p_max_intercept = float(coeffs[1])   # kW

    # R² — quality of the linear approximation
    ss_res = float(np.sum((p_avail_grid - (p_max_slope * soc_kwh_grid + p_max_intercept))**2))
    ss_tot = float(np.sum((p_avail_grid - np.mean(p_avail_grid))**2))
    p_max_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # Simple average as a fallback constant (kept for backward compat)
    p_max_kw = float(np.mean(p_avail_grid))

    return LinearizedFESSParams(
        rated_energy_kwh  = rated_energy_kwh,
        min_energy_kwh    = min_energy_kwh,
        p_max_kw          = p_max_kw,
        p_max_slope       = p_max_slope,
        p_max_intercept   = p_max_intercept,
        eta_charge        = eta_c,
        eta_discharge     = eta_d,
        eta_roundtrip     = eta_rt,
        standby_loss_kw   = avg_mech_kw,
        auxiliary_load_kw = avg_aux_kw,
        opt_power_pu      = opt_power_pu,
        opt_power_kw      = opt_power_kw,
        opt_speed_ratio   = sr_energy_mid,
        p_max_r2          = p_max_r2,
    )


# ---------------------------------------------------------------------------
# Fleet-level linearization
# ---------------------------------------------------------------------------

@dataclass
class FleetLinearizedParams:
    """
    LP-ready constants for a homogeneous fleet of N identical FESS units
    with a shared grid connection (transformer + auxiliary load).

    All unit-level quantities are scaled by N.  Grid-side constraints are
    added on top and are independent of N.

    LP variable: E[t] = total fleet SoC in kWh (sum over all units)

    SoC update (per interval dt_h hours):
        E[t] = E[t-1]
               + P_charge[t]    * eta_charge    * dt_h
               - P_discharge[t] / eta_discharge * dt_h
               - fleet_standby_loss_kw           * dt_h

    Net grid power (positive = import, negative = export):
        P_grid[t] = (P_charge[t] - P_discharge[t]) * eta_transformer
                  + fleet_auxiliary_kw

    Constraints:
        fleet_min_energy_kwh <= E[t] <= fleet_rated_energy_kwh
        0 <= P_charge[t]    <= p_max_slope * E[t] + fleet_p_max_intercept
        0 <= P_discharge[t] <= p_max_slope * E[t] + fleet_p_max_intercept
        P_charge[t] + P_discharge[t] <= p_max_slope * E[t] + fleet_p_max_intercept
        |P_grid[t]| <= transformer_capacity_kw   (transformer limit)
        P_charge[t] <= transformer_capacity_kw - fleet_auxiliary_kw  (net import headroom)
    """
    # Fleet energy bounds
    fleet_rated_energy_kwh: float   # N * unit rated_energy_kwh
    fleet_min_energy_kwh:   float   # N * unit min_energy_kwh
    fleet_usable_energy_kwh:float   # fleet_rated - fleet_min

    # Fleet power limit — linear in fleet SoC (same slope, intercept scaled by N)
    fleet_p_max_kw:         float   # N * unit p_max_kw (average fallback)
    p_max_slope:            float   # kW per kWh — same as unit (slope is intensive)
    fleet_p_max_intercept:  float   # N * unit p_max_intercept
    p_max_r2:               float   # R² of regression (same as unit)

    # Efficiency (same as unit — efficiency is intensive, not extensive)
    eta_charge:             float
    eta_discharge:          float
    eta_roundtrip:          float

    # Fleet standby losses (scaled by N)
    fleet_standby_loss_kw:  float   # N * unit standby_loss_kw  (drains SoC)
    fleet_auxiliary_kw:     float   # N * unit auxiliary_load_kw (grid draw)

    # Grid interface constraints
    transformer_capacity_kw:float   # Hard cap on total plant grid power (kW)
    eta_transformer:        float   # Transformer efficiency (applied to P_grid)
    effective_charge_limit_kw:  float  # min(fleet_p_max_kw, transformer_capacity_kw - fleet_auxiliary_kw)
    effective_discharge_limit_kw: float  # min(fleet_p_max_kw, transformer_capacity_kw)

    # Diagnostics
    n_units:                int
    opt_power_kw:           float   # Per-unit optimal shaft power
    opt_power_pu:           float

    def summary(self) -> dict:
        return {
            "n_units":                      self.n_units,
            "fleet_rated_energy_kwh":       round(self.fleet_rated_energy_kwh,   1),
            "fleet_min_energy_kwh":         round(self.fleet_min_energy_kwh,     1),
            "fleet_usable_energy_kwh":      round(self.fleet_usable_energy_kwh,  1),
            "fleet_p_max_kw (avg)":         round(self.fleet_p_max_kw,           1),
            "p_max_slope (kW/kWh)":         round(self.p_max_slope,              6),
            "fleet_p_max_intercept (kW)":   round(self.fleet_p_max_intercept,    2),
            "p_max_regression_R2":          round(self.p_max_r2,                 6),
            "eta_charge":                   f"{self.eta_charge:.4%}",
            "eta_discharge":                f"{self.eta_discharge:.4%}",
            "eta_roundtrip":                f"{self.eta_roundtrip:.4%}",
            "fleet_standby_loss_kw":        round(self.fleet_standby_loss_kw,    4),
            "fleet_auxiliary_kw":           round(self.fleet_auxiliary_kw,       3),
            "transformer_capacity_kw":      round(self.transformer_capacity_kw,  1),
            "eta_transformer":              f"{self.eta_transformer:.4%}",
            "effective_charge_limit_kw":    round(self.effective_charge_limit_kw,  1),
            "effective_discharge_limit_kw": round(self.effective_discharge_limit_kw, 1),
            "opt_power_kw (per unit)":      round(self.opt_power_kw,             2),
        }


def linearize_fleet(
    unit_params: FESSParams,
    grid_params,                    # GridInterfaceParams from fess_plant
    n_units: int = 50,
    n_speed_points: int = 200,
    n_power_points: int = 200,
) -> FleetLinearizedParams:
    """
    Derive LP-ready fleet constants by scaling unit linearization and
    applying grid-interface constraints.

    The slope of the power-availability regression (p_max_slope) is
    an intensive quantity — it does not scale with N because both
    numerator (kW) and denominator (kWh) scale by the same N.
    The intercept scales linearly with N.

    Parameters
    ----------
    unit_params : FESSParams
        Single-unit physics parameters.
    grid_params : GridInterfaceParams
        Plant-level grid connection parameters.
    n_units : int
        Number of identical units in the fleet.
    """
    unit = linearize_fess(unit_params, n_speed_points, n_power_points)

    # Transformer capacity in kW (derate by power factor)
    transformer_capacity_kw = (
        grid_params.transformer_capacity_kva
        * grid_params.power_factor
        * grid_params.transformer_efficiency
    )

    fleet_auxiliary_kw = n_units * unit.auxiliary_load_kw + grid_params.auxiliary_load_kw

    fleet_p_max_kw        = n_units * unit.p_max_kw
    fleet_p_max_intercept = n_units * unit.p_max_intercept  # slope unchanged (intensive)

    # Effective limits: tighter of fleet physics and transformer capacity
    effective_charge_limit_kw = min(
        fleet_p_max_kw,
        transformer_capacity_kw - fleet_auxiliary_kw,
    )
    effective_discharge_limit_kw = min(
        fleet_p_max_kw,
        transformer_capacity_kw,
    )

    return FleetLinearizedParams(
        fleet_rated_energy_kwh      = n_units * unit.rated_energy_kwh,
        fleet_min_energy_kwh        = n_units * unit.min_energy_kwh,
        fleet_usable_energy_kwh     = n_units * (unit.rated_energy_kwh - unit.min_energy_kwh),
        fleet_p_max_kw              = fleet_p_max_kw,
        p_max_slope                 = unit.p_max_slope,
        fleet_p_max_intercept       = fleet_p_max_intercept,
        p_max_r2                    = unit.p_max_r2,
        eta_charge                  = unit.eta_charge,
        eta_discharge               = unit.eta_discharge,
        eta_roundtrip               = unit.eta_roundtrip,
        fleet_standby_loss_kw       = n_units * unit.standby_loss_kw,
        fleet_auxiliary_kw          = fleet_auxiliary_kw,
        transformer_capacity_kw     = transformer_capacity_kw,
        eta_transformer             = grid_params.transformer_efficiency,
        effective_charge_limit_kw   = effective_charge_limit_kw,
        effective_discharge_limit_kw= effective_discharge_limit_kw,
        n_units                     = n_units,
        opt_power_kw                = unit.opt_power_kw,
        opt_power_pu                = unit.opt_power_pu,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_linearization(
    params: FESSParams,
    lp: LinearizedFESSParams,
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Four-panel diagnostic figure:

    Panel 1 — Combined round-trip efficiency vs shaft power at several
               fixed speed ratios; vertical line at optimal LP power point.

    Panel 2 — Combined round-trip efficiency vs speed ratio at the optimal
               shaft power; horizontal line at the LP constant used.

    Panel 3 — Standby power components vs speed ratio; horizontal line at
               the LP average standby loss.

    Panel 4 — Available discharge power vs SoC fraction; horizontal line
               at the LP constant p_max.
    """
    eff_model  = params.build_efficiency_model()
    sb_params  = params.standby_params
    sr_min     = params.min_speed_ratio
    sr_max     = params.max_speed_ratio
    P_rat      = params.rated_power_kw

    p_range    = np.linspace(0.05, 1.0, 300)
    sr_range   = np.linspace(sr_min, sr_max, 300)
    soc_range  = sr_range ** 2   # SoC fraction = speed_ratio²

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ----------------------------------------------------------------
    # Panel 1: Round-trip efficiency vs shaft power (multiple speeds)
    #          + speed-averaged RT efficiency curve
    # ----------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    speed_samples = np.linspace(sr_min, sr_max, 6)
    colors1 = plt.cm.viridis(np.linspace(0.15, 0.9, len(speed_samples)))

    for sr, col in zip(speed_samples, colors1):
        eta_rt_arr = np.array([eff_model.eta_roundtrip(p, sr) for p in p_range])
        ax1.plot(p_range * P_rat, eta_rt_arr * 100,
                 color=col, linewidth=1.4, alpha=0.7,
                 label=f"sr = {sr:.2f}")

    # Speed-averaged RT efficiency across the full operational window
    sr_full = np.linspace(sr_min, sr_max, 100)
    eta_avg_arr = np.array([
        np.mean([eff_model.eta_roundtrip(p, sr) for sr in sr_full])
        for p in p_range
    ])
    ax1.plot(p_range * P_rat, eta_avg_arr * 100,
             color="black", linewidth=2.5, linestyle="-",
             label="Speed-averaged RT eta")

    ax1.axvline(lp.opt_power_kw, color="red", linewidth=2, linestyle="--",
                label=f"LP opt: {lp.opt_power_kw:.1f} kW\n"
                      f"(p_pu={lp.opt_power_pu:.3f}, avg eta={eta_avg_arr[np.argmax(eta_avg_arr)]:.3%})")
    ax1.set_xlabel("Shaft power (kW)")
    ax1.set_ylabel("Round-trip efficiency (%)")
    ax1.set_title("Combined RT Efficiency vs Shaft Power\n"
                  "(individual speeds + speed-averaged curve)")
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(88, 100)

    # ----------------------------------------------------------------
    # Panel 2: Round-trip efficiency vs speed ratio at optimal power
    # ----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    eta_vs_speed = np.array([
        eff_model.eta_roundtrip(lp.opt_power_pu, sr) for sr in sr_range
    ])
    ax2.plot(sr_range, eta_vs_speed * 100,
             color="#2196F3", linewidth=2.5, label="RT efficiency at LP power")
    ax2.axhline(lp.eta_roundtrip * 100, color="red", linewidth=2,
                linestyle="--",
                label=f"LP constant: {lp.eta_roundtrip:.4%}\n"
                      f"(at sr_energy_mid = {lp.opt_speed_ratio:.3f})")
    ax2.fill_between(sr_range, eta_vs_speed * 100, lp.eta_roundtrip * 100,
                     alpha=0.15, color="orange",
                     label="Linearization error band")
    ax2.set_xlabel("Speed ratio (ω/ωmax)")
    ax2.set_ylabel("Round-trip efficiency (%)")
    ax2.set_title(f"Combined RT Efficiency vs Speed Ratio\n"
                  f"at optimal shaft power ({lp.opt_power_kw:.1f} kW, p_pu={lp.opt_power_pu:.3f})\n"
                  f"eta sampled at energy midpoint sr={lp.opt_speed_ratio:.3f}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(sr_min, sr_max)

    # ----------------------------------------------------------------
    # Panel 3: Standby power components vs speed ratio
    # ----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    vac_duty  = sb_params.vacuum_on_duration_s / sb_params.vacuum_cycle_period_s
    vac_avg_w = sb_params.p_vacuum_pump_w * vac_duty

    p_aero_arr     = sb_params.k_aero_w        * sr_range**3
    p_tmb_arr      = sb_params.k_tmb_eddy_w    * sr_range**2
    p_rmb_mech_arr = sb_params.k_rmb_eddy_sync_w * sr_range**2
    mech_total_arr = p_aero_arr + p_tmb_arr + p_rmb_mech_arr

    p_tmb_bias_arr = np.full_like(sr_range, sb_params.p_tmb_bias_w)
    p_rmb_bias_arr = sb_params.k_rmb_bias_w * sr_range**2
    p_pwm_arr      = np.full_like(sr_range, sb_params.p_rmb_eddy_pwm_w)
    p_cool_arr     = np.full_like(sr_range, sb_params.p_cooling_w)
    p_vac_arr      = np.full_like(sr_range, vac_avg_w)
    aux_total_arr  = p_tmb_bias_arr + p_rmb_bias_arr + p_pwm_arr + p_cool_arr + p_vac_arr

    grand_total_arr = mech_total_arr + aux_total_arr

    ax3.stackplot(
        sr_range,
        p_aero_arr, p_tmb_arr, p_rmb_mech_arr,
        p_tmb_bias_arr, p_rmb_bias_arr, p_pwm_arr, p_cool_arr, p_vac_arr,
        labels=[
            "Aero drag (∝ sr³)", "TMB eddy (∝ sr²)", "RMB sync eddy (∝ sr²)",
            "TMB bias (const)", "RMB bias (∝ sr²)", "RMB PWM eddy (const)",
            "Cooling (const)", f"Vacuum avg (duty={vac_duty:.1%})",
        ],
        colors=["#1565C0","#1E88E5","#42A5F5",
                "#E65100","#FB8C00","#FFA726","#FFCC02","#B0BEC5"],
        alpha=0.82,
    )
    ax3.plot(sr_range, grand_total_arr, color="black", linewidth=1.5,
             linestyle="-", label="Grand total (W)")
    avg_total_kw = lp.standby_loss_kw + lp.auxiliary_load_kw
    ax3.axhline(avg_total_kw * 1000, color="red", linewidth=2, linestyle="--",
                label=f"LP avg total: {avg_total_kw * 1000:.1f} W\n"
                      f"  mech={lp.standby_loss_kw * 1000:.1f} W, "
                      f"aux={lp.auxiliary_load_kw * 1000:.1f} W")
    ax3.set_xlabel("Speed ratio (ω/ωmax)")
    ax3.set_ylabel("Standby power (W)")
    ax3.set_title("Standby Loss Components vs Speed Ratio\n"
                  "(LP uses average over operational window)")
    ax3.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(sr_min, sr_max)

    # ----------------------------------------------------------------
    # Panel 4: Available discharge power vs SoC (kWh)
    #          Physics curve vs least-squares linear regression
    # ----------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    # Physics: P_available = P_rated * speed_ratio = P_rated * sqrt(SoC/E_rated)
    soc_kwh_range = params.rated_energy_kwh * sr_range**2
    p_avail_arr   = P_rat * sr_range

    # LP regression line evaluated over the same SoC range
    p_regression  = lp.p_max_slope * soc_kwh_range + lp.p_max_intercept

    ax4.plot(soc_kwh_range, p_avail_arr,
             color="#2196F3", linewidth=2.5,
             label="Physics: P = P_rated x sqrt(SoC/E_rated)")
    ax4.plot(soc_kwh_range, p_regression,
             color="red", linewidth=2.5, linestyle="--",
             label=f"LP regression: P = {lp.p_max_slope:.4f} x SoC + {lp.p_max_intercept:.2f}\n"
                   f"R\u00b2 = {lp.p_max_r2:.6f}")
    ax4.fill_between(soc_kwh_range, p_avail_arr, p_regression,
                     alpha=0.15, color="orange", label="Regression error band")

    soc_min_kwh = sr_min**2 * params.rated_energy_kwh
    soc_max_kwh = sr_max**2 * params.rated_energy_kwh
    ax4.axvline(soc_min_kwh, color="gray", linewidth=1, linestyle=":",
                label=f"SoC min ({soc_min_kwh:.0f} kWh)")
    ax4.axvline(soc_max_kwh, color="gray", linewidth=1, linestyle=":",
                label=f"SoC max ({soc_max_kwh:.0f} kWh)")

    ax4.set_xlabel("State of Charge (kWh)")
    ax4.set_ylabel("Available discharge power (kW)")
    ax4.set_title("Available Discharge Power vs SoC\n"
                  "(sqrt non-linearity -> least-squares linear regression)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(soc_min_kwh * 0.95, soc_max_kwh * 1.02)

    plt.suptitle(
        f"FESS Linearization for LP — Unit: {P_rat:.0f} kW / "
        f"{params.rated_energy_kwh:.0f} kWh",
        fontsize=13, fontweight="bold",
    )

    if save:
        plt.savefig("fess_linearization.png", dpi=150, bbox_inches="tight")
        print("Plot saved: fess_linearization.png")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Entry point — run standalone to inspect results
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from fess_unit import FESSParams
    from fess_plant import GridInterfaceParams

    params = FESSParams(
        rated_power_kw   = 292.0,
        rated_energy_kwh = 1169.0,
        min_speed_ratio  = 0.20,
    )
    grid = GridInterfaceParams(
        transformer_capacity_kva = 20_000.0,
        transformer_efficiency   = 0.995,
        auxiliary_load_kw        = 50.0,
        power_factor             = 0.95,
    )
    N_UNITS = 50

    print("Computing linearized FESS parameters for LP...")
    lp = linearize_fess(params)

    print("\n=== Single Unit — Linearized Parameters ===")
    for k, v in lp.summary().items():
        print(f"  {k:30s}: {v}")

    print(f"\n=== Fleet ({N_UNITS} units) — Linearized Parameters ===")
    fleet = linearize_fleet(params, grid, n_units=N_UNITS)
    for k, v in fleet.summary().items():
        print(f"  {k:35s}: {v}")

    print("\n=== Fleet LP Formulation ===")
    print(f"  Variables per interval:")
    print(f"    E[t]          : fleet SoC (kWh), [{fleet.fleet_min_energy_kwh:.0f}, {fleet.fleet_rated_energy_kwh:.0f}]")
    print(f"    P_charge[t]   : fleet charge power (kW), [0, {fleet.effective_charge_limit_kw:.1f}]")
    print(f"    P_discharge[t]: fleet discharge power (kW), [0, {fleet.effective_discharge_limit_kw:.1f}]")
    print()
    print(f"  SoC update (dt_h = interval length in hours):")
    print(f"    E[t] = E[t-1]")
    print(f"         + P_charge[t]    x {fleet.eta_charge:.6f} x dt_h")
    print(f"         - P_discharge[t] / {fleet.eta_discharge:.6f} x dt_h")
    print(f"         - {fleet.fleet_standby_loss_kw:.4f} x dt_h")
    print()
    print(f"  Net grid power (kW, +ve = import):")
    print(f"    P_grid[t] = (P_charge[t] - P_discharge[t]) x {fleet.eta_transformer:.4f}")
    print(f"              + {fleet.fleet_auxiliary_kw:.3f}  [auxiliary]")
    print()
    print(f"  Power-availability constraint (linear in E[t]):")
    print(f"    P_charge[t]    <= {fleet.p_max_slope:.6f} x E[t] + {fleet.fleet_p_max_intercept:.2f}")
    print(f"    P_discharge[t] <= {fleet.p_max_slope:.6f} x E[t] + {fleet.fleet_p_max_intercept:.2f}")
    print(f"    (R2 = {fleet.p_max_r2:.6f})")
    print()
    print(f"  Transformer capacity constraints:")
    print(f"    P_charge[t]    <= {fleet.effective_charge_limit_kw:.1f} kW  (transformer - auxiliary headroom)")
    print(f"    P_discharge[t] <= {fleet.effective_discharge_limit_kw:.1f} kW  (transformer capacity)")

    print("\nGenerating linearization diagnostic plot...")
    plot_linearization(params, lp, save=True, show=False)
    print("Done.")
