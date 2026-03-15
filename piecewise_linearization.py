"""
piecewise_linearization.py
==========================
Piecewise linearization of FESS physics for MILP-based energy dispatch.

Extends linearized_physics.py by segmenting the operational space into a 2D
grid of (power band j × SoC segment k) cells. Each cell carries its own
efficiency constants, computed from the full-physics model at the cell's
representative (p_mid, sr_mid) operating point.

Why 2D is needed
----------------
The combined round-trip efficiency is a surface η_rt(p_pu, speed_ratio):

  - Along the power axis:   inverter fixed losses dominate at low power
                            (η drops at part load), conduction losses at high
                            power (η drops again). Peak η at ~35–45% load.
  - Along the SoC axis:     copper losses ∝ (p/sr)² → rise sharply at low
                            SoC/speed for the same delivered power. P_available
                            also shrinks at low SoC.

A single global constant misrepresents both effects.  The 2D grid approximates
the surface with K_p × K_e constant patches — one per cell.

Avoiding bilinear products
--------------------------
In the MILP, the SoC update contains the term  p_c[t] × η_c(p[t], SoC[t]).
Both p_c and SoC are continuous decision variables, so this product is
bilinear — not directly compatible with LP/MILP.

The key trick: introduce auxiliary variable q_c[t,j,k] = p_c[t] when cell
(j,k) is active, else 0.  Then:

    p_c[t] × η_c(p[t], SoC[t])  ≈  Σ_{j,k}  η_c[j,k] × q_c[t,j,k]

The right-hand side is a sum of (constant × continuous variable) — fully
linear. The binary variables z_c[t,j,k] enforce exactly one cell is active
and bound q_c to the correct power band and SoC segment.

MILP variable layout (per interval t)
--------------------------------------
  p_c[t]        : fleet charge power (kW), ≥ 0
  p_d[t]        : fleet discharge power (kW), ≥ 0
  e[t]          : fleet SoC (kWh)
  z_c[t,j,k]   : 1 if charging in cell (power band j, SoC segment k)     [binary]
  z_d[t,j,k]   : 1 if discharging in cell (power band j, SoC segment k)  [binary]
  z_idle[t,k]  : 1 if idle in SoC segment k                              [binary]
  q_c[t,j,k]   : charge power allocated to cell (j,k) (kW)
  q_d[t,j,k]   : discharge power allocated to cell (j,k) (kW)

Key constraints (see solve_day_ahead_milp for full LP matrix)
--------------------------------------------------------------
  [one-cell]  Σ_{j,k} z_c + Σ_{j,k} z_d + Σ_k z_idle = 1    (equality)
  [SoC-lo-k]  e[t] ≥ e_break[k]   − M×(1 − w[t,k])           (big-M lower)
  [SoC-hi-k]  e[t] ≤ e_break[k+1] + M×(1 − w[t,k])           (big-M upper)
               where  w[t,k] = Σ_j z_c[t,j,k] + Σ_j z_d[t,j,k] + z_idle[t,k]
  [psum-c]    Σ_{j,k} q_c[t,j,k] = p_c[t]
  [psum-d]    Σ_{j,k} q_d[t,j,k] = p_d[t]
  [qcap-c]    q_c[t,j,k] ≤ p_hi_c[j,k] × z_c[t,j,k]
  [qcap-d]    q_d[t,j,k] ≤ p_hi_d[j,k] × z_d[t,j,k]
  [soc-upd]   e[t] = e[t-1]
                   + Σ_{j,k} η_c[j,k] × q_c[t,j,k] × dt_h
                   − Σ_{j,k} (1/η_d[j,k]) × q_d[t,j,k] × dt_h
                   − Σ_k sb_mech[k] × w[t,k] × dt_h

All terms are linear in the decision variables.

Dependencies
------------
  pip install numpy scipy matplotlib
  fess_unit.py, efficiency_models.py, standby_losses.py (existing codebase)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field

from efficiency_models import (
    MachineParams, MachineEfficiency,
    InverterParams, InverterEfficiency,
    FESSEfficiencyModel,
)
from standby_losses import StandbyLossParams
from fess_unit import FESSParams


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GridCell:
    """
    A single cell in the Power × SoC efficiency grid.

    Represents the constant efficiency and power bounds applicable when:
      - power is in [p_lo_kw, p_hi_kw]
      - SoC    is in [e_lo_kwh, e_hi_kwh]

    Used for both charge and discharge directions (stored separately).
    """
    j: int           # power band index
    k: int           # SoC segment index
    # Power band (unit level, kW)
    p_lo_kw:  float
    p_hi_kw:  float
    p_mid_pu: float  # representative per-unit power for efficiency evaluation
    # SoC segment (unit level, kWh)
    e_lo_kwh: float
    e_hi_kwh: float
    sr_mid:   float  # representative speed ratio
    # Efficiency constants at (p_mid_pu, sr_mid)
    eta_charge:    float
    eta_discharge: float
    # Discharge physical limit in this cell: min(p_hi_kw, P_rated × sr_mid)
    p_max_discharge_kw: float


@dataclass
class SoCSegment:
    """One segment along the SoC axis — carries standby loss constants."""
    k:                int
    e_lo_kwh:         float
    e_hi_kwh:         float
    sr_mid:           float   # sqrt((e_lo+e_hi) / (2 × E_rated))
    standby_mech_kw:  float   # mechanical loss at sr_mid (drains SoC)
    standby_aux_kw:   float   # auxiliary loss at sr_mid (drawn from grid)
    p_available_kw:   float   # P_rated × sr_mid (physical discharge limit)


@dataclass
class PiecewiseFESSParams:
    """
    Piecewise-linearized constants for a single FESS unit.

    charge_cells[j][k]    — GridCell for charging in power band j, SoC segment k
    discharge_cells[j][k] — GridCell for discharging in power band j, SoC segment k
    soc_segments[k]       — SoCSegment k (standby, P_available)

    Charge and discharge grids are kept separate because:
      - η_motor ≠ η_generator (different loss formulas)
      - Discharge power is physically limited by P_available = P_rated × sr_mid[k]
    """
    n_soc_segments:   int
    n_power_segments: int
    soc_breakpoints_kwh:   np.ndarray   # shape (K_e + 1,)
    power_breakpoints_kw:  np.ndarray   # shape (K_p + 1,)
    charge_cells:     list              # [K_p][K_e] of GridCell
    discharge_cells:  list              # [K_p][K_e] of GridCell
    soc_segments:     list              # [K_e] of SoCSegment
    # Unit-level energy bounds
    rated_energy_kwh: float
    min_energy_kwh:   float
    # Diagnostics
    opt_power_pu:      float
    eta_charge_avg:    float
    eta_discharge_avg: float
    standby_loss_kw:   float   # speed-averaged mechanical standby (for reference)
    auxiliary_load_kw: float   # speed-averaged auxiliary load (for reference)

    def cell(self, j: int, k: int, direction: str = "charge") -> GridCell:
        """Return a single cell by (j, k) index."""
        if direction == "charge":
            return self.charge_cells[j][k]
        return self.discharge_cells[j][k]

    def eta_grid(self, direction: str = "charge") -> np.ndarray:
        """Return K_p × K_e array of efficiency constants."""
        cells = self.charge_cells if direction == "charge" else self.discharge_cells
        return np.array([[cells[j][k].eta_charge if direction == "charge"
                          else cells[j][k].eta_discharge
                          for k in range(self.n_soc_segments)]
                         for j in range(self.n_power_segments)])

    def summary(self) -> dict:
        eta_c_grid = self.eta_grid("charge")
        eta_d_grid = self.eta_grid("discharge")
        return {
            "n_soc_segments":           self.n_soc_segments,
            "n_power_segments":         self.n_power_segments,
            "rated_energy_kwh":         round(self.rated_energy_kwh, 2),
            "min_energy_kwh":           round(self.min_energy_kwh, 2),
            "soc_breakpoints_kwh":      [round(x, 2) for x in self.soc_breakpoints_kwh],
            "power_breakpoints_kw":     [round(x, 2) for x in self.power_breakpoints_kw],
            "eta_charge_grid (K_p x K_e)":    np.round(eta_c_grid, 5).tolist(),
            "eta_discharge_grid (K_p x K_e)": np.round(eta_d_grid, 5).tolist(),
            "standby_mech_kw per segment":
                [round(s.standby_mech_kw, 5) for s in self.soc_segments],
            "standby_aux_kw per segment":
                [round(s.standby_aux_kw, 5) for s in self.soc_segments],
            "p_available_kw per segment":
                [round(s.p_available_kw, 2) for s in self.soc_segments],
            "eta_charge_avg":           f"{self.eta_charge_avg:.4%}",
            "eta_discharge_avg":        f"{self.eta_discharge_avg:.4%}",
            "opt_power_pu":             round(self.opt_power_pu, 4),
        }


@dataclass
class PiecewiseFleetParams:
    """
    Piecewise-linearized constants for a fleet of N identical units.

    Power bounds and energy bounds are scaled by N.
    Efficiencies are intensive (same as unit).
    Standby losses scale by N.

    The charge_cells and discharge_cells store fleet-level power bounds
    (p_lo_kw × N, p_hi_kw × N, p_max_discharge_kw × N) while efficiencies
    and sr_mid remain unit-level.
    """
    n_soc_segments:   int
    n_power_segments: int
    soc_breakpoints_kwh:  np.ndarray   # fleet-level (× N)
    power_breakpoints_kw: np.ndarray   # fleet-level (× N)
    charge_cells:     list             # [K_p][K_e] GridCell with fleet power bounds
    discharge_cells:  list             # [K_p][K_e] GridCell with fleet power bounds
    soc_segments:     list             # [K_e] SoCSegment with fleet energy + standby
    # Fleet energy bounds
    fleet_rated_energy_kwh: float
    fleet_min_energy_kwh:   float
    fleet_usable_energy_kwh: float
    # Grid interface
    transformer_capacity_kw:      float
    eta_transformer:              float
    fleet_auxiliary_kw:           float
    effective_charge_limit_kw:    float
    effective_discharge_limit_kw: float
    # Diagnostics
    n_units:           int
    opt_power_pu:      float
    eta_charge_avg:    float
    eta_discharge_avg: float

    def summary(self) -> dict:
        n_cells = self.n_power_segments * self.n_soc_segments
        return {
            "n_units":                        self.n_units,
            "n_soc_segments":                 self.n_soc_segments,
            "n_power_segments":               self.n_power_segments,
            "n_cells_per_direction":          n_cells,
            "fleet_rated_energy_kwh":         round(self.fleet_rated_energy_kwh, 1),
            "fleet_min_energy_kwh":           round(self.fleet_min_energy_kwh, 1),
            "fleet_usable_energy_kwh":        round(self.fleet_usable_energy_kwh, 1),
            "fleet_power_breakpoints_kw":
                [round(x, 1) for x in self.power_breakpoints_kw],
            "fleet_soc_breakpoints_kwh":
                [round(x, 1) for x in self.soc_breakpoints_kwh],
            "effective_charge_limit_kw":      round(self.effective_charge_limit_kw,  1),
            "effective_discharge_limit_kw":   round(self.effective_discharge_limit_kw, 1),
            "transformer_capacity_kw":        round(self.transformer_capacity_kw, 1),
            "eta_transformer":                f"{self.eta_transformer:.4%}",
            "fleet_auxiliary_kw":             round(self.fleet_auxiliary_kw, 3),
            "standby_mech_kw per segment":
                [round(s.standby_mech_kw, 4) for s in self.soc_segments],
            "eta_charge_avg":                 f"{self.eta_charge_avg:.4%}",
            "eta_discharge_avg":              f"{self.eta_discharge_avg:.4%}",
        }


# ---------------------------------------------------------------------------
# Core linearization function — unit level
# ---------------------------------------------------------------------------

def piecewise_linearize_fess(
    params:           FESSParams,
    n_soc_segments:   int = 4,
    n_power_segments: int = 3,
    n_speed_points:   int = 200,
) -> PiecewiseFESSParams:
    """
    Build a K_p × K_e piecewise-linearized efficiency model from full physics.

    Parameters
    ----------
    params : FESSParams
    n_soc_segments : int
        K_e — number of equal-width (kWh) SoC segments.
    n_power_segments : int
        K_p — number of equal-width (kW) power bands (0 to P_rated).
    n_speed_points : int
        Resolution for speed-average when finding the optimal power point.

    Returns
    -------
    PiecewiseFESSParams
    """
    eff_model = params.build_efficiency_model()
    sb_params  = params.standby_params

    sr_min = params.min_speed_ratio
    sr_max = params.max_speed_ratio
    P_rat  = params.rated_power_kw
    E_rat  = params.rated_energy_kwh

    e_min = E_rat * sr_min ** 2
    e_max = E_rat * sr_max ** 2

    # ------------------------------------------------------------------
    # Find speed-averaged optimal power (same logic as linearize_fess)
    # ------------------------------------------------------------------
    speed_grid = np.linspace(sr_min, sr_max, n_speed_points)
    power_grid = np.linspace(0.05, 1.0, 200)
    mean_rt = np.array([
        np.mean([eff_model.eta_roundtrip(p_pu, sr) for sr in speed_grid])
        for p_pu in power_grid
    ])
    opt_power_pu = float(power_grid[int(np.argmax(mean_rt))])

    # ------------------------------------------------------------------
    # SoC breakpoints: equal-width in kWh (natural for LP state variable)
    # ------------------------------------------------------------------
    soc_breakpoints = np.linspace(e_min, e_max, n_soc_segments + 1)

    # ------------------------------------------------------------------
    # Power breakpoints: equal-width in kW from 0 to P_rated
    # ------------------------------------------------------------------
    power_breakpoints = np.linspace(0.0, P_rat, n_power_segments + 1)

    # Vacuum pump duty-cycle average
    vac_duty  = sb_params.vacuum_on_duration_s / sb_params.vacuum_cycle_period_s
    vac_avg_w = sb_params.p_vacuum_pump_w * vac_duty

    # ------------------------------------------------------------------
    # Build SoC segments
    # ------------------------------------------------------------------
    soc_segments: list[SoCSegment] = []
    for k in range(n_soc_segments):
        e_lo  = soc_breakpoints[k]
        e_hi  = soc_breakpoints[k + 1]
        e_mid = 0.5 * (e_lo + e_hi)
        sr_mid = float(np.clip(np.sqrt(e_mid / E_rat), sr_min, sr_max))

        # Mechanical standby losses — drain flywheel kinetic energy
        sb_mech_w = (
            sb_params.k_aero_w            * sr_mid ** 3
            + sb_params.k_tmb_eddy_w      * sr_mid ** 2
            + sb_params.k_rmb_eddy_sync_w * sr_mid ** 2
        )
        # Auxiliary standby losses — drawn from grid
        sb_aux_w = (
            sb_params.p_tmb_bias_w
            + sb_params.k_rmb_bias_w * sr_mid ** 2
            + sb_params.p_rmb_eddy_pwm_w
            + sb_params.p_cooling_w
            + vac_avg_w
        )
        soc_segments.append(SoCSegment(
            k                 = k,
            e_lo_kwh          = e_lo,
            e_hi_kwh          = e_hi,
            sr_mid            = sr_mid,
            standby_mech_kw   = sb_mech_w / 1000.0,
            standby_aux_kw    = sb_aux_w  / 1000.0,
            p_available_kw    = P_rat * sr_mid,
        ))

    # ------------------------------------------------------------------
    # Build K_p × K_e charge and discharge cell grids
    # ------------------------------------------------------------------
    charge_cells:    list[list[GridCell]] = []
    discharge_cells: list[list[GridCell]] = []

    for j in range(n_power_segments):
        p_lo    = power_breakpoints[j]
        p_hi    = power_breakpoints[j + 1]
        p_mid_kw = 0.5 * (p_lo + p_hi)
        p_mid_pu = p_mid_kw / P_rat if P_rat > 0 else 0.0

        row_c: list[GridCell] = []
        row_d: list[GridCell] = []

        for k in range(n_soc_segments):
            seg    = soc_segments[k]
            sr_mid = seg.sr_mid

            # Clamp p_mid_pu to avoid evaluating at exactly 0 (numerical guard)
            p_eval = max(p_mid_pu, 0.01)

            eta_c = eff_model.eta_charge(p_eval, sr_mid)
            eta_d = eff_model.eta_discharge(p_eval, sr_mid)

            # Discharge physical limit in this cell:
            # cannot exceed the flywheel's available power at sr_mid
            p_max_d = float(min(p_hi, seg.p_available_kw))

            cell_c = GridCell(
                j=j, k=k,
                p_lo_kw=p_lo, p_hi_kw=p_hi,
                p_mid_pu=p_eval,
                e_lo_kwh=seg.e_lo_kwh, e_hi_kwh=seg.e_hi_kwh,
                sr_mid=sr_mid,
                eta_charge=eta_c, eta_discharge=eta_d,
                p_max_discharge_kw=p_max_d,
            )
            # Discharge cell is identical in structure — efficiencies are
            # the same object (motor vs generator is baked into the η formulas)
            cell_d = GridCell(
                j=j, k=k,
                p_lo_kw=p_lo, p_hi_kw=p_hi,
                p_mid_pu=p_eval,
                e_lo_kwh=seg.e_lo_kwh, e_hi_kwh=seg.e_hi_kwh,
                sr_mid=sr_mid,
                eta_charge=eta_c, eta_discharge=eta_d,
                p_max_discharge_kw=p_max_d,
            )
            row_c.append(cell_c)
            row_d.append(cell_d)

        charge_cells.append(row_c)
        discharge_cells.append(row_d)

    # ------------------------------------------------------------------
    # Average η and standby for backward-compat diagnostics
    # ------------------------------------------------------------------
    eta_c_vals = [charge_cells[j][k].eta_charge
                  for j in range(n_power_segments)
                  for k in range(n_soc_segments)]
    eta_d_vals = [discharge_cells[j][k].eta_discharge
                  for j in range(n_power_segments)
                  for k in range(n_soc_segments)]
    avg_eta_c  = float(np.mean(eta_c_vals))
    avg_eta_d  = float(np.mean(eta_d_vals))
    avg_sb_mech = float(np.mean([s.standby_mech_kw for s in soc_segments]))
    avg_sb_aux  = float(np.mean([s.standby_aux_kw  for s in soc_segments]))

    return PiecewiseFESSParams(
        n_soc_segments    = n_soc_segments,
        n_power_segments  = n_power_segments,
        soc_breakpoints_kwh  = soc_breakpoints,
        power_breakpoints_kw = power_breakpoints,
        charge_cells      = charge_cells,
        discharge_cells   = discharge_cells,
        soc_segments      = soc_segments,
        rated_energy_kwh  = e_max,
        min_energy_kwh    = e_min,
        opt_power_pu      = opt_power_pu,
        eta_charge_avg    = avg_eta_c,
        eta_discharge_avg = avg_eta_d,
        standby_loss_kw   = avg_sb_mech,
        auxiliary_load_kw = avg_sb_aux,
    )


# ---------------------------------------------------------------------------
# Fleet-level scaling
# ---------------------------------------------------------------------------

def piecewise_linearize_fleet(
    unit_params:  FESSParams,
    grid_params,               # GridInterfaceParams from fess_plant
    n_units:      int  = 50,
    n_soc_segments:   int = 4,
    n_power_segments: int = 3,
    n_speed_points:   int = 200,
) -> PiecewiseFleetParams:
    """
    Derive piecewise-linearized fleet constants by scaling the unit model.

    Power bounds scale by N (extensive).
    Energy bounds scale by N (extensive).
    Efficiencies are unchanged (intensive).
    Standby losses scale by N.

    The slope of the power-SoC relationship is intensive — it does not
    change with N because both numerator (kW) and denominator (kWh) scale
    by the same N.

    Parameters
    ----------
    unit_params  : FESSParams
    grid_params  : GridInterfaceParams
    n_units      : int
    n_soc_segments, n_power_segments, n_speed_points : see piecewise_linearize_fess
    """
    unit = piecewise_linearize_fess(
        unit_params, n_soc_segments, n_power_segments, n_speed_points
    )

    # Transformer capacity (derated by power factor and transformer efficiency)
    transformer_capacity_kw = (
        grid_params.transformer_capacity_kva
        * grid_params.power_factor
        * grid_params.transformer_efficiency
    )
    fleet_auxiliary_kw = (
        n_units * unit.auxiliary_load_kw
        + grid_params.auxiliary_load_kw
    )

    fleet_p_max_kw     = n_units * unit.power_breakpoints_kw[-1]
    effective_charge   = min(fleet_p_max_kw,
                             transformer_capacity_kw - fleet_auxiliary_kw)
    effective_discharge = min(fleet_p_max_kw, transformer_capacity_kw)

    # Scale SoC breakpoints and energy bounds by N
    fleet_soc_bps  = unit.soc_breakpoints_kwh  * n_units
    fleet_pwr_bps  = unit.power_breakpoints_kw * n_units

    # Scale SoC segments
    fleet_soc_segs = [
        SoCSegment(
            k               = s.k,
            e_lo_kwh        = s.e_lo_kwh        * n_units,
            e_hi_kwh        = s.e_hi_kwh        * n_units,
            sr_mid          = s.sr_mid,             # intensive
            standby_mech_kw = s.standby_mech_kw * n_units,
            standby_aux_kw  = s.standby_aux_kw  * n_units,
            p_available_kw  = s.p_available_kw  * n_units,
        )
        for s in unit.soc_segments
    ]

    # Scale grid cells — power bounds × N, efficiencies unchanged
    def _scale_row(row: list[GridCell]) -> list[GridCell]:
        return [
            GridCell(
                j                  = c.j,
                k                  = c.k,
                p_lo_kw            = c.p_lo_kw            * n_units,
                p_hi_kw            = c.p_hi_kw            * n_units,
                p_mid_pu           = c.p_mid_pu,           # intensive
                e_lo_kwh           = c.e_lo_kwh            * n_units,
                e_hi_kwh           = c.e_hi_kwh            * n_units,
                sr_mid             = c.sr_mid,             # intensive
                eta_charge         = c.eta_charge,         # intensive
                eta_discharge      = c.eta_discharge,      # intensive
                p_max_discharge_kw = c.p_max_discharge_kw * n_units,
            )
            for c in row
        ]

    fleet_charge_cells    = [_scale_row(row) for row in unit.charge_cells]
    fleet_discharge_cells = [_scale_row(row) for row in unit.discharge_cells]

    # Apply transformer cap as a hard ceiling on per-cell upper bounds
    for row in fleet_charge_cells:
        for c in row:
            c.p_hi_kw = min(c.p_hi_kw, effective_charge)
    for row in fleet_discharge_cells:
        for c in row:
            c.p_hi_kw           = min(c.p_hi_kw, effective_discharge)
            c.p_max_discharge_kw = min(c.p_max_discharge_kw, effective_discharge)

    fleet_e_max = unit.rated_energy_kwh * n_units
    fleet_e_min = unit.min_energy_kwh   * n_units

    return PiecewiseFleetParams(
        n_soc_segments            = n_soc_segments,
        n_power_segments          = n_power_segments,
        soc_breakpoints_kwh       = fleet_soc_bps,
        power_breakpoints_kw      = fleet_pwr_bps,
        charge_cells              = fleet_charge_cells,
        discharge_cells           = fleet_discharge_cells,
        soc_segments              = fleet_soc_segs,
        fleet_rated_energy_kwh    = fleet_e_max,
        fleet_min_energy_kwh      = fleet_e_min,
        fleet_usable_energy_kwh   = fleet_e_max - fleet_e_min,
        transformer_capacity_kw   = transformer_capacity_kw,
        eta_transformer           = grid_params.transformer_efficiency,
        fleet_auxiliary_kw        = fleet_auxiliary_kw,
        effective_charge_limit_kw = effective_charge,
        effective_discharge_limit_kw = effective_discharge,
        n_units                   = n_units,
        opt_power_pu              = unit.opt_power_pu,
        eta_charge_avg            = unit.eta_charge_avg,
        eta_discharge_avg         = unit.eta_discharge_avg,
    )


# ---------------------------------------------------------------------------
# Diagnostic plotting
# ---------------------------------------------------------------------------

def plot_piecewise_linearization(
    params:    FESSParams,
    pw:        PiecewiseFESSParams,
    save:      bool = True,
    show:      bool = True,
) -> plt.Figure:
    """
    Four-panel diagnostic figure comparing the physics model to the
    piecewise linearization.

    Panel 1 — Round-trip η heatmap (charge direction) with K_p × K_e
               cell grid overlaid; cell-constant η shown as colour patches.
    Panel 2 — Round-trip η heatmap (discharge direction) with grid overlay.
    Panel 3 — Standby mechanical + auxiliary loss vs SoC: physics curve
               and K_e segment step approximation.
    Panel 4 — Available discharge power vs SoC: physics curve (√SoC) and
               exact K_e segment step values (no regression needed).
    """
    eff_model  = params.build_efficiency_model()
    sb_params  = params.standby_params
    sr_min     = params.min_speed_ratio
    sr_max     = params.max_speed_ratio
    P_rat      = params.rated_power_kw
    E_rat      = params.rated_energy_kwh
    K_p        = pw.n_power_segments
    K_e        = pw.n_soc_segments

    sr_range   = np.linspace(sr_min, sr_max, 300)
    soc_range  = E_rat * sr_range ** 2        # kWh
    p_range_pu = np.linspace(0.02, 1.0, 300)  # 0–1

    fig = plt.figure(figsize=(18, 13))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.34)

    # Colour map for efficiency heatmaps
    cmap = plt.cm.RdYlGn

    for panel_idx, direction in enumerate(["charge", "discharge"]):
        ax = fig.add_subplot(gs[0, panel_idx])

        # --- Physics heatmap over (p_pu, SoC) grid ---
        P_grid, S_grid = np.meshgrid(p_range_pu, sr_range)
        if direction == "charge":
            eta_fn = np.vectorize(eff_model.eta_charge)
        else:
            eta_fn = np.vectorize(eff_model.eta_discharge)
        eta_2d = eta_fn(P_grid, S_grid) * 100.0  # percent

        cf = ax.contourf(
            p_range_pu * P_rat, soc_range, eta_2d,
            levels=20, cmap=cmap, alpha=0.85,
        )
        cs = ax.contour(
            p_range_pu * P_rat, soc_range, eta_2d,
            levels=8, colors="black", linewidths=0.4, alpha=0.3,
        )
        ax.clabel(cs, fmt="%.1f%%", fontsize=6)
        plt.colorbar(cf, ax=ax, label="η (%)", shrink=0.9)

        # --- Piecewise grid overlay ---
        pwr_bps = pw.power_breakpoints_kw
        soc_bps = pw.soc_breakpoints_kwh

        for j in range(K_p):
            for k in range(K_e):
                cell = pw.charge_cells[j][k] if direction == "charge" \
                       else pw.discharge_cells[j][k]
                eta_val = cell.eta_charge if direction == "charge" \
                          else cell.eta_discharge
                x0, x1 = pwr_bps[j], pwr_bps[j + 1]
                y0, y1 = soc_bps[k], soc_bps[k + 1]
                # White rectangle outline
                rect = plt.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=1.6, edgecolor="white", facecolor="none",
                )
                ax.add_patch(rect)
                # η value annotation
                ax.text(
                    0.5 * (x0 + x1), 0.5 * (y0 + y1),
                    f"{eta_val:.3f}",
                    ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold",
                )

        # Optimal power line
        ax.axvline(pw.opt_power_pu * P_rat, color="cyan",
                   linewidth=1.5, linestyle="--",
                   label=f"Opt power: {pw.opt_power_pu * P_rat:.0f} kW")

        ax.set_xlabel("Power (kW)")
        ax.set_ylabel("SoC (kWh)")
        dir_label = "Charge (Grid→Shaft)" if direction == "charge" \
                    else "Discharge (Shaft→Grid)"
        ax.set_title(
            f"η_{direction[:1]} surface + {K_p}×{K_e} PWL cells\n{dir_label}",
        )
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, P_rat)
        ax.set_ylim(soc_bps[0], soc_bps[-1])

    # ------------------------------------------------------------------
    # Panel 3: Standby losses vs SoC
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    vac_duty  = sb_params.vacuum_on_duration_s / sb_params.vacuum_cycle_period_s
    vac_avg_w = sb_params.p_vacuum_pump_w * vac_duty

    mech_arr = (
        sb_params.k_aero_w            * sr_range ** 3
        + sb_params.k_tmb_eddy_w      * sr_range ** 2
        + sb_params.k_rmb_eddy_sync_w * sr_range ** 2
    )
    aux_arr = (
        sb_params.p_tmb_bias_w
        + sb_params.k_rmb_bias_w * sr_range ** 2
        + sb_params.p_rmb_eddy_pwm_w
        + sb_params.p_cooling_w
        + vac_avg_w
    )

    ax3.fill_between(soc_range, (mech_arr + aux_arr) / 1000, 0,
                     color="#90CAF9", alpha=0.4, label="Aux standby (grid)")
    ax3.fill_between(soc_range, mech_arr / 1000, 0,
                     color="#1565C0", alpha=0.6, label="Mech standby (SoC drain)")
    ax3.plot(soc_range, (mech_arr + aux_arr) / 1000,
             color="black", linewidth=1.8, label="Total (W/1000)")

    # PWL step approximation
    for k, seg in enumerate(pw.soc_segments):
        total_kw = seg.standby_mech_kw + seg.standby_aux_kw
        color = "#E53935" if k % 2 == 0 else "#FF7043"
        ax3.hlines(
            total_kw, seg.e_lo_kwh, seg.e_hi_kwh,
            colors=color, linewidths=2.5,
            label=f"PWL k={k}: {total_kw * 1000:.1f} W" if k < 2 else None,
        )
        ax3.vlines(
            [seg.e_lo_kwh, seg.e_hi_kwh], 0, total_kw,
            colors=color, linewidths=0.8, linestyles=":",
        )

    ax3.set_xlabel("Fleet SoC (kWh)")
    ax3.set_ylabel("Standby power (kW)")
    ax3.set_title(
        f"Standby Losses vs SoC\n"
        f"Physics curve vs {K_e}-segment PWL step approximation",
    )
    ax3.legend(fontsize=7, loc="upper left", ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(soc_bps[0], soc_bps[-1])

    # ------------------------------------------------------------------
    # Panel 4: Available discharge power vs SoC
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    p_avail_arr = P_rat * sr_range

    ax4.plot(soc_range, p_avail_arr,
             color="#2196F3", linewidth=2.5,
             label="Physics: P = P_rated × √(SoC / E_rated)")

    # LP regression line (from linearize_fess for comparison)
    from linearized_physics import linearize_fess
    lp = linearize_fess(params)
    p_regression = lp.p_max_slope * soc_range + lp.p_max_intercept
    ax4.plot(soc_range, p_regression,
             color="orange", linewidth=2, linestyle="--",
             label=f"Current LP: linear regression (R²={lp.p_max_r2:.5f})",
             alpha=0.8)

    # PWL exact steps
    for k, seg in enumerate(pw.soc_segments):
        color = "#43A047" if k % 2 == 0 else "#81C784"
        ax4.hlines(
            seg.p_available_kw, seg.e_lo_kwh, seg.e_hi_kwh,
            colors=color, linewidths=2.5,
            label=f"PWL k={k}: {seg.p_available_kw:.1f} kW" if k < 2 else None,
        )
        ax4.vlines(
            [seg.e_lo_kwh, seg.e_hi_kwh], 0, seg.p_available_kw,
            colors=color, linewidths=0.8, linestyles=":",
        )

    ax4.set_xlabel("SoC (kWh)")
    ax4.set_ylabel("Available discharge power (kW)")
    ax4.set_title(
        f"Available Discharge Power vs SoC\n"
        f"Physics (√SoC) vs LP regression vs {K_e}-segment PWL exact steps",
    )
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(soc_bps[0], soc_bps[-1])

    plt.suptitle(
        f"FESS Piecewise Linearization  —  {K_p} power bands × {K_e} SoC segments  "
        f"({K_p * K_e} cells per direction)\n"
        f"Unit: {P_rat:.0f} kW / {params.rated_energy_kwh:.0f} kWh  "
        f"| sr ∈ [{params.min_speed_ratio:.2f}, {params.max_speed_ratio:.2f}]",
        fontsize=12, fontweight="bold",
    )

    if save:
        fname = "fess_piecewise_linearization.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Plot saved: {fname}")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Entry point — standalone inspection
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    from fess_unit import FESSParams
    from fess_plant import GridInterfaceParams

    unit_params = FESSParams(
        rated_power_kw   = 292.0,
        rated_energy_kwh = 1169.0,
        min_speed_ratio  = 0.20,
    )
    grid_params = GridInterfaceParams(
        transformer_capacity_kva = 20_000.0,
        transformer_efficiency   = 0.995,
        auxiliary_load_kw        = 50.0,
        power_factor             = 0.95,
    )
    N_UNITS       = 50
    N_SOC_SEGS    = 4
    N_POWER_SEGS  = 3

    print(f"Computing {N_POWER_SEGS}×{N_SOC_SEGS} piecewise linearization...")
    pw = piecewise_linearize_fess(unit_params, N_SOC_SEGS, N_POWER_SEGS)

    print("\n=== Single Unit — Piecewise Params ===")
    for k, v in pw.summary().items():
        print(f"  {k:40s}: {v}")

    print(f"\n=== Fleet ({N_UNITS} units) — Piecewise Fleet Params ===")
    fleet_pw = piecewise_linearize_fleet(
        unit_params, grid_params, n_units=N_UNITS,
        n_soc_segments=N_SOC_SEGS, n_power_segments=N_POWER_SEGS,
    )
    for k, v in fleet_pw.summary().items():
        print(f"  {k:40s}: {v}")

    print("\nGenerating diagnostic plot...")
    plot_piecewise_linearization(unit_params, pw, save=True, show=False)
    print("Done.")
