"""
lp_piecewise_example.py
=======================
MILP-based day-ahead energy arbitrage for a FESS fleet using piecewise
linearized physics (Power × SoC efficiency grid).

Extends lp_day_ahead_example.py by replacing the single-point global
efficiency constants with a K_p × K_e cell grid solved as a MILP.

Requires
--------
  scipy >= 1.9  (linprog `integrality` parameter for MILP via HiGHS)
  numpy, matplotlib

MILP variable layout
--------------------
For T intervals, K_p power bands, K_e SoC segments, n_jk = K_p × K_e:

  Block   Range                     Description
  ------  ----------------------    -------------------------------------------
  0       [0,       T)              p_c[t]      fleet charge power (kW)
  1       [T,       2T)             p_d[t]      fleet discharge power (kW)
  2       [2T,      3T)             e[t]        fleet SoC (kWh)
  3       [3T,      3T+T·n_jk)     z_c[t,j,k]  charging in cell (j,k) — binary
  4       [3T+T·n_jk, 3T+2T·n_jk) z_d[t,j,k]  discharging in cell (j,k) — binary
  5       [3T+2T·n_jk, 3T+2T·n_jk+T·K_e)  z_idle[t,k] idle in SoC seg k — binary
  6       […,  …+T·n_jk)           q_c[t,j,k]  charge power in cell (j,k) (kW)
  7       […,  …+T·n_jk)           q_d[t,j,k]  discharge power in cell (j,k) (kW)

  Total: T × (3 + 4·n_jk + K_e) variables
  Binaries: T × (2·n_jk + K_e)

Key constraints (all linear in the decision variables)
-------------------------------------------------------
  [one-cell]  ∀t: Σ_{j,k} z_c[t,j,k] + Σ_{j,k} z_d[t,j,k] + Σ_k z_idle[t,k] = 1
  [soc-lo-k]  ∀t,k: e[t] − M·w[t,k] ≥ e_break[k] − M           (big-M lower)
  [soc-hi-k]  ∀t,k: e[t] + M·w[t,k] ≤ e_break[k+1] + M         (big-M upper)
              where  w[t,k] = Σ_j z_c[t,j,k] + Σ_j z_d[t,j,k] + z_idle[t,k]
  [psum-c]    ∀t:   Σ_{j,k} q_c[t,j,k] = p_c[t]
  [psum-d]    ∀t:   Σ_{j,k} q_d[t,j,k] = p_d[t]
  [qcap-c]    ∀t,j,k: q_c[t,j,k] ≤ p_hi_c[j,k] · z_c[t,j,k]
  [qcap-d]    ∀t,j,k: q_d[t,j,k] ≤ p_hi_d[j,k] · z_d[t,j,k]
  [soc-upd]   ∀t:   e[t] = e[t-1]
                         + Σ_{j,k} η_c[j,k] · q_c[t,j,k] · dt_h
                         − Σ_{j,k} (1/η_d[j,k]) · q_d[t,j,k] · dt_h
                         − Σ_k sb_mech[k] · w[t,k] · dt_h
  [tx-import] ∀t:   (p_c[t] − p_d[t]) · η_tx ≤ tx_cap − aux_kw
  [tx-export] ∀t:   (p_d[t] − p_c[t]) · η_tx ≤ tx_cap
  [soc-ret]   e[T-1] ≥ e_initial  (optional)

Bilinearity resolution
----------------------
The term  p_c[t] × η_c(p[t], SoC[t])  is bilinear (product of two unknowns).
By routing power through cell-level auxiliary variables q_c[t,j,k], the SoC
update becomes  Σ_{j,k} η_c[j,k] × q_c[t,j,k] — a sum of (constant × var),
which is linear.  The constraint q_c ≤ p_hi · z_c combined with
Σ q_c = p_c ensures the routing is consistent.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy import __version__ as _scipy_version

from fess_unit import FESSParams
from fess_plant import GridInterfaceParams
from linearized_physics import linearize_fleet, FleetLinearizedParams
from piecewise_linearization import (
    PiecewiseFESSParams, PiecewiseFleetParams,
    piecewise_linearize_fleet,
)

# Check scipy version for MILP support
_scipy_major_minor = tuple(int(x) for x in _scipy_version.split(".")[:2])
if _scipy_major_minor < (1, 9):
    raise ImportError(
        f"scipy >= 1.9.0 required for MILP (integrality parameter). "
        f"Found {_scipy_version}.  Run: pip install --upgrade scipy"
    )


# ---------------------------------------------------------------------------
# Index helper
# ---------------------------------------------------------------------------

class _Idx:
    """
    Flat index calculator for the MILP variable array.

    Variable layout (see module docstring):
      p_c, p_d, e  — T each
      z_c, z_d     — T × K_p × K_e each  (binaries)
      z_idle       — T × K_e             (binaries)
      q_c, q_d     — T × K_p × K_e each  (continuous)
    """

    def __init__(self, T: int, K_p: int, K_e: int):
        self.T    = T
        self.K_p  = K_p
        self.K_e  = K_e
        self.n_jk = K_p * K_e

        # Block offsets
        self.OFF_PC    = 0
        self.OFF_PD    = T
        self.OFF_E     = 2 * T
        self.OFF_ZC    = 3 * T
        self.OFF_ZD    = 3 * T + T * self.n_jk
        self.OFF_ZIDLE = 3 * T + 2 * T * self.n_jk
        self.OFF_QC    = 3 * T + 2 * T * self.n_jk + T * K_e
        self.OFF_QD    = 3 * T + 2 * T * self.n_jk + T * K_e + T * self.n_jk
        self.n_vars    = 3 * T + 4 * T * self.n_jk + T * K_e

    # Continuous variables
    def pc(self, t):       return self.OFF_PC    + t
    def pd(self, t):       return self.OFF_PD    + t
    def e(self, t):        return self.OFF_E     + t
    def qc(self, t, j, k): return self.OFF_QC   + t * self.n_jk + j * self.K_e + k
    def qd(self, t, j, k): return self.OFF_QD   + t * self.n_jk + j * self.K_e + k

    # Binary variables
    def zc(self, t, j, k): return self.OFF_ZC   + t * self.n_jk + j * self.K_e + k
    def zd(self, t, j, k): return self.OFF_ZD   + t * self.n_jk + j * self.K_e + k
    def zidle(self, t, k): return self.OFF_ZIDLE + t * self.K_e + k

    def integrality_array(self) -> np.ndarray:
        """Return array: 1 for binary variables, 0 for continuous."""
        arr = np.zeros(self.n_vars, dtype=int)
        # z_c, z_d, z_idle are binary
        arr[self.OFF_ZC   : self.OFF_ZIDLE]          = 1   # z_c + z_d
        arr[self.OFF_ZIDLE: self.OFF_QC]              = 1   # z_idle
        return arr


# ---------------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------------

def solve_day_ahead_milp(
    fleet:               PiecewiseFleetParams,
    prices_eur_per_mwh:  np.ndarray,
    dt_h:                float = 1.0,
    e_initial_kwh:       float | None = None,
    enforce_soc_return:  bool  = True,
) -> dict:
    """
    Solve the day-ahead arbitrage MILP for one fleet dispatch day.

    Parameters
    ----------
    fleet : PiecewiseFleetParams
        MILP-ready fleet constants from piecewise_linearize_fleet().
    prices_eur_per_mwh : np.ndarray, shape (T,)
        Day-ahead prices in EUR/MWh for each interval.
    dt_h : float
        Interval length in hours (1.0 = hourly, 0.25 = 15-min).
    e_initial_kwh : float or None
        Starting SoC in kWh. Defaults to midpoint of usable range.
    enforce_soc_return : bool
        If True, add e[T-1] >= e_initial (prevents depleting over the day).

    Returns
    -------
    dict with keys:
        status        : str  — 'Optimal' | solver message
        revenue_eur   : float
        p_charge      : ndarray (kW)
        p_discharge   : ndarray (kW)
        soc           : ndarray (kWh), length T+1 (starts with e_initial)
        p_grid        : ndarray (kW), net grid import (+) / export (-)
        active_cells  : list of (t, 'charge'/'discharge'/'idle', j, k)
    """
    T   = len(prices_eur_per_mwh)
    K_p = fleet.n_power_segments
    K_e = fleet.n_soc_segments
    idx = _Idx(T, K_p, K_e)

    if e_initial_kwh is None:
        e_initial_kwh = (
            fleet.fleet_min_energy_kwh + 0.5 * fleet.fleet_usable_energy_kwh
        )

    # Shorthand
    eta_tx  = fleet.eta_transformer
    aux_kw  = fleet.fleet_auxiliary_kw
    tx_cap  = fleet.transformer_capacity_kw
    e_min   = fleet.fleet_min_energy_kwh
    e_max   = fleet.fleet_rated_energy_kwh
    p_c_lim = fleet.effective_charge_limit_kw
    p_d_lim = fleet.effective_discharge_limit_kw

    soc_bps = fleet.soc_breakpoints_kwh   # length K_e+1
    # Big-M: safe upper bound on SoC range
    M = float(e_max - e_min) + 1.0

    # ------------------------------------------------------------------
    # Objective: minimise  Σ price × dt_h × (p_c[t] − p_d[t])
    # (auxiliary cost is constant — not a decision variable)
    # ------------------------------------------------------------------
    prices_kwh = prices_eur_per_mwh / 1000.0
    c_obj      = np.zeros(idx.n_vars)
    for t in range(T):
        c_obj[idx.pc(t)] =  prices_kwh[t] * dt_h   # charge costs
        c_obj[idx.pd(t)] = -prices_kwh[t] * dt_h   # discharge earns

    # ------------------------------------------------------------------
    # Equality constraints
    # ------------------------------------------------------------------
    eq_rows, eq_rhs = [], []

    def add_eq(row: np.ndarray, rhs: float):
        eq_rows.append(row)
        eq_rhs.append(rhs)

    for t in range(T):
        # [one-cell] Σ z_c + Σ z_d + Σ z_idle = 1
        row = np.zeros(idx.n_vars)
        for j in range(K_p):
            for k in range(K_e):
                row[idx.zc(t, j, k)] = 1.0
                row[idx.zd(t, j, k)] = 1.0
        for k in range(K_e):
            row[idx.zidle(t, k)] = 1.0
        add_eq(row, 1.0)

        # [psum-c] Σ_{j,k} q_c[t,j,k] = p_c[t]
        row = np.zeros(idx.n_vars)
        row[idx.pc(t)] = -1.0
        for j in range(K_p):
            for k in range(K_e):
                row[idx.qc(t, j, k)] = 1.0
        add_eq(row, 0.0)

        # [psum-d] Σ_{j,k} q_d[t,j,k] = p_d[t]
        row = np.zeros(idx.n_vars)
        row[idx.pd(t)] = -1.0
        for j in range(K_p):
            for k in range(K_e):
                row[idx.qd(t, j, k)] = 1.0
        add_eq(row, 0.0)

        # [soc-upd]  e[t] = e[t-1]
        #                  + Σ η_c[j,k] · q_c[t,j,k] · dt_h
        #                  − Σ (1/η_d[j,k]) · q_d[t,j,k] · dt_h
        #                  − Σ_k sb_mech[k] · w[t,k] · dt_h
        # where w[t,k] = Σ_j z_c[t,j,k] + Σ_j z_d[t,j,k] + z_idle[t,k]
        row = np.zeros(idx.n_vars)
        row[idx.e(t)] = 1.0
        if t > 0:
            row[idx.e(t - 1)] = -1.0
        rhs = 0.0 if t > 0 else e_initial_kwh

        for j in range(K_p):
            for k in range(K_e):
                seg   = fleet.soc_segments[k]
                cell_c = fleet.charge_cells[j][k]
                cell_d = fleet.discharge_cells[j][k]

                row[idx.qc(t, j, k)] = -cell_c.eta_charge * dt_h
                row[idx.qd(t, j, k)] =  (1.0 / cell_d.eta_discharge) * dt_h

                # Standby: sb_mech[k] per segment (drains SoC when z is active)
                sb = seg.standby_mech_kw * dt_h
                row[idx.zc(t, j, k)]    += sb
                row[idx.zd(t, j, k)]    += sb

        for k in range(K_e):
            sb = fleet.soc_segments[k].standby_mech_kw * dt_h
            row[idx.zidle(t, k)] += sb

        add_eq(row, rhs)

    A_eq = np.array(eq_rows)
    b_eq = np.array(eq_rhs)

    # ------------------------------------------------------------------
    # Inequality constraints  A_ub @ x <= b_ub
    # ------------------------------------------------------------------
    ineq_rows, ineq_rhs = [], []

    def add_ineq(row: np.ndarray, rhs: float):
        ineq_rows.append(row)
        ineq_rhs.append(rhs)

    for t in range(T):
        # [soc-lo-k]  e[t] − M·w[t,k] ≥ e_break[k] − M
        # → −e[t] + M·w[t,k] ≤ M − e_break[k]
        for k in range(K_e):
            row = np.zeros(idx.n_vars)
            row[idx.e(t)] = -1.0
            for j in range(K_p):
                row[idx.zc(t, j, k)] = M
                row[idx.zd(t, j, k)] = M
            row[idx.zidle(t, k)] = M
            add_ineq(row, M - soc_bps[k])

        # [soc-hi-k]  e[t] + M·w[t,k] ≤ e_break[k+1] + M
        for k in range(K_e):
            row = np.zeros(idx.n_vars)
            row[idx.e(t)] = 1.0
            for j in range(K_p):
                row[idx.zc(t, j, k)] = M
                row[idx.zd(t, j, k)] = M
            row[idx.zidle(t, k)] = M
            add_ineq(row, soc_bps[k + 1] + M)

        # [qcap-c]  q_c[t,j,k] ≤ p_hi_c[j,k] · z_c[t,j,k]
        for j in range(K_p):
            for k in range(K_e):
                cell = fleet.charge_cells[j][k]
                p_hi = float(min(cell.p_hi_kw, p_c_lim))
                if p_hi <= 0:
                    continue
                row = np.zeros(idx.n_vars)
                row[idx.qc(t, j, k)] =  1.0
                row[idx.zc(t, j, k)] = -p_hi
                add_ineq(row, 0.0)

        # [qcap-d]  q_d[t,j,k] ≤ p_hi_d[j,k] · z_d[t,j,k]
        for j in range(K_p):
            for k in range(K_e):
                cell = fleet.discharge_cells[j][k]
                p_hi = float(min(cell.p_max_discharge_kw, p_d_lim))
                if p_hi <= 0:
                    continue
                row = np.zeros(idx.n_vars)
                row[idx.qd(t, j, k)] =  1.0
                row[idx.zd(t, j, k)] = -p_hi
                add_ineq(row, 0.0)

        # [tx-import]  (p_c − p_d) · η_tx ≤ tx_cap − aux_kw
        row = np.zeros(idx.n_vars)
        row[idx.pc(t)] =  eta_tx
        row[idx.pd(t)] = -eta_tx
        add_ineq(row, tx_cap - aux_kw)

        # [tx-export]  (p_d − p_c) · η_tx ≤ tx_cap
        row = np.zeros(idx.n_vars)
        row[idx.pc(t)] = -eta_tx
        row[idx.pd(t)] =  eta_tx
        add_ineq(row, tx_cap)

    # Optional: e[T-1] >= e_initial
    if enforce_soc_return:
        row = np.zeros(idx.n_vars)
        row[idx.e(T - 1)] = -1.0
        add_ineq(row, -e_initial_kwh)

    A_ub = np.array(ineq_rows)
    b_ub = np.array(ineq_rhs)

    # ------------------------------------------------------------------
    # Variable bounds
    # ------------------------------------------------------------------
    bounds = (
        [(0.0, p_c_lim)]  * T                  # p_c
        + [(0.0, p_d_lim)] * T                 # p_d
        + [(e_min, e_max)] * T                 # e
        + [(0.0, 1.0)]     * (T * idx.n_jk)   # z_c  (binary via integrality)
        + [(0.0, 1.0)]     * (T * idx.n_jk)   # z_d
        + [(0.0, 1.0)]     * (T * K_e)        # z_idle
    )
    # Continuous q_c and q_d bounds — capped per cell
    for t in range(T):
        for j in range(K_p):
            for k in range(K_e):
                p_hi_c = float(min(fleet.charge_cells[j][k].p_hi_kw, p_c_lim))
                bounds.append((0.0, p_hi_c))
    for t in range(T):
        for j in range(K_p):
            for k in range(K_e):
                p_hi_d = float(min(
                    fleet.discharge_cells[j][k].p_max_discharge_kw, p_d_lim
                ))
                bounds.append((0.0, max(p_hi_d, 0.0)))

    # ------------------------------------------------------------------
    # Integrality: 1 for binary variables, 0 for continuous
    # ------------------------------------------------------------------
    integrality = idx.integrality_array()

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    result = linprog(
        c           = c_obj,
        A_ub        = A_ub,
        b_ub        = b_ub,
        A_eq        = A_eq,
        b_eq        = b_eq,
        bounds      = bounds,
        integrality = integrality,
        method      = "highs",
    )

    if result.status != 0:
        return {
            "status":       result.message,
            "revenue_eur":  float("nan"),
            "p_charge":     np.zeros(T),
            "p_discharge":  np.zeros(T),
            "soc":          np.full(T + 1, e_initial_kwh),
            "p_grid":       np.zeros(T),
            "active_cells": [],
        }

    x   = result.x
    p_c = x[idx.OFF_PC  : idx.OFF_PC + T]
    p_d = x[idx.OFF_PD  : idx.OFF_PD + T]
    e   = x[idx.OFF_E   : idx.OFF_E  + T]
    soc = np.concatenate([[e_initial_kwh], e])

    p_grid = (p_c - p_d) * eta_tx + aux_kw

    revenue = float(np.sum(
        prices_kwh * dt_h * (p_d - p_c)
        - prices_kwh * dt_h * aux_kw
    ))

    # Decode active cells (diagnostic)
    active_cells = []
    for t in range(T):
        for j in range(K_p):
            for k in range(K_e):
                if x[idx.zc(t, j, k)] > 0.5:
                    active_cells.append((t, "charge", j, k))
                if x[idx.zd(t, j, k)] > 0.5:
                    active_cells.append((t, "discharge", j, k))
        # If neither z_c nor z_d is active, it's idle (don't need to check z_idle)

    return {
        "status":       "Optimal",
        "revenue_eur":  revenue,
        "p_charge":     p_c,
        "p_discharge":  p_d,
        "soc":          soc,
        "p_grid":       p_grid,
        "active_cells": active_cells,
    }


# ---------------------------------------------------------------------------
# LP comparison runner
# ---------------------------------------------------------------------------

def compare_lp_vs_milp(
    unit_params:         FESSParams,
    grid_params:         "GridInterfaceParams",
    prices_eur_per_mwh:  np.ndarray,
    n_units:             int   = 50,
    dt_h:                float = 1.0,
    n_soc_segments:      int   = 4,
    n_power_segments:    int   = 3,
    enforce_soc_return:  bool  = True,
) -> tuple[dict, dict, FleetLinearizedParams, PiecewiseFleetParams]:
    """
    Solve both the classic LP and the piecewise MILP on the same price profile.

    Returns
    -------
    result_lp, result_milp, fleet_lp, fleet_pw
    """
    from lp_day_ahead_example import solve_day_ahead_lp

    print("Building LP constants (global linearization)...")
    fleet_lp = linearize_fleet(unit_params, grid_params, n_units=n_units)

    print(f"Building MILP constants ({n_power_segments}×{n_soc_segments} grid)...")
    fleet_pw = piecewise_linearize_fleet(
        unit_params, grid_params,
        n_units=n_units,
        n_soc_segments=n_soc_segments,
        n_power_segments=n_power_segments,
    )

    e0 = fleet_lp.fleet_min_energy_kwh + 0.5 * fleet_lp.fleet_usable_energy_kwh

    print("Solving LP...")
    result_lp = solve_day_ahead_lp(
        fleet_lp, prices_eur_per_mwh, dt_h=dt_h,
        e_initial_kwh=e0, enforce_soc_return=enforce_soc_return,
    )
    print(f"  LP  status: {result_lp['status']}  "
          f"revenue: EUR {result_lp['revenue_eur']:.2f}")

    print("Solving MILP...")
    result_milp = solve_day_ahead_milp(
        fleet_pw, prices_eur_per_mwh, dt_h=dt_h,
        e_initial_kwh=e0, enforce_soc_return=enforce_soc_return,
    )
    print(f"  MILP status: {result_milp['status']}  "
          f"revenue: EUR {result_milp['revenue_eur']:.2f}")

    if result_lp["status"] == "Optimal" and result_milp["status"] == "Optimal":
        diff = result_milp["revenue_eur"] - result_lp["revenue_eur"]
        pct  = diff / abs(result_lp["revenue_eur"]) * 100 if result_lp["revenue_eur"] != 0 else float("nan")
        print(f"\n  Revenue difference (MILP - LP): EUR {diff:+.2f}  ({pct:+.2f}%)")
        print(f"  (Positive = MILP found higher revenue through better efficiency modelling)")

    return result_lp, result_milp, fleet_lp, fleet_pw


# ---------------------------------------------------------------------------
# Comparison plot
# ---------------------------------------------------------------------------

def plot_comparison(
    prices:     np.ndarray,
    result_lp:  dict,
    result_milp: dict,
    fleet_lp:   FleetLinearizedParams,
    fleet_pw:   PiecewiseFleetParams,
    dt_h:       float = 1.0,
    save:       bool  = True,
    show:       bool  = True,
) -> plt.Figure:
    """
    Six-panel comparison plot: LP vs MILP dispatch.

    Panel 1 — Day-ahead prices
    Panel 2 — Charge schedule: LP vs MILP
    Panel 3 — Discharge schedule: LP vs MILP
    Panel 4 — SoC trajectory: LP vs MILP
    Panel 5 — Net grid power: LP vs MILP
    Panel 6 — Active efficiency per interval: LP (constant) vs MILP (cell-varying)
    """
    T      = len(prices)
    hours  = np.arange(T)

    fig, axes = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

    lp_rev   = result_lp["revenue_eur"]
    milp_rev = result_milp["revenue_eur"]
    diff     = milp_rev - lp_rev
    fig.suptitle(
        f"LP vs MILP Day-Ahead Dispatch — Fleet of {fleet_pw.n_units} FESS units\n"
        f"LP revenue: EUR {lp_rev:.2f}  |  "
        f"MILP revenue: EUR {milp_rev:.2f}  |  "
        f"Δ = EUR {diff:+.2f}  ({fleet_pw.n_power_segments}×{fleet_pw.n_soc_segments} cells)",
        fontsize=11, fontweight="bold",
    )

    # Panel 1: Prices
    ax = axes[0]
    ax.step(hours, prices, where="post", color="#E65100", linewidth=2)
    ax.set_ylabel("Price\n(EUR/MWh)")
    ax.set_title("Day-Ahead Spot Prices")
    ax.grid(True, alpha=0.3)

    # Panel 2: Charge schedule
    ax = axes[1]
    ax.step(hours, result_lp["p_charge"]   / 1000, where="post",
            color="#90CAF9", linewidth=2.0, linestyle="--", label="LP")
    ax.step(hours, result_milp["p_charge"] / 1000, where="post",
            color="#1565C0", linewidth=2.0, label="MILP")
    ax.set_ylabel("Charge\n(MW)")
    ax.set_title("Fleet Charge Power")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Discharge schedule
    ax = axes[2]
    ax.step(hours, result_lp["p_discharge"]   / 1000, where="post",
            color="#A5D6A7", linewidth=2.0, linestyle="--", label="LP")
    ax.step(hours, result_milp["p_discharge"] / 1000, where="post",
            color="#2E7D32", linewidth=2.0, label="MILP")
    ax.set_ylabel("Discharge\n(MW)")
    ax.set_title("Fleet Discharge Power")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 4: SoC trajectory
    ax = axes[3]
    soc_h = np.arange(T + 1) * dt_h
    ax.step(soc_h, result_lp["soc"]   / 1000, where="post",
            color="#CE93D8", linewidth=2.0, linestyle="--", label="LP")
    ax.step(soc_h, result_milp["soc"] / 1000, where="post",
            color="#6A1B9A", linewidth=2.0, label="MILP")
    ax.axhline(fleet_pw.fleet_min_energy_kwh / 1000,
               color="red", linestyle=":", linewidth=1,
               label=f"SoC min ({fleet_pw.fleet_min_energy_kwh/1000:.0f} MWh)")
    ax.axhline(fleet_pw.fleet_rated_energy_kwh / 1000,
               color="gray", linestyle=":", linewidth=1,
               label=f"SoC max ({fleet_pw.fleet_rated_energy_kwh/1000:.0f} MWh)")
    ax.set_ylabel("Fleet SoC\n(MWh)")
    ax.set_title("Fleet State of Charge Trajectory")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 5: Net grid power
    ax = axes[4]
    ax.step(hours, result_lp["p_grid"]   / 1000, where="post",
            color="#80DEEA", linewidth=2.0, linestyle="--", label="LP")
    ax.step(hours, result_milp["p_grid"] / 1000, where="post",
            color="#00838F", linewidth=2.0, label="MILP")
    ax.axhline( fleet_pw.transformer_capacity_kw / 1000,
                color="red", linestyle=":", linewidth=1.2,
                label=f"Tx limit ±{fleet_pw.transformer_capacity_kw/1000:.1f} MW")
    ax.axhline(-fleet_pw.transformer_capacity_kw / 1000,
                color="red", linestyle=":", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net grid\n(MW, +=import)")
    ax.set_title("Net Grid Power")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 6: Active efficiency per interval
    ax = axes[5]
    t_range = np.arange(T)

    # LP: constant efficiency regardless of operating point
    lp_eta_rt = fleet_lp.eta_roundtrip

    # MILP: report the cell's round-trip efficiency for each active interval
    milp_eta_rt = np.full(T, float("nan"))
    for (t_act, direction, j, k) in result_milp["active_cells"]:
        cell_c = fleet_pw.charge_cells[j][k]
        cell_d = fleet_pw.discharge_cells[j][k]
        milp_eta_rt[t_act] = cell_c.eta_charge * cell_d.eta_discharge

    # Idle intervals: show as 1.0 (no loss)
    for t in range(T):
        p_c_t = result_milp["p_charge"][t]
        p_d_t = result_milp["p_discharge"][t]
        if p_c_t < 1.0 and p_d_t < 1.0:
            milp_eta_rt[t] = 1.0

    ax.axhline(lp_eta_rt * 100, color="#E65100", linewidth=2, linestyle="--",
               label=f"LP constant η_rt = {lp_eta_rt:.3%}")
    ax.scatter(t_range, milp_eta_rt * 100,
               color="#1565C0", s=40, zorder=5,
               label="MILP cell η_rt per interval")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Round-trip η (%)")
    ax.set_title("Active Round-Trip Efficiency per Interval\n"
                 "(LP: single constant; MILP: per-cell, varies with power and SoC)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(88, 101)

    axes[-1].set_xticks(range(0, T + 1, 2))
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save:
        fname = "lp_vs_milp_comparison.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")
    if show:
        plt.show()
    plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
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
    N_UNITS      = 50
    N_SOC_SEGS   = 4
    N_POWER_SEGS = 3

    # ------------------------------------------------------------------
    # Synthetic day-ahead price profile (same as lp_day_ahead_example.py)
    # ------------------------------------------------------------------
    prices_da = np.array([
         32,  30,  29,  28,   # 00-03  overnight low
         31,  38,  55,  72,   # 04-07  morning ramp-up
         85,  90,  88,  82,   # 08-11  morning peak
         70,  65,  62,  60,   # 12-15  midday dip
         63,  72,  88,  98,   # 16-19  evening ramp-up
        105, 100,  75,  45,   # 20-23  evening peak, wind ramp
    ], dtype=float)

    print("=" * 60)
    print(f" FESS MILP Dispatch — {N_UNITS} units, {N_POWER_SEGS}×{N_SOC_SEGS} grid")
    print("=" * 60)
    print(f"\nPrices (EUR/MWh): {prices_da.tolist()}")

    # ------------------------------------------------------------------
    # Solve and compare
    # ------------------------------------------------------------------
    result_lp, result_milp, fleet_lp, fleet_pw = compare_lp_vs_milp(
        unit_params, grid_params, prices_da,
        n_units=N_UNITS,
        n_soc_segments=N_SOC_SEGS,
        n_power_segments=N_POWER_SEGS,
        enforce_soc_return=True,
    )

    # ------------------------------------------------------------------
    # Print MILP constants
    # ------------------------------------------------------------------
    print("\n=== MILP Fleet Constants ===")
    for k, v in fleet_pw.summary().items():
        print(f"  {k:40s}: {v}")

    # ------------------------------------------------------------------
    # Print hourly schedules side-by-side
    # ------------------------------------------------------------------
    if result_milp["status"] == "Optimal":
        print("\nHourly schedule comparison:")
        header = (
            f"  {'Hr':>3}  {'Price':>7}  "
            f"{'LP P_c':>8}  {'LP P_d':>8}  {'LP SoC':>8}  "
            f"{'MILP P_c':>9}  {'MILP P_d':>9}  {'MILP SoC':>9}  "
            f"{'Cell(j,k)':>10}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        cell_map = {
            t: (j, k, direction)
            for (t, direction, j, k) in result_milp["active_cells"]
        }
        for t in range(len(prices_da)):
            cell_str = ""
            if t in cell_map:
                j, k, d = cell_map[t]
                cell_str = f"{d[0]}({j},{k})"
            print(
                f"  {t:3d}  {prices_da[t]:7.1f}  "
                f"{result_lp['p_charge'][t]:8.1f}  "
                f"{result_lp['p_discharge'][t]:8.1f}  "
                f"{result_lp['soc'][t+1]:8.1f}  "
                f"{result_milp['p_charge'][t]:9.1f}  "
                f"{result_milp['p_discharge'][t]:9.1f}  "
                f"{result_milp['soc'][t+1]:9.1f}  "
                f"{cell_str:>10}"
            )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating diagnostic plots...")

    from piecewise_linearization import piecewise_linearize_fess, plot_piecewise_linearization
    pw_unit = piecewise_linearize_fess(unit_params, N_SOC_SEGS, N_POWER_SEGS)
    plot_piecewise_linearization(unit_params, pw_unit, save=True, show=False)

    plot_comparison(
        prices_da, result_lp, result_milp, fleet_lp, fleet_pw,
        dt_h=1.0, save=True, show=False,
    )
    print("Done.")
