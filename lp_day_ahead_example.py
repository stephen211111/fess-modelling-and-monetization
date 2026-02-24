"""
lp_day_ahead_example.py
=======================
Example: Day-ahead energy arbitrage LP for a fleet of 50 FESS units.

Uses the LP-ready constants produced by linearized_physics.linearize_fleet()
to build and solve a standard linear programme with scipy.optimize.linprog.

Problem
-------
Given a 24-hour day-ahead price profile (e.g. hourly spot prices in EUR/MWh),
find the charge / discharge schedule that maximises arbitrage revenue subject
to the physical and grid constraints of the FESS fleet.

Variables (per hourly interval t = 0 … T-1):
    p_c[t]   : Fleet charge power  (kW), >= 0
    p_d[t]   : Fleet discharge power (kW), >= 0
    e[t]     : Fleet state-of-charge  (kWh) at end of interval t

Objective (maximise revenue = discharge income - charge cost - auxiliary cost):
    max  Σ_t dt_h * price[t] * (p_d[t] - p_c[t]) - dt_h * price[t] * aux_kw

Constraints per interval:
    1. SoC update (energy balance):
           e[t] = e[t-1]
                + p_c[t] * eta_c * dt_h
                - p_d[t] / eta_d * dt_h
                - standby_kw * dt_h

    2. Power-availability (linear in e[t], from least-squares regression):
           p_c[t] <= slope * e[t] + intercept
           p_d[t] <= slope * e[t] + intercept

    3. Rated-power cap (hard upper bound):
           p_c[t] <= effective_charge_limit_kw
           p_d[t] <= effective_discharge_limit_kw

    4. No simultaneous charge and discharge:
           p_c[t] + p_d[t] <= effective_charge_limit_kw

    5. Transformer / grid limit (applied to net grid power):
           (p_c[t] - p_d[t]) * eta_tx + aux_kw <= transformer_capacity_kw
           (p_d[t] - p_c[t]) * eta_tx          <= transformer_capacity_kw

    6. SoC bounds:
           fleet_min_energy_kwh <= e[t] <= fleet_rated_energy_kwh

    7. Boundary condition (optional — e[T-1] = e_initial):
           e[T-1] >= e_initial  (don't deplete the battery across the day)

Dependencies
------------
    pip install scipy numpy matplotlib

All FESS physics constants come from linearized_physics.py — no manual tuning
required.  Run this script directly to execute with a synthetic price profile.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from fess_unit import FESSParams
from fess_plant import GridInterfaceParams
from linearized_physics import linearize_fleet, FleetLinearizedParams


# ---------------------------------------------------------------------------
# Helper: build and solve the LP
# ---------------------------------------------------------------------------

def solve_day_ahead_lp(
    fleet: FleetLinearizedParams,
    prices_eur_per_mwh: np.ndarray,
    dt_h: float = 1.0,
    e_initial_kwh: float | None = None,
    enforce_soc_return: bool = True,
) -> dict:
    """
    Solve the day-ahead arbitrage LP for one fleet dispatch day.

    Parameters
    ----------
    fleet : FleetLinearizedParams
        LP-ready fleet constants from linearize_fleet().
    prices_eur_per_mwh : np.ndarray, shape (T,)
        Day-ahead prices in EUR/MWh for each interval.
    dt_h : float
        Interval length in hours (default 1.0 for hourly prices).
    e_initial_kwh : float or None
        Starting SoC in kWh.  Defaults to 50 % of usable range.
    enforce_soc_return : bool
        If True, add constraint e[T-1] >= e_initial (don't drain battery
        across the day).  Useful when the same fleet is dispatched daily.

    Returns
    -------
    dict with keys:
        status      : str — 'Optimal' | 'Infeasible' | solver message
        revenue_eur : float — net arbitrage revenue (EUR)
        p_charge    : np.ndarray (kW) — charge schedule
        p_discharge : np.ndarray (kW) — discharge schedule
        soc         : np.ndarray (kWh) — SoC trajectory (length T+1, starts with e_initial)
        p_grid      : np.ndarray (kW) — net grid import schedule (+ve = import)
    """
    T = len(prices_eur_per_mwh)

    if e_initial_kwh is None:
        e_initial_kwh = fleet.fleet_min_energy_kwh + 0.5 * fleet.fleet_usable_energy_kwh

    # Shorthand
    eta_c   = fleet.eta_charge
    eta_d   = fleet.eta_discharge
    eta_tx  = fleet.eta_transformer
    sb_kw   = fleet.fleet_standby_loss_kw
    aux_kw  = fleet.fleet_auxiliary_kw
    slope   = fleet.p_max_slope
    intcpt  = fleet.fleet_p_max_intercept
    p_c_lim = fleet.effective_charge_limit_kw
    p_d_lim = fleet.effective_discharge_limit_kw
    tx_cap  = fleet.transformer_capacity_kw
    e_min   = fleet.fleet_min_energy_kwh
    e_max   = fleet.fleet_rated_energy_kwh

    # ------------------------------------------------------------------
    # Decision variable layout (length = 3T):
    #   x[0:T]     = p_c[0..T-1]   charge power  (kW)
    #   x[T:2T]    = p_d[0..T-1]   discharge power (kW)
    #   x[2T:3T]   = e[0..T-1]     SoC at end of each interval (kWh)
    # ------------------------------------------------------------------
    n_vars = 3 * T

    # ------------------------------------------------------------------
    # Objective: minimise cost (scipy uses minimisation)
    #   Minimise  Σ price * dt_h * (p_c - p_d)
    #   (auxiliary cost is constant — ignored in optimisation, added back
    #    to revenue calculation at the end)
    # ------------------------------------------------------------------
    prices_kwh = prices_eur_per_mwh / 1000.0   # EUR/kWh
    c_obj = np.zeros(n_vars)
    c_obj[0:T]   =  prices_kwh * dt_h   # charge costs money  (+)
    c_obj[T:2*T] = -prices_kwh * dt_h   # discharge earns money (-)

    # ------------------------------------------------------------------
    # Equality constraints: SoC update equation for each interval
    #   e[t] - e[t-1] - eta_c * dt_h * p_c[t] + (1/eta_d) * dt_h * p_d[t]
    #   = -sb_kw * dt_h
    # where e[-1] = e_initial (known constant, moved to RHS).
    # ------------------------------------------------------------------
    # Build as A_eq @ x = b_eq (T rows)
    A_eq = np.zeros((T, n_vars))
    b_eq = np.full(T, -sb_kw * dt_h)

    for t in range(T):
        A_eq[t, t]       = -eta_c * dt_h      # p_c[t]
        A_eq[t, T + t]   =  (1.0 / eta_d) * dt_h  # p_d[t]
        A_eq[t, 2*T + t] =  1.0               # e[t]
        if t > 0:
            A_eq[t, 2*T + t - 1] = -1.0       # -e[t-1]
        else:
            b_eq[0] += e_initial_kwh           # -e[-1] moves to RHS with sign flip

    # ------------------------------------------------------------------
    # Inequality constraints (A_ub @ x <= b_ub)
    # ------------------------------------------------------------------
    ineq_rows = []
    ineq_rhs  = []

    def add_ineq(row, rhs):
        ineq_rows.append(row)
        ineq_rhs.append(rhs)

    for t in range(T):
        row_pc  = np.zeros(n_vars)
        row_pd  = np.zeros(n_vars)
        row_sum = np.zeros(n_vars)
        row_tx_import = np.zeros(n_vars)
        row_tx_export = np.zeros(n_vars)

        row_pc[t]         = 1.0
        row_pd[T + t]     = 1.0
        row_sum[t]        = 1.0
        row_sum[T + t]    = 1.0

        # 2. Power-availability: p_c[t] <= slope*e[t] + intercept
        row_c_avail = np.zeros(n_vars)
        row_c_avail[t]        =  1.0
        row_c_avail[2*T + t]  = -slope
        add_ineq(row_c_avail, intcpt)

        # 2. Power-availability: p_d[t] <= slope*e[t] + intercept
        row_d_avail = np.zeros(n_vars)
        row_d_avail[T + t]    =  1.0
        row_d_avail[2*T + t]  = -slope
        add_ineq(row_d_avail, intcpt)

        # 4. No simultaneous charge+discharge: p_c[t] + p_d[t] <= p_c_lim
        add_ineq(row_sum.copy(), p_c_lim)

        # 5a. Transformer import limit: (p_c - p_d)*eta_tx <= tx_cap - aux_kw
        row_tx_import[t]      =  eta_tx
        row_tx_import[T + t]  = -eta_tx
        add_ineq(row_tx_import, tx_cap - aux_kw)

        # 5b. Transformer export limit: (p_d - p_c)*eta_tx <= tx_cap
        row_tx_export[t]      = -eta_tx
        row_tx_export[T + t]  =  eta_tx
        add_ineq(row_tx_export, tx_cap)

    # Optional: SoC at end of day >= initial SoC (don't drain across days)
    if enforce_soc_return:
        row_soc_return = np.zeros(n_vars)
        row_soc_return[2*T + T - 1] = -1.0
        add_ineq(row_soc_return, -e_initial_kwh)

    A_ub = np.array(ineq_rows)
    b_ub = np.array(ineq_rhs)

    # ------------------------------------------------------------------
    # Variable bounds
    # ------------------------------------------------------------------
    bounds = (
        [(0.0, p_c_lim)] * T        # p_c
        + [(0.0, p_d_lim)] * T      # p_d
        + [(e_min, e_max)] * T      # e
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    result = linprog(
        c       = c_obj,
        A_ub    = A_ub,
        b_ub    = b_ub,
        A_eq    = A_eq,
        b_eq    = b_eq,
        bounds  = bounds,
        method  = "highs",
    )

    if result.status != 0:
        return {
            "status":      result.message,
            "revenue_eur": float("nan"),
            "p_charge":    np.zeros(T),
            "p_discharge": np.zeros(T),
            "soc":         np.full(T + 1, e_initial_kwh),
            "p_grid":      np.zeros(T),
        }

    p_c = result.x[0:T]
    p_d = result.x[T:2*T]
    e   = result.x[2*T:3*T]

    # Reconstruct SoC trajectory including initial value
    soc = np.concatenate([[e_initial_kwh], e])

    # Net grid import: positive = importing from grid (charging), negative = exporting (discharging)
    p_grid = (p_c - p_d) * eta_tx + aux_kw

    # Revenue: income from discharge - cost of charge - auxiliary cost
    revenue = float(np.sum(
        prices_kwh * dt_h * (p_d - p_c)
        - prices_kwh * dt_h * aux_kw
    ))

    return {
        "status":      "Optimal",
        "revenue_eur": revenue,
        "p_charge":    p_c,
        "p_discharge": p_d,
        "soc":         soc,
        "p_grid":      p_grid,
    }


# ---------------------------------------------------------------------------
# Helper: plot results
# ---------------------------------------------------------------------------

def plot_dispatch(
    prices: np.ndarray,
    result: dict,
    fleet: FleetLinearizedParams,
    dt_h: float = 1.0,
    save: bool = True,
    show: bool = True,
) -> None:
    """Four-panel plot of the optimised dispatch schedule."""
    T      = len(prices)
    hours  = np.arange(T)
    soc    = result["soc"]
    p_c    = result["p_charge"]
    p_d    = result["p_discharge"]
    p_grid = result["p_grid"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"Day-Ahead LP Dispatch — Fleet of {fleet.n_units} FESS units\n"
        f"Net revenue: EUR {result['revenue_eur']:.2f}  |  "
        f"RT efficiency: {fleet.eta_roundtrip:.2%}",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: Day-ahead prices
    ax = axes[0]
    ax.step(hours, prices, where="post", color="#E65100", linewidth=2)
    ax.set_ylabel("Price (EUR/MWh)")
    ax.set_title("Day-Ahead Spot Prices")
    ax.grid(True, alpha=0.3)

    # Panel 2: Charge / discharge schedule
    ax = axes[1]
    ax.bar(hours, p_c,  width=0.4, align="edge",        color="#1565C0", alpha=0.8, label="Charge (kW)")
    ax.bar(hours, -p_d, width=0.4, align="center",      color="#2E7D32", alpha=0.8, label="Discharge (kW)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Power (kW)\n(+ = charge, - = discharge)")
    ax.set_title("Fleet Charge / Discharge Schedule")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: State of charge
    ax = axes[2]
    soc_hours = np.arange(T + 1) * dt_h
    ax.step(soc_hours, soc / 1000, where="post", color="#6A1B9A", linewidth=2)
    ax.axhline(fleet.fleet_min_energy_kwh / 1000, color="red",   linestyle="--", linewidth=1,
               label=f"SoC min ({fleet.fleet_min_energy_kwh/1000:.0f} MWh)")
    ax.axhline(fleet.fleet_rated_energy_kwh / 1000, color="gray", linestyle="--", linewidth=1,
               label=f"SoC max ({fleet.fleet_rated_energy_kwh/1000:.0f} MWh)")
    ax.set_ylabel("Fleet SoC (MWh)")
    ax.set_title("Fleet State of Charge")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Net grid power
    ax = axes[3]
    ax.step(hours, p_grid / 1000, where="post", color="#00838F", linewidth=2)
    ax.fill_between(hours, p_grid / 1000, step="post",
                    where=(p_grid > 0), color="#00838F", alpha=0.3, label="Importing")
    ax.fill_between(hours, p_grid / 1000, step="post",
                    where=(p_grid < 0), color="#FF6F00", alpha=0.3, label="Exporting")
    ax.axhline(fleet.transformer_capacity_kw / 1000, color="red", linestyle=":",
               linewidth=1.5, label=f"Tx limit ({fleet.transformer_capacity_kw/1000:.1f} MW)")
    ax.axhline(-fleet.transformer_capacity_kw / 1000, color="red", linestyle=":", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Net grid power (MW)\n(+ = import, - = export)")
    ax.set_xlabel("Hour of day")
    ax.set_title("Net Grid Power (including auxiliary load)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(range(0, T + 1, 2))
    plt.tight_layout()

    if save:
        fig.savefig("lp_dispatch_result.png", dpi=150, bbox_inches="tight")
        print("Saved: lp_dispatch_result.png")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Build fleet LP constants from FESS physics
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
    N_UNITS = 50

    print("Linearizing FESS physics for LP...")
    fleet = linearize_fleet(unit_params, grid_params, n_units=N_UNITS)

    print(f"\nFleet LP constants ({N_UNITS} units):")
    for k, v in fleet.summary().items():
        print(f"  {k:35s}: {v}")

    # ------------------------------------------------------------------
    # 2. Synthetic day-ahead price profile (EUR/MWh)
    #    Two-peak pattern typical of Northern European day-ahead markets:
    #    low overnight, morning peak, midday dip, evening peak.
    # ------------------------------------------------------------------
    prices_da = np.array([
         32,  30,  29,  28,   # 00-03  overnight low
         31,  38,  55,  72,   # 04-07  morning ramp-up
         85,  90,  88,  82,   # 08-11  morning peak
         70,  65,  62,  60,   # 12-15  midday dip (solar generation)
         63,  72,  88,  98,   # 16-19  evening ramp-up
        105, 100,  75,  45,   # 20-23  evening peak, wind ramp
    ], dtype=float)

    print("\nDay-ahead prices (EUR/MWh):")
    print("  " + "  ".join(f"{p:5.1f}" for p in prices_da))

    # ------------------------------------------------------------------
    # 3. Solve LP
    # ------------------------------------------------------------------
    e0 = fleet.fleet_min_energy_kwh + 0.5 * fleet.fleet_usable_energy_kwh
    print(f"\nSolving LP: T={len(prices_da)} intervals, dt=1h, e_initial={e0:.0f} kWh ...")

    result = solve_day_ahead_lp(
        fleet                  = fleet,
        prices_eur_per_mwh     = prices_da,
        dt_h                   = 1.0,
        e_initial_kwh          = e0,
        enforce_soc_return     = True,
    )

    print(f"\nSolver status : {result['status']}")
    print(f"Net revenue   : EUR {result['revenue_eur']:.2f}")

    if result["status"] == "Optimal":
        print("\nHourly schedule:")
        print(f"  {'Hour':>4}  {'Price':>8}  {'P_charge':>10}  {'P_discharge':>12}  "
              f"{'SoC_end':>10}  {'P_grid':>10}")
        print("  " + "-" * 60)
        for t in range(len(prices_da)):
            print(f"  {t:4d}  {prices_da[t]:8.1f}  "
                  f"{result['p_charge'][t]:10.1f}  "
                  f"{result['p_discharge'][t]:12.1f}  "
                  f"{result['soc'][t+1]:10.1f}  "
                  f"{result['p_grid'][t]:10.1f}")

        # ------------------------------------------------------------------
        # 4. Plot results
        # ------------------------------------------------------------------
        import matplotlib
        matplotlib.use("Agg")
        plot_dispatch(prices_da, result, fleet, dt_h=1.0, save=True, show=False)
