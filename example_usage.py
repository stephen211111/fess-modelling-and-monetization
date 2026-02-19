"""
example_usage.py
================
Demonstrates the FESSUnit and FESSPlant classes with three scenarios:

  1. Single unit  - basic charge/discharge cycle
  2. 20-unit plant - aFRR regulation following a synthetic AGC signal
  3. 20-unit plant - energy arbitrage over a 24-hour price profile

Run:
    pip install numpy pandas matplotlib
    python example_usage.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fess_unit import FESSUnit, FESSParams
from fess_plant import (
    FESSPlant, GridInterfaceParams,
    DispatchStrategy, MarketService, RevenueServiceConfig,
)


# ============================================================
# SCENARIO 1: Single unit basic cycle
# ============================================================

def scenario_single_unit():
    print("\n" + "="*60)
    print("SCENARIO 1: Single FESS Unit - Basic Cycle")
    print("="*60)

    params = FESSParams(
        rated_power_kw   = 292.0,
        rated_energy_kwh = 1169.0,
        min_speed_ratio  = 0.20,
    )
    unit = FESSUnit(params, unit_id="FW-01", initial_soc_frac=0.5)

    dt = 1 / 60  # 1-minute steps

    # Charge at full power for 4 minutes
    print("\nCharging at 292 kW for 4 minutes...")
    for _ in range(4):
        snap = unit.step(power_setpoint_kw=292.0, dt_hours=dt)

    print(f"  After charge: SoC={snap.soc_kwh:.2f} kWh "
          f"({snap.soc_frac:.1%}), speed_ratio={snap.speed_ratio:.3f}")

    # Idle for 5 minutes (standby losses only)
    print("\nIdle for 5 minutes (standby losses)...")
    for _ in range(5):
        snap = unit.step(power_setpoint_kw=0.0, dt_hours=dt)

    print(f"  After idle:   SoC={snap.soc_kwh:.2f} kWh "
          f"({snap.soc_frac:.1%}), standby_loss={snap.standby_mechanical_kw + snap.standby_auxiliary_kw:.2f} kW")

    # Discharge at full available power for 4 minutes
    print("\nDischarging at max available power for 4 minutes...")
    for _ in range(4):
        snap = unit.step(power_setpoint_kw=-292.0, dt_hours=dt)

    print(f"  After discharge: SoC={snap.soc_kwh:.2f} kWh "
          f"({snap.soc_frac:.1%}), P_delivered={abs(snap.power_kw):.1f} kW")

    print("\n--- Unit Summary ---")
    for k, v in unit.summary().items():
        print(f"  {k}: {v}")

    return unit


# ============================================================
# SCENARIO 2: 20-unit plant following an aFRR AGC signal
# ============================================================

def scenario_afrr_plant():
    print("\n" + "="*60)
    print("SCENARIO 2: 20-Unit Plant - aFRR Regulation (1 hour)")
    print("="*60)

    # --- Build fleet ---
    params = FESSParams(
        rated_power_kw   = 292.0,
        rated_energy_kwh = 1169.0,
        min_speed_ratio  = 0.20,
    )
    units = [FESSUnit(params, f"FW-{i:02d}", initial_soc_frac=0.55)
             for i in range(20)]

    grid  = GridInterfaceParams(
        transformer_capacity_kva  = 10_000,
        auxiliary_load_kw         = 50.0,
        max_ramp_rate_kw_per_min  = 10_000,  # Fast ramp for regulation
    )
    plant = FESSPlant(units, grid, DispatchStrategy.SOC_BALANCED, "TERMINUS-AFRR")

    plant.configure_revenue_stack([
        RevenueServiceConfig(
            service        = MarketService.AFRR,
            enabled        = True,
            capacity_mw    = 5.0,
            price_per_mw_h = 18.0,   # $/MW/h (realistic 2024 price)
            priority       = 1,
        )
    ])

    # --- Synthetic AGC signal (realistic ±100% swings at 1-minute resolution) ---
    dt     = 1 / 60
    n_steps = 60  # 1 hour
    np.random.seed(42)

    # AGC signal: mean-reverting around zero, scaled to ±5 MW (plant rated)
    plant_rated_kw = plant.total_rated_power_kw
    agc_signal = np.zeros(n_steps)
    agc_signal[0] = 0.0
    for t in range(1, n_steps):
        agc_signal[t] = (
            0.7 * agc_signal[t-1]
            + 0.3 * np.random.normal(0, 0.3)
        )
    agc_signal = np.clip(agc_signal, -1.0, 1.0)
    agc_power_kw = agc_signal * plant_rated_kw * 0.4  # ±40% of rated

    print(f"\nPlant: {plant}")
    print(f"Simulating {n_steps} steps at {dt*60:.0f}-minute resolution...")

    for t in range(n_steps):
        plant.step(
            power_setpoint_kw = agc_power_kw[t],
            dt_hours          = dt,
            active_service    = MarketService.AFRR,
        )

    df = plant.to_dataframe()

    print("\n--- Plant Summary ---")
    for k, v in plant.plant_summary().items():
        print(f"  {k}: {v}")

    print(f"\n  Total revenue (1 hr): ${plant._cumulative_revenue:.2f}")
    print(f"  Annualised revenue:   ${plant._cumulative_revenue * 8760:.0f} (extrapolated)")

    return plant, df, agc_power_kw


# ============================================================
# SCENARIO 3: Energy arbitrage over 24-hour price profile
# ============================================================

def scenario_arbitrage_plant():
    print("\n" + "="*60)
    print("SCENARIO 3: 20-Unit Plant - Energy Arbitrage (24 hours)")
    print("="*60)

    params = FESSParams(
        rated_power_kw   = 292.0,
        rated_energy_kwh = 1169.0,
        min_speed_ratio  = 0.20,
    )
    units = [FESSUnit(params, f"FW-{i:02d}", initial_soc_frac=0.6)
             for i in range(20)]

    grid  = GridInterfaceParams(transformer_capacity_kva=10_000, auxiliary_load_kw=50.0)
    plant = FESSPlant(units, grid, DispatchStrategy.SOC_BALANCED, "TERMINUS-ARB")

    # --- Synthetic 24-hour price signal (15-minute intervals) ---
    dt       = 0.25  # 15-minute steps
    n_steps  = 96    # 24 hours
    hours    = np.arange(n_steps) * dt

    # Typical day-ahead price with morning and evening peaks
    price_signal = (
        40
        + 30 * np.sin(2 * np.pi * (hours - 7) / 24)   # Evening peak
        + 15 * np.sin(2 * np.pi * (hours - 7) / 12)   # Double-peak
        + np.random.RandomState(99).normal(0, 3, n_steps)
    )
    price_signal = np.maximum(price_signal, 5.0)  # Floor at $5/MWh

    # Simple threshold dispatch
    price_mean = np.mean(price_signal)
    charge_threshold    = price_mean * 0.80
    discharge_threshold = price_mean * 1.20

    print(f"\nPlant: {plant}")
    print(f"Price stats: mean=${price_mean:.1f}, "
          f"min=${price_signal.min():.1f}, max=${price_signal.max():.1f} /MWh")
    print(f"Charge below ${charge_threshold:.1f}, discharge above ${discharge_threshold:.1f} /MWh")

    for t in range(n_steps):
        price = price_signal[t]

        if price < charge_threshold:
            setpoint = plant.total_rated_power_kw  # Full charge
        elif price > discharge_threshold:
            setpoint = -plant.total_power_available_kw  # Full discharge
        else:
            setpoint = 0.0  # Hold

        plant.step(
            power_setpoint_kw = setpoint,
            dt_hours          = dt,
            active_service    = MarketService.ARBITRAGE,
            spot_price_per_mwh = price,
        )

    df = plant.to_dataframe()

    # Net revenue accounting for charge cost
    charge_cost = (df["grid_energy_consumed_kwh"] * (price_signal / 1000)).sum()
    discharge_revenue = (df["grid_energy_delivered_kwh"] * (price_signal / 1000)).sum()
    net_revenue = discharge_revenue - charge_cost

    print(f"\n  Discharge revenue: ${discharge_revenue:.2f}")
    print(f"  Charge cost:       ${charge_cost:.2f}")
    print(f"  Net arbitrage:     ${net_revenue:.2f}")
    print(f"  Auxiliary cost:    ${df['auxiliary_load_kwh'].sum() / 1000 * price_mean:.2f}")

    print("\n--- Plant Summary ---")
    for k, v in plant.plant_summary().items():
        print(f"  {k}: {v}")

    return plant, df, price_signal


# ============================================================
# Plotting
# ============================================================

def plot_results(
    unit_s1: FESSUnit,
    plant_s2: FESSPlant,
    df_s2: pd.DataFrame,
    agc_signal: np.ndarray,
    plant_s3: FESSPlant,
    df_s3: pd.DataFrame,
    price_signal: np.ndarray,
):
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ---- Plot 1: Single unit SoC and power over the basic cycle ----
    ax1a = fig.add_subplot(gs[0, 0])
    times_s1 = [s.time_h * 60 for s in unit_s1.history]  # convert to minutes
    soc_s1   = [s.soc_frac for s in unit_s1.history]
    power_s1 = [s.power_kw for s in unit_s1.history]

    ax1a.plot(times_s1, soc_s1, color="#2196F3", linewidth=2, label="SoC fraction")
    ax1a.set_ylabel("SoC (fraction)", color="#2196F3")
    ax1a.set_xlabel("Time (min)")
    ax1a.set_title("S1: Single Unit - SoC & Power")
    ax1a.tick_params(axis="y", labelcolor="#2196F3")

    ax1b = ax1a.twinx()
    ax1b.bar(times_s1, power_s1, width=0.7,
             color=["#4CAF50" if p > 0 else "#F44336" for p in power_s1],
             alpha=0.6, label="Power (kW)")
    ax1b.set_ylabel("Power (kW)", color="#555")
    ax1b.axhline(0, color="black", linewidth=0.5)

    # Speed ratio subplot
    ax1c = fig.add_subplot(gs[0, 1])
    speed_s1 = [s.speed_ratio for s in unit_s1.history]
    loss_s1  = [s.standby_mechanical_kw + s.standby_auxiliary_kw for s in unit_s1.history]
    ax1c.plot(times_s1, speed_s1, color="#9C27B0", linewidth=2, label="Speed ratio")
    ax1c.set_ylabel("Speed ratio (ω/ωmax)", color="#9C27B0")
    ax1c.set_xlabel("Time (min)")
    ax1c.set_title("S1: Speed Ratio vs Standby Loss")
    ax1c.tick_params(axis="y", labelcolor="#9C27B0")

    ax1d = ax1c.twinx()
    ax1d.plot(times_s1, loss_s1, color="#FF9800", linewidth=2, linestyle="--", label="Standby loss (kW)")
    ax1d.set_ylabel("Standby loss (kW)", color="#FF9800")
    ax1d.tick_params(axis="y", labelcolor="#FF9800")

    # ---- Plot 2: aFRR plant dispatch ----
    ax2a = fig.add_subplot(gs[1, 0])
    t_s2 = df_s2["time_h"] * 60
    ax2a.plot(t_s2, df_s2["plant_power_kw"] / 1000, color="#2196F3", linewidth=1.5, label="Plant power (MW)")
    agc_times = np.arange(len(agc_signal)) / 60
    ax2a.plot(agc_times * 60, agc_signal / 1000,
              color="gray", linewidth=1, alpha=0.6, linestyle="--", label="AGC setpoint")
    ax2a.set_ylabel("Power (MW)")
    ax2a.set_xlabel("Time (min)")
    ax2a.set_title("S2: aFRR Plant - Power Following")
    ax2a.legend(fontsize=8)
    ax2a.axhline(0, color="black", linewidth=0.5)
    ax2a.grid(True, alpha=0.3)

    ax2b = fig.add_subplot(gs[1, 1])
    ax2b.plot(t_s2, df_s2["avg_soc_frac"], color="#2196F3", linewidth=2, label="Avg SoC")
    ax2b.fill_between(t_s2, df_s2["min_soc_frac"], df_s2["max_soc_frac"],
                      alpha=0.2, color="#2196F3", label="Min-Max range")
    ax2b.set_ylabel("SoC fraction")
    ax2b.set_xlabel("Time (min)")
    ax2b.set_title("S2: aFRR Plant - Fleet SoC Spread")
    ax2b.legend(fontsize=8)
    ax2b.set_ylim(0, 1)
    ax2b.grid(True, alpha=0.3)

    # ---- Plot 3: Arbitrage plant ----
    t_s3   = df_s3["time_h"]
    hours3 = np.arange(len(price_signal)) * 0.25

    ax3a = fig.add_subplot(gs[2, 0])
    ax3a.fill_between(hours3, price_signal, alpha=0.3, color="#FF9800")
    ax3a.plot(hours3, price_signal, color="#FF9800", linewidth=1.5, label="Spot price")
    ax3a.set_ylabel("Price ($/MWh)", color="#FF9800")
    ax3a.set_xlabel("Time (h)")
    ax3a.set_title("S3: Arbitrage - Price & Plant Power")
    ax3a.tick_params(axis="y", labelcolor="#FF9800")

    ax3b = ax3a.twinx()
    ax3b.bar(t_s3, df_s3["plant_power_kw"] / 1000, width=0.2,
             color=["#4CAF50" if p > 0 else "#F44336" for p in df_s3["plant_power_kw"]],
             alpha=0.7, label="Plant power (MW)")
    ax3b.set_ylabel("Power (MW)")
    ax3b.axhline(0, color="black", linewidth=0.5)

    ax3c = fig.add_subplot(gs[2, 1])
    ax3c.plot(t_s3, df_s3["avg_soc_frac"], color="#2196F3", linewidth=2)
    ax3c.fill_between(t_s3, df_s3["min_soc_frac"], df_s3["max_soc_frac"],
                      alpha=0.2, color="#2196F3")
    ax3c.set_ylabel("SoC fraction")
    ax3c.set_xlabel("Time (h)")
    ax3c.set_title("S3: Arbitrage - Fleet SoC")
    ax3c.set_ylim(0, 1)
    ax3c.grid(True, alpha=0.3)

    plt.suptitle("FESS Unit & Plant Simulation - Three Scenarios", fontsize=14, fontweight="bold")
    plt.savefig("fess_simulation_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved: fess_simulation_results.png")
    plt.show()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    unit_s1                          = scenario_single_unit()
    plant_s2, df_s2, agc_signal      = scenario_afrr_plant()
    plant_s3, df_s3, price_signal    = scenario_arbitrage_plant()

    plot_results(unit_s1, plant_s2, df_s2, agc_signal, plant_s3, df_s3, price_signal)

    print("\n\nQuick reference: key classes")
    print("  FESSParams       - all unit design parameters")
    print("  FESSUnit         - single flywheel physics model")
    print("  FESSSnapshot     - per-timestep unit telemetry")
    print("  GridInterfaceParams - plant transformer + grid codes")
    print("  RevenueServiceConfig - market service configuration")
    print("  FESSPlant        - fleet aggregation + dispatch + revenue")
    print("  PlantSnapshot    - per-timestep plant telemetry")
    print("  DispatchStrategy - EQUAL_SHARE | PRIORITY | SOC_BALANCED | DROOP")
    print("  MarketService    - FCR | AFRR | MFRR | ARBITRAGE | SYNTHETIC_INERTIA")
