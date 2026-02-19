"""
fess_plant.py
=============
Plant-level aggregation of multiple FESS units with:
  - Fleet dispatch strategies (equal share, priority, SoC-balanced)
  - Grid interface modelling (transformer, auxiliary load, grid code limits)
  - Revenue stacking logic (FCR, aFRR, mFRR, energy arbitrage)
  - Plant-level telemetry and performance reporting
  - Pandas DataFrame export for analysis

Requires:  fess_unit.py in the same package / directory.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from fess_unit import FESSUnit, FESSParams, FESSSnapshot, FESSState


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class DispatchStrategy(Enum):
    EQUAL_SHARE     = "equal_share"      # Split power equally across all available units
    PRIORITY        = "priority"         # Dispatch units in order until setpoint met
    SOC_BALANCED    = "soc_balanced"     # Assign highest power to units with most/least SoC
    DROOP           = "droop"            # Proportional to available power (frequency droop)


class MarketService(Enum):
    NONE       = "none"
    FCR        = "fcr"          # Primary frequency containment reserve
    AFRR       = "afrr"        # Automatic secondary frequency regulation
    MFRR       = "mfrr"        # Manual tertiary frequency reserve
    ARBITRAGE  = "arbitrage"   # Energy price arbitrage
    SYNTHETIC_INERTIA = "synthetic_inertia"


@dataclass
class GridInterfaceParams:
    """
    Parameters for the plant-level grid connection.
    Covers transformer, central inverter, auxiliary loads, reactive power, and grid codes.

    Topology note
    -------------
    Two common configurations are supported via central_inverter_efficiency:

    Topology A — Distributed AC (no central inverter):
        Each unit has its own AC-side inverter (already modelled in InverterEfficiency).
        Units connect to a common AC bus; only transformer losses apply at plant level.
        Set central_inverter_efficiency = 1.0  (default disabled).

    Topology B — Centralised DC bus + shared PCS:
        All units connect to a shared DC bus; one central PCS connects to grid.
        Per-unit InverterEfficiency should be zeroed (set k_switch=k_cond=k_fixed=0).
        Set central_inverter_efficiency to the shared PCS efficiency (typ. 0.982–0.992).

    Topology C — Hybrid DC/DC + central PCS:
        Each unit has a DC/DC stage (manages speed-varying rectifier voltage).
        Central inverter handles AC grid interface.
        Both per-unit and central inverter losses apply.
    """
    transformer_capacity_kva:    float = 10_000.0  # Plant transformer rating (kVA)
    transformer_efficiency:      float = 0.995      # ~0.5% transformer losses
    central_inverter_efficiency: float = 1.0        # Set < 1.0 for Topology B/C (see above)
    auxiliary_load_kw:           float = 50.0       # HVAC, controls, site services (kW)
    power_factor:                float = 0.95       # Target PF at grid connection
    max_ramp_rate_kw_per_min:    float = 5_000.0    # Grid code ramp limit (plant level)
    frequency_deadband_hz:       float = 0.02       # FCR activation threshold (Hz)
    nominal_frequency_hz:        float = 50.0       # 50 Hz (EU) or 60 Hz (US/NA)
    grid_voltage_kv:             float = 33.0       # Grid connection voltage (kV)


@dataclass
class RevenueServiceConfig:
    """
    Configuration for a single market service within the revenue stack.
    Used to parameterise the plant dispatch logic.
    """
    service:               MarketService
    enabled:               bool  = True
    capacity_mw:           float = 0.0    # Contracted MW for this service
    price_per_mw_h:        float = 0.0    # $/MW/h availability payment
    charge_price_per_mwh:  float = 0.0    # $/MWh cost of charging energy (arbitrage)
    priority:              int   = 1      # 1 = highest priority in stack
    soc_target_frac:       float = 0.5    # Target SoC for this service window
    soc_min_frac:          float = 0.2    # Hard minimum SoC before service degraded
    soc_max_frac:          float = 0.9    # Hard maximum SoC before service degraded
    # FCR-specific: restoration rate after activation to return to soc_target_frac
    fcr_restoration_rate_kw: float = 0.0  # kW; 0 = no active restoration


@dataclass
class PlantSnapshot:
    """Aggregated plant-level telemetry for a single timestep."""
    time_h:                    float
    active_service:            MarketService
    plant_power_kw:            float   # +ve charge, -ve discharge (at grid meter)
    plant_power_available_kw:  float
    total_soc_kwh:             float
    avg_soc_frac:              float
    min_soc_frac:              float
    max_soc_frac:              float
    units_available:           int
    units_charging:            int
    units_discharging:         int
    units_idle:                int
    total_standby_loss_kw:     float
    total_machine_loss_kw:     float
    total_inverter_loss_kw:    float
    central_inverter_loss_kw:  float   # plant-level PCS loss (0 if Topology A)
    avg_eta_actual:            float   # fleet-average realised one-way efficiency
    total_equivalent_cycles:   float   # sum of equivalent_full_cycles across fleet
    grid_energy_consumed_kwh:  float
    grid_energy_delivered_kwh: float
    auxiliary_load_kwh:        float
    frequency_hz:              Optional[float]
    incremental_revenue:       float   # gross revenue this step
    incremental_charge_cost:   float   # energy cost this step (negative = cost)
    net_revenue:               float   # revenue minus charge cost
    cumulative_revenue:        float   # cumulative gross revenue
    cumulative_charge_cost:    float   # cumulative charge energy cost
    cumulative_net_revenue:    float   # cumulative net revenue


# ---------------------------------------------------------------------------
# FESS Plant
# ---------------------------------------------------------------------------

class FESSPlant:
    """
    Aggregates N FESSUnit objects into a dispatchable storage plant.

    Key responsibilities
    --------------------
    1. Fleet dispatch: translate a single plant-level power setpoint into
       per-unit setpoints using the chosen DispatchStrategy.
    2. Grid interface: apply transformer losses, auxiliary load, and ramp
       limits at the plant boundary.
    3. Revenue stacking: select the active market service each timestep and
       calculate incremental revenue.
    4. Reporting: accumulate PlantSnapshot history for DataFrame export.

    Parameters
    ----------
    units : list[FESSUnit]
        Pre-constructed unit objects.  Allows heterogeneous fleets.
    grid_params : GridInterfaceParams
    dispatch_strategy : DispatchStrategy
    plant_id : str

    Quick start
    -----------
    >>> params = FESSParams(rated_power_kw=250, rated_energy_kwh=16.67)
    >>> units  = [FESSUnit(params, f"FW-{i:02d}") for i in range(20)]
    >>> grid   = GridInterfaceParams(transformer_capacity_kva=10_000)
    >>> plant  = FESSPlant(units, grid, DispatchStrategy.SOC_BALANCED, "TERMINUS-1")
    >>> snap   = plant.step(power_setpoint_kw=-3000, dt_hours=1/60)
    """

    def __init__(
        self,
        units:             list[FESSUnit],
        grid_params:       GridInterfaceParams      = None,
        dispatch_strategy: DispatchStrategy         = DispatchStrategy.SOC_BALANCED,
        plant_id:          str                      = "FESS-PLANT-01",
    ):
        if not units:
            raise ValueError("Plant must have at least one FESS unit.")

        self.units             = units
        self.grid_params       = grid_params or GridInterfaceParams()
        self.dispatch_strategy = dispatch_strategy
        self.plant_id          = plant_id

        # Revenue services (set via configure_revenue_stack)
        self._services: list[RevenueServiceConfig] = []

        # Running totals
        self._cumulative_revenue: float = 0.0
        self._cumulative_charge_cost: float = 0.0
        self._cumulative_grid_consumed_kwh: float  = 0.0
        self._cumulative_grid_delivered_kwh: float = 0.0
        self._cumulative_auxiliary_kwh: float = 0.0
        self._sim_time_h: float = 0.0
        self._last_plant_power_kw: float = 0.0  # For plant-level ramp tracking

        # History
        self.history: list[PlantSnapshot] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_revenue_stack(self, services: list[RevenueServiceConfig]) -> None:
        """
        Define the ordered revenue stack.
        Services are sorted ascending by priority (1 = highest).

        Example
        -------
        >>> plant.configure_revenue_stack([
        ...     RevenueServiceConfig(MarketService.AFRR,      price_per_mw_h=18,  priority=1, capacity_mw=3),
        ...     RevenueServiceConfig(MarketService.FCR,       price_per_mw_h=25,  priority=2, capacity_mw=2),
        ...     RevenueServiceConfig(MarketService.ARBITRAGE, price_per_mw_h=0,   priority=3),
        ... ])
        """
        self._services = sorted(
            [s for s in services if s.enabled],
            key=lambda s: s.priority,
        )

    # ------------------------------------------------------------------
    # Fleet aggregation properties
    # ------------------------------------------------------------------

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def available_units(self) -> list[FESSUnit]:
        return [u for u in self.units if u.state != FESSState.FAULT]

    @property
    def total_rated_power_kw(self) -> float:
        return sum(u.params.rated_power_kw for u in self.available_units)

    @property
    def total_power_available_kw(self) -> float:
        """Aggregate discharge power available right now (speed-limited)."""
        return sum(u.power_available_kw for u in self.available_units)

    @property
    def total_soc_kwh(self) -> float:
        return sum(u.soc_kwh for u in self.units)

    @property
    def avg_soc_frac(self) -> float:
        avail = self.available_units
        if not avail:
            return 0.0
        return float(np.mean([u.soc_frac for u in avail]))

    @property
    def total_rated_energy_kwh(self) -> float:
        return sum(u.params.rated_energy_kwh for u in self.available_units)

    # ------------------------------------------------------------------
    # Grid interface helpers
    # ------------------------------------------------------------------

    def _apply_grid_interface(self, raw_power_kw: float, dt_hours: float) -> tuple[float, float]:
        """
        Apply plant-level ramp limit, transformer losses, and central inverter losses.

        Returns
        -------
        (net_grid_power_kw, central_inverter_loss_kw)
            net_grid_power_kw    : power at the grid metering point (after all losses)
            central_inverter_loss_kw : loss in the shared PCS (0 if eta=1.0)
        """
        # 1. Plant ramp rate limit
        dt_min = dt_hours * 60.0
        max_delta = self.grid_params.max_ramp_rate_kw_per_min * dt_min
        delta = raw_power_kw - self._last_plant_power_kw
        delta_clamped = float(np.clip(delta, -max_delta, max_delta))
        ramped_power = self._last_plant_power_kw + delta_clamped

        # 2. Transformer capacity limit
        max_kw = self.grid_params.transformer_capacity_kva * self.grid_params.power_factor
        ramped_power = float(np.clip(ramped_power, -max_kw, max_kw))

        # 3. Central inverter loss (Topology B/C — shared PCS between DC bus and transformer)
        eta_ci = self.grid_params.central_inverter_efficiency
        ci_loss_kw = 0.0
        if abs(ramped_power) > 0 and eta_ci < 1.0:
            after_ci = ramped_power * eta_ci
            ci_loss_kw = abs(ramped_power) - abs(after_ci)
            ramped_power = after_ci

        # 4. Transformer losses
        if abs(ramped_power) > 0:
            grid_power = ramped_power * self.grid_params.transformer_efficiency
        else:
            grid_power = 0.0

        self._last_plant_power_kw = ramped_power
        return grid_power, ci_loss_kw

    # ------------------------------------------------------------------
    # Dispatch strategies
    # ------------------------------------------------------------------

    def _dispatch_equal_share(
        self,
        setpoint_kw: float,
        avail: list[FESSUnit],
    ) -> dict[str, float]:
        """Split setpoint equally across available units."""
        if not avail:
            return {}
        per_unit = setpoint_kw / len(avail)
        return {u.unit_id: per_unit for u in avail}

    def _dispatch_priority(
        self,
        setpoint_kw: float,
        avail: list[FESSUnit],
    ) -> dict[str, float]:
        """
        Dispatch units sequentially (by index) until setpoint is met.
        First units take maximum load; remaining units idle.
        """
        assignments: dict[str, float] = {u.unit_id: 0.0 for u in avail}
        remaining = setpoint_kw

        for u in avail:
            if abs(remaining) < 1e-3:
                break
            if remaining > 0:  # Charging
                alloc = min(remaining, u.params.rated_power_kw - u.power_available_kw)
            else:              # Discharging
                alloc = max(remaining, -u.power_available_kw)

            assignments[u.unit_id] = alloc
            remaining -= alloc

        return assignments

    def _dispatch_soc_balanced(
        self,
        setpoint_kw: float,
        avail: list[FESSUnit],
    ) -> dict[str, float]:
        """
        Weight dispatch proportionally to SoC (discharge) or inverse SoC (charge),
        so that units remain balanced in state of charge.

        For discharge: units with higher SoC carry more load.
        For charge:    units with lower SoC receive more power.
        """
        if not avail:
            return {}

        assignments: dict[str, float] = {}

        if setpoint_kw < 0:
            # Discharging: weight by available power (∝ speed_ratio ∝ √SoC)
            weights = np.array([u.power_available_kw for u in avail], dtype=float)
        else:
            # Charging: weight by inverse SoC (fill lowest first)
            inv_soc = np.array([1.0 - u.soc_frac for u in avail], dtype=float)
            weights = inv_soc

        total_weight = weights.sum()
        if total_weight < 1e-9:
            # Fallback to equal share
            return self._dispatch_equal_share(setpoint_kw, avail)

        for i, u in enumerate(avail):
            share = (weights[i] / total_weight) * setpoint_kw
            # Clip each unit to its physical limits
            if share < 0:
                share = max(share, -u.power_available_kw)
            else:
                headroom = u.params.rated_power_kw - u.power_available_kw
                share = min(share, headroom)
            assignments[u.unit_id] = share

        return assignments

    def _dispatch_droop(
        self,
        setpoint_kw: float,
        avail: list[FESSUnit],
        frequency_hz: float,
    ) -> dict[str, float]:
        """
        Droop control: power proportional to frequency deviation.
        Used for FCR / primary response.

        Each unit responds with:
            P = P_rated * (Δf / f_deadband) * droop_gain
        capped at its current available power.
        """
        f_nom      = self.grid_params.nominal_frequency_hz
        f_deadband = self.grid_params.frequency_deadband_hz
        delta_f    = frequency_hz - f_nom

        if abs(delta_f) <= f_deadband:
            return {u.unit_id: 0.0 for u in avail}

        # Normalised response signal (-1 to +1)
        # Under-frequency → discharge → negative convention
        response_signal = -delta_f / (f_nom * 0.01)  # % droop
        response_signal = float(np.clip(response_signal, -1.0, 1.0))

        assignments: dict[str, float] = {}
        for u in avail:
            power = response_signal * u.power_available_kw
            assignments[u.unit_id] = power

        return assignments

    def _resolve_dispatch(
        self,
        setpoint_kw: float,
        avail: list[FESSUnit],
        frequency_hz: Optional[float] = None,
    ) -> dict[str, float]:
        """Route to the correct dispatch strategy."""
        if self.dispatch_strategy == DispatchStrategy.EQUAL_SHARE:
            return self._dispatch_equal_share(setpoint_kw, avail)
        elif self.dispatch_strategy == DispatchStrategy.PRIORITY:
            return self._dispatch_priority(setpoint_kw, avail)
        elif self.dispatch_strategy == DispatchStrategy.SOC_BALANCED:
            return self._dispatch_soc_balanced(setpoint_kw, avail)
        elif self.dispatch_strategy == DispatchStrategy.DROOP:
            if frequency_hz is None:
                warnings.warn("DROOP strategy requires frequency_hz; falling back to SOC_BALANCED.")
                return self._dispatch_soc_balanced(setpoint_kw, avail)
            return self._dispatch_droop(setpoint_kw, avail, frequency_hz)
        else:
            return self._dispatch_equal_share(setpoint_kw, avail)

    # ------------------------------------------------------------------
    # FCR SoC restoration helper
    # ------------------------------------------------------------------

    def _fcr_restoration_setpoint(self) -> float:
        """
        Compute a gentle restoration setpoint (kW) to return fleet SoC
        toward the FCR target after an activation event.

        Uses the configured fcr_restoration_rate_kw if set, otherwise zero.
        Direction: positive (charge) if below target, negative (discharge) if above.
        Returns 0.0 if within a 2% band of target to avoid hunting.
        """
        fcr_cfgs = [s for s in self._services if s.service == MarketService.FCR and s.enabled]
        if not fcr_cfgs:
            return 0.0
        cfg = fcr_cfgs[0]
        if cfg.fcr_restoration_rate_kw <= 0.0:
            return 0.0

        soc = self.avg_soc_frac
        target = cfg.soc_target_frac
        band = 0.02  # ±2% dead-band before restoration activates

        if soc < target - band:
            return cfg.fcr_restoration_rate_kw          # charge to restore
        elif soc > target + band:
            return -cfg.fcr_restoration_rate_kw         # discharge to restore
        return 0.0

    # ------------------------------------------------------------------
    # Revenue calculation
    # ------------------------------------------------------------------

    def _calculate_revenue(
        self,
        service:            MarketService,
        plant_power_kw:     float,
        grid_consumed_kwh:  float,
        grid_delivered_kwh: float,
        spot_price:         float,
        dt_hours:           float,
    ) -> tuple[float, float]:
        """
        Calculate incremental revenue and charge cost for this timestep.

        Returns
        -------
        (gross_revenue, charge_cost)
            gross_revenue : positive value — income from grid delivery or capacity payment
            charge_cost   : positive value — energy cost of charging (caller subtracts)

        Revenue conventions
        -------------------
        FCR / aFRR / mFRR  : availability payment $/MW/h × contracted_MW × dt
        ARBITRAGE          : delivery revenue = spot_price × MWh_delivered
                             charge cost      = charge_price × MWh_consumed
        """
        matched = [s for s in self._services if s.service == service]
        svc = matched[0] if matched else None

        gross_revenue = 0.0
        charge_cost   = 0.0

        if service in (MarketService.FCR, MarketService.AFRR, MarketService.MFRR):
            if svc:
                gross_revenue = svc.price_per_mw_h * (svc.capacity_mw * dt_hours)

        elif service == MarketService.ARBITRAGE:
            # Delivery revenue (discharging at spot price)
            gross_revenue = spot_price * (grid_delivered_kwh / 1000.0)
            # Charging cost — use service-level charge price if set, else spot price
            if svc and svc.charge_price_per_mwh > 0:
                charge_cost = svc.charge_price_per_mwh * (grid_consumed_kwh / 1000.0)
            else:
                # Assume spot_price also applies to charging (caller should pass
                # the actual charging price; this is a safe conservative fallback)
                charge_cost = spot_price * (grid_consumed_kwh / 1000.0)

        return gross_revenue, charge_cost

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(
        self,
        power_setpoint_kw: float,
        dt_hours: float,
        active_service:   MarketService   = MarketService.NONE,
        spot_price_per_mwh: float         = 0.0,
        charge_price_per_mwh: float       = 0.0,
        frequency_hz:     Optional[float] = None,
        apply_fcr_restoration: bool       = True,
        log:              bool            = True,
    ) -> PlantSnapshot:
        """
        Advance the entire plant by one timestep.

        Parameters
        ----------
        power_setpoint_kw : float
            Plant-level power setpoint at the flywheel DC bus.
            +ve = charging, -ve = discharging.
        dt_hours : float
            Timestep size in hours.
        active_service : MarketService
            Which market service is being provided this timestep.
        spot_price_per_mwh : float
            Real-time energy price for discharge revenue (arbitrage).
        charge_price_per_mwh : float
            Energy purchase price for charging cost calculation.
            If 0.0 and service is ARBITRAGE, falls back to spot_price_per_mwh.
        frequency_hz : float, optional
            Measured grid frequency.  Required for DROOP dispatch.
        apply_fcr_restoration : bool
            When True and active_service is FCR, superimpose a slow restoration
            setpoint to return fleet SoC toward fcr_restoration_rate_kw target.
        log : bool
            Whether to append to self.history.

        Returns
        -------
        PlantSnapshot
        """
        self._sim_time_h += dt_hours
        avail = self.available_units

        # --- FCR restoration overlay ---
        if apply_fcr_restoration and active_service == MarketService.FCR:
            restoration_kw = self._fcr_restoration_setpoint()
            # Add restoration to setpoint only if it doesn't oppose the primary signal
            if abs(power_setpoint_kw) < 1e-3:
                power_setpoint_kw = restoration_kw
            elif (power_setpoint_kw * restoration_kw) > 0:
                # Same direction: add
                power_setpoint_kw += restoration_kw

        # --- 1. Apply grid interface limits ---
        grid_setpoint_kw, ci_loss_kw = self._apply_grid_interface(power_setpoint_kw, dt_hours)

        # --- 2. Resolve per-unit setpoints ---
        unit_setpoints = self._resolve_dispatch(grid_setpoint_kw, avail, frequency_hz)

        # --- 3. Step each unit ---
        unit_snaps: list[FESSSnapshot] = []
        for u in self.units:
            sp = unit_setpoints.get(u.unit_id, 0.0)
            snap = u.step(sp, dt_hours, log=log)
            unit_snaps.append(snap)

        # --- 4. Aggregate plant telemetry ---
        plant_power_kw          = sum(s.power_kw for s in unit_snaps)
        grid_consumed_kwh       = sum(s.grid_energy_consumed_kwh for s in unit_snaps)
        grid_delivered_kwh      = sum(s.grid_energy_delivered_kwh for s in unit_snaps)
        total_standby_loss_kw   = sum(
            s.standby_mechanical_kw + s.standby_auxiliary_kw for s in unit_snaps
        )
        total_machine_loss_kw   = sum(s.machine_loss_kw for s in unit_snaps)
        total_inverter_loss_kw  = sum(s.inverter_loss_kw for s in unit_snaps)
        total_soc_kwh           = sum(u.soc_kwh for u in self.units)
        power_available_kw      = sum(u.power_available_kw for u in avail)
        total_equiv_cycles      = sum(u.equivalent_full_cycles for u in self.units)

        # Fleet-average realised efficiency (active units only)
        active_etas = [s.eta_actual for s in unit_snaps if s.eta_actual > 0]
        avg_eta     = float(np.mean(active_etas)) if active_etas else 0.0

        # Auxiliary load (always consumed, drawn from grid)
        auxiliary_kwh = self.grid_params.auxiliary_load_kw * dt_hours

        # Unit status counts
        status_counts = {s: 0 for s in FESSState}
        for u in self.units:
            status_counts[u.state] += 1

        # SoC stats
        soc_fracs = [u.soc_frac for u in avail] if avail else [0.0]

        # --- 5. Calculate revenue and charge cost ---
        # Override charge price: use explicit arg > service config > spot price
        effective_charge_price = charge_price_per_mwh if charge_price_per_mwh > 0 else spot_price_per_mwh
        gross_revenue, charge_cost = self._calculate_revenue(
            active_service,
            plant_power_kw,
            grid_consumed_kwh,
            grid_delivered_kwh,
            spot_price_per_mwh,
            dt_hours,
        )
        # Override charge cost with explicit price if passed
        if active_service == MarketService.ARBITRAGE and effective_charge_price > 0:
            charge_cost = effective_charge_price * (grid_consumed_kwh / 1000.0)

        net_rev = gross_revenue - charge_cost
        self._cumulative_revenue      += gross_revenue
        self._cumulative_charge_cost  += charge_cost
        self._cumulative_grid_consumed_kwh  += grid_consumed_kwh
        self._cumulative_grid_delivered_kwh += grid_delivered_kwh
        self._cumulative_auxiliary_kwh      += auxiliary_kwh

        # --- 6. Build snapshot ---
        snap = PlantSnapshot(
            time_h                    = self._sim_time_h,
            active_service            = active_service,
            plant_power_kw            = plant_power_kw,
            plant_power_available_kw  = power_available_kw,
            total_soc_kwh             = total_soc_kwh,
            avg_soc_frac              = float(np.mean(soc_fracs)),
            min_soc_frac              = float(np.min(soc_fracs)),
            max_soc_frac              = float(np.max(soc_fracs)),
            units_available           = len(avail),
            units_charging            = status_counts[FESSState.CHARGING],
            units_discharging         = status_counts[FESSState.DISCHARGING],
            units_idle                = status_counts[FESSState.IDLE],
            total_standby_loss_kw     = total_standby_loss_kw,
            total_machine_loss_kw     = total_machine_loss_kw,
            total_inverter_loss_kw    = total_inverter_loss_kw,
            central_inverter_loss_kw  = ci_loss_kw,
            avg_eta_actual            = avg_eta,
            total_equivalent_cycles   = total_equiv_cycles,
            grid_energy_consumed_kwh  = grid_consumed_kwh,
            grid_energy_delivered_kwh = grid_delivered_kwh,
            auxiliary_load_kwh        = auxiliary_kwh,
            frequency_hz              = frequency_hz,
            incremental_revenue       = gross_revenue,
            incremental_charge_cost   = charge_cost,
            net_revenue               = net_rev,
            cumulative_revenue        = self._cumulative_revenue,
            cumulative_charge_cost    = self._cumulative_charge_cost,
            cumulative_net_revenue    = self._cumulative_revenue - self._cumulative_charge_cost,
        )

        if log:
            self.history.append(snap)

        return snap

    # ------------------------------------------------------------------
    # Reporting and export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Export plant history to a tidy Pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()

        rows = []
        for s in self.history:
            rows.append({
                "time_h":                    s.time_h,
                "active_service":            s.active_service.value,
                "plant_power_kw":            s.plant_power_kw,
                "plant_power_available_kw":  s.plant_power_available_kw,
                "total_soc_kwh":             s.total_soc_kwh,
                "avg_soc_frac":              s.avg_soc_frac,
                "min_soc_frac":              s.min_soc_frac,
                "max_soc_frac":              s.max_soc_frac,
                "units_available":           s.units_available,
                "units_charging":            s.units_charging,
                "units_discharging":         s.units_discharging,
                "units_idle":                s.units_idle,
                "total_standby_loss_kw":     s.total_standby_loss_kw,
                "total_machine_loss_kw":     s.total_machine_loss_kw,
                "total_inverter_loss_kw":    s.total_inverter_loss_kw,
                "central_inverter_loss_kw":  s.central_inverter_loss_kw,
                "avg_eta_actual":            s.avg_eta_actual,
                "total_equivalent_cycles":   s.total_equivalent_cycles,
                "grid_energy_consumed_kwh":  s.grid_energy_consumed_kwh,
                "grid_energy_delivered_kwh": s.grid_energy_delivered_kwh,
                "auxiliary_load_kwh":        s.auxiliary_load_kwh,
                "frequency_hz":              s.frequency_hz,
                "incremental_revenue":       s.incremental_revenue,
                "incremental_charge_cost":   s.incremental_charge_cost,
                "net_revenue":               s.net_revenue,
                "cumulative_revenue":        s.cumulative_revenue,
                "cumulative_charge_cost":    s.cumulative_charge_cost,
                "cumulative_net_revenue":    s.cumulative_net_revenue,
            })

        df = pd.DataFrame(rows)
        df["time_h"] = df["time_h"].astype(float)
        return df

    def unit_dataframe(self, unit_id: str) -> pd.DataFrame:
        """Export a single unit's history to a DataFrame."""
        unit = next((u for u in self.units if u.unit_id == unit_id), None)
        if unit is None:
            raise ValueError(f"Unit {unit_id!r} not found in plant.")

        rows = []
        for s in unit.history:
            rows.append({
                "time_h":                   s.time_h,
                "state":                    s.state.value,
                "soc_kwh":                  s.soc_kwh,
                "soc_frac":                 s.soc_frac,
                "speed_ratio":              s.speed_ratio,
                "power_kw":                 s.power_kw,
                "power_available_kw":       s.power_available_kw,
                "standby_loss_kw":          s.standby_loss_kw,
                "energy_charged_kwh":       s.energy_charged_kwh,
                "energy_discharged_kwh":    s.energy_discharged_kwh,
                "grid_energy_consumed_kwh": s.grid_energy_consumed_kwh,
                "grid_energy_delivered_kwh":s.grid_energy_delivered_kwh,
                "equivalent_full_cycles":   s.equivalent_full_cycles,
            })

        return pd.DataFrame(rows)

    def plant_summary(self) -> dict:
        """High-level plant performance summary."""
        total_throughput = (
            self._cumulative_grid_consumed_kwh
            + self._cumulative_grid_delivered_kwh
        )
        rt_eff = (
            self._cumulative_grid_delivered_kwh / self._cumulative_grid_consumed_kwh
            if self._cumulative_grid_consumed_kwh > 0
            else None
        )
        return {
            "plant_id":                          self.plant_id,
            "n_units":                           self.n_units,
            "dispatch_strategy":                 self.dispatch_strategy.value,
            "total_rated_power_kw":              self.total_rated_power_kw,
            "total_rated_energy_kwh":            self.total_rated_energy_kwh,
            "avg_soc_frac":                      round(self.avg_soc_frac, 4),
            "total_soc_kwh":                     round(self.total_soc_kwh, 3),
            "power_available_kw":                round(self.total_power_available_kw, 2),
            "cumulative_grid_consumed_kwh":       round(self._cumulative_grid_consumed_kwh, 2),
            "cumulative_grid_delivered_kwh":      round(self._cumulative_grid_delivered_kwh, 2),
            "cumulative_auxiliary_kwh":           round(self._cumulative_auxiliary_kwh, 2),
            "total_equivalent_cycles":            round(
                sum(u.equivalent_full_cycles for u in self.units), 3
            ),
            "realised_roundtrip_efficiency":      round(rt_eff, 4) if rt_eff else None,
            "cumulative_gross_revenue":           round(self._cumulative_revenue, 2),
            "cumulative_charge_cost":             round(self._cumulative_charge_cost, 2),
            "cumulative_net_revenue":             round(
                self._cumulative_revenue - self._cumulative_charge_cost, 2
            ),
            "units_available":                    len(self.available_units),
        }

    def reset(self, initial_soc_frac: float = 0.5) -> None:
        """Reset the entire plant (all units + plant counters)."""
        for u in self.units:
            u.reset(initial_soc_frac)
        self._cumulative_revenue = 0.0
        self._cumulative_charge_cost = 0.0
        self._cumulative_grid_consumed_kwh  = 0.0
        self._cumulative_grid_delivered_kwh = 0.0
        self._cumulative_auxiliary_kwh = 0.0
        self._sim_time_h = 0.0
        self._last_plant_power_kw = 0.0
        self.history.clear()

    # ------------------------------------------------------------------
    # Discharge sequencing
    # ------------------------------------------------------------------

    def discharge_sequence(
        self,
        total_energy_kwh:    float,
        mode:                str   = "all_at_once",
        batch_size:          int   = 1,
        power_per_unit_kw:   float = None,
        optimize_efficiency: bool  = True,
    ) -> list[dict]:
        """
        Plan a discharge of *total_energy_kwh* from the current fleet state
        without modifying any unit's actual SoC.  Returns a list of dispatch
        stage descriptions for analysis or scheduling.

        This method is read-only — it estimates expected energy delivery,
        efficiency, and duration for each stage using the current SoC of each
        unit but does not call step().

        Parameters
        ----------
        total_energy_kwh : float
            Total grid-side energy to deliver.
        mode : str
            "all_at_once"  — all available units discharge simultaneously.
            "batches"      — batch_size units at a time; next batch activates
                             when the previous batch is depleted.
            "one_by_one"   — one unit at a time.
        batch_size : int
            Number of units per batch (only used when mode="batches").
        power_per_unit_kw : float, optional
            Requested discharge power per active unit.  Defaults to the
            speed-limited power_available_kw of each unit at current SoC.
        optimize_efficiency : bool
            When True, sort units highest-SoC-first before batching.
            Rationale: copper loss ∝ (P/sr)²; high-SoC units have higher sr,
            so the same output power causes less I²R loss.

        Returns
        -------
        list[dict]  — one dict per stage:
            stage          : int — stage index
            unit_ids       : list[str]
            units_active   : int
            power_kw_each  : float — discharge power per unit (negative convention)
            total_power_kw : float
            eta_avg        : float — average discharge efficiency across active units
            energy_kwh_available : float — usable kWh in these units at current SoC
            estimated_duration_h : float — hours to deliver total_energy_kwh from this stage

        Efficiency note
        ---------------
        one_by_one with optimize_efficiency=True is generally best for slow
        arbitrage discharges: each unit operates near its peak-efficiency load
        (≈50–75% rated) before moving to the next.  all_at_once may be
        required for ancillary services where total plant power is the
        primary constraint.
        """
        avail = self.available_units
        if not avail:
            return []

        # Sort by SoC descending if optimising for efficiency
        if optimize_efficiency:
            avail = sorted(avail, key=lambda u: u.soc_frac, reverse=True)

        # Build batches of unit lists
        if mode == "all_at_once":
            batches = [avail]
        elif mode == "one_by_one":
            batches = [[u] for u in avail]
        elif mode == "batches":
            bs = max(1, int(batch_size))
            batches = [avail[i:i+bs] for i in range(0, len(avail), bs)]
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'all_at_once', 'batches', or 'one_by_one'.")

        stages = []
        energy_remaining = total_energy_kwh

        for stage_idx, batch in enumerate(batches):
            if energy_remaining <= 0:
                break

            # Usable energy in this batch (above SoC floor)
            soc_floor_kwh = (
                batch[0].params.soc_min_frac * batch[0].params.rated_energy_kwh
            )
            energy_available = sum(
                max(u.soc_kwh - soc_floor_kwh, 0.0) for u in batch
            )
            if energy_available < 1e-3:
                continue

            # Power per unit
            if power_per_unit_kw is not None:
                p_each = abs(power_per_unit_kw)
            else:
                p_each = float(np.mean([u.power_available_kw for u in batch]))
            p_each = max(p_each, 1.0)  # floor at 1 kW to avoid division by zero

            total_power_kw = p_each * len(batch)

            # Average discharge efficiency across batch using current SoC
            etas = []
            for u in batch:
                p_pu = min(p_each / u.params.rated_power_kw, 1.0)
                sr   = u.speed_ratio
                eta  = u.params.build_efficiency_model().eta_discharge(p_pu, sr)
                etas.append(eta)
            eta_avg = float(np.mean(etas)) if etas else 0.0

            # Energy contribution from this batch (limited by remaining need and availability)
            energy_from_batch = min(energy_remaining, energy_available)
            duration_h = energy_from_batch / total_power_kw if total_power_kw > 0 else 0.0

            stages.append({
                "stage":                 stage_idx + 1,
                "unit_ids":              [u.unit_id for u in batch],
                "units_active":          len(batch),
                "power_kw_each":         -p_each,  # negative = discharge
                "total_power_kw":        -total_power_kw,
                "eta_avg":               round(eta_avg, 4),
                "energy_kwh_available":  round(energy_available, 3),
                "energy_kwh_planned":    round(energy_from_batch, 3),
                "estimated_duration_h":  round(duration_h, 4),
            })

            energy_remaining -= energy_from_batch

        return stages

    # ------------------------------------------------------------------
    # Market simulation helpers
    # ------------------------------------------------------------------

    def simulate_schedule(
        self,
        schedule: pd.DataFrame,
        dt_hours: float = 1.0 / 4,
        reset_before: bool = True,
    ) -> pd.DataFrame:
        """
        Run a full simulation from a pre-built schedule DataFrame.

        Designed for day-ahead and intraday arbitrage back-testing where the
        dispatch optimiser produces a power schedule and this method executes
        it through the physics model timestep-by-timestep.

        Parameters
        ----------
        schedule : pd.DataFrame
            Must contain at minimum the column 'power_setpoint_kw'.
            Optional columns (used if present):
                'spot_price_per_mwh'   — discharge revenue price
                'charge_price_per_mwh' — charging cost (defaults to spot price if absent)
                'active_service'       — MarketService value string (default "arbitrage")
                'frequency_hz'         — for DROOP dispatch
            Index is treated as time labels (not used for physics — only dt_hours matters).

        dt_hours : float
            Timestep duration.  Must match the schedule row spacing.
            Common values: 1/4 (15 min), 1/2 (30 min), 1.0 (hourly).

        reset_before : bool
            If True, call reset() before running.  Set False to continue from
            current plant state (e.g. rolling intraday re-dispatch).

        Returns
        -------
        pd.DataFrame
            Combined schedule + PlantSnapshot columns for each timestep.

        Example
        -------
        >>> import pandas as pd
        >>> prices  = [30, 28, 35, 80, 90, 75, 40, 25]   # $/MWh, 8 × 30-min slots
        >>> setpts  = [-1000, -1000, 0, 500, 500, 0, -500, -500]  # kW
        >>> sched = pd.DataFrame({"power_setpoint_kw": setpts, "spot_price_per_mwh": prices})
        >>> results = plant.simulate_schedule(sched, dt_hours=0.5)
        >>> print(results[["plant_power_kw", "net_revenue", "avg_soc_frac"]])
        """
        if reset_before:
            self.reset()

        results = []
        has_spot  = "spot_price_per_mwh"   in schedule.columns
        has_cprce = "charge_price_per_mwh" in schedule.columns
        has_svc   = "active_service"        in schedule.columns
        has_freq  = "frequency_hz"          in schedule.columns

        for i, row in enumerate(schedule.itertuples(index=True)):
            setpoint  = float(row.power_setpoint_kw)
            spot      = float(getattr(row, "spot_price_per_mwh",   0.0)) if has_spot  else 0.0
            cpr       = float(getattr(row, "charge_price_per_mwh", 0.0)) if has_cprce else 0.0
            freq      = float(getattr(row, "frequency_hz",         50.0)) if has_freq  else None
            svc_str   = str(getattr(row, "active_service", "arbitrage")) if has_svc else "arbitrage"

            try:
                svc = MarketService(svc_str)
            except ValueError:
                svc = MarketService.ARBITRAGE

            snap = self.step(
                power_setpoint_kw    = setpoint,
                dt_hours             = dt_hours,
                active_service       = svc,
                spot_price_per_mwh   = spot,
                charge_price_per_mwh = cpr,
                frequency_hz         = freq,
                log                  = True,
            )

            row_dict = {}
            # Copy schedule columns
            for col in schedule.columns:
                row_dict[col] = getattr(row, col, None)
            # Append snapshot fields
            row_dict.update({
                "sim_time_h":                snap.time_h,
                "plant_power_kw":            snap.plant_power_kw,
                "plant_power_available_kw":  snap.plant_power_available_kw,
                "total_soc_kwh":             snap.total_soc_kwh,
                "avg_soc_frac":              snap.avg_soc_frac,
                "min_soc_frac":              snap.min_soc_frac,
                "max_soc_frac":              snap.max_soc_frac,
                "units_charging":            snap.units_charging,
                "units_discharging":         snap.units_discharging,
                "units_idle":                snap.units_idle,
                "total_standby_loss_kw":     snap.total_standby_loss_kw,
                "total_machine_loss_kw":     snap.total_machine_loss_kw,
                "total_inverter_loss_kw":    snap.total_inverter_loss_kw,
                "central_inverter_loss_kw":  snap.central_inverter_loss_kw,
                "avg_eta_actual":            snap.avg_eta_actual,
                "grid_energy_consumed_kwh":  snap.grid_energy_consumed_kwh,
                "grid_energy_delivered_kwh": snap.grid_energy_delivered_kwh,
                "incremental_revenue":       snap.incremental_revenue,
                "incremental_charge_cost":   snap.incremental_charge_cost,
                "net_revenue":               snap.net_revenue,
                "cumulative_net_revenue":    snap.cumulative_net_revenue,
                "total_equivalent_cycles":   snap.total_equivalent_cycles,
            })
            results.append(row_dict)

        return pd.DataFrame(results)

    def __repr__(self) -> str:
        return (
            f"FESSPlant(id={self.plant_id!r}, "
            f"units={self.n_units}, "
            f"rated={self.total_rated_power_kw:.0f} kW / "
            f"{self.total_rated_energy_kwh:.0f} kWh, "
            f"SoC={self.avg_soc_frac:.1%}, "
            f"strategy={self.dispatch_strategy.value})"
        )
