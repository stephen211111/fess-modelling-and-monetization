"""
fess_unit.py
============
Physics-accurate model of a single Flywheel Energy Storage System (FESS) unit.

Key physics principles modelled
---------------------------------
  - Kinetic energy:          E = 0.5 * I * omega²
  - SoC–speed:               SoC_frac = (omega/omega_max)²
  - Power–speed coupling:    P_available = P_rated * speed_ratio
  - Self-discharge:          Exponential decay, speed-dependent windage/friction
  - Machine efficiency:      Curve model (copper + iron + windage losses)
  - Inverter efficiency:     Curve model (switching + conduction + fixed losses)
  - Cycle tracking:          Equivalent full-cycle counter (no degradation)
  - Ramp rate:               Power electronics limit

Units convention
-----------------
  Energy      : kWh
  Power       : kW
  Speed ratio : dimensionless 0-1  (omega / omega_max)
  Time        : hours (dt_hours)

Dependencies
------------
  efficiency_models.py  (MachineEfficiency, InverterEfficiency, FESSEfficiencyModel)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from efficiency_models import (
    MachineParams, MachineEfficiency,
    InverterParams, InverterEfficiency,
    FESSEfficiencyModel,
)
from standby_losses import StandbyLossParams, StandbyLossModel, StandbyLossBreakdown


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class FESSState(Enum):
    IDLE        = "idle"
    CHARGING    = "charging"
    DISCHARGING = "discharging"
    FAULT       = "fault"


# ---------------------------------------------------------------------------
# Design parameters
# ---------------------------------------------------------------------------

@dataclass
class FESSParams:
    """
    All design-point parameters for a single FESS unit.

    Efficiency is determined by the MachineParams and InverterParams
    component models, not a scalar.  Both can be populated from
    manufacturer data or fitted to measured efficiency curves.
    """
    # Nameplate
    rated_power_kw:   float = 250.0
    rated_energy_kwh: float = 16.67    # ~15 min at rated power

    # Speed window
    min_speed_ratio:  float = 0.20
    max_speed_ratio:  float = 1.00

    # Standby losses — decomposed physical model
    standby_params: StandbyLossParams = None

    # Power electronics ramp limit
    ramp_rate_kw_per_sec: float = 500.0

    # Efficiency component models (defaults used if not supplied)
    machine_params:  MachineParams  = None
    inverter_params: InverterParams = None

    def __post_init__(self):
        assert 0 < self.min_speed_ratio < self.max_speed_ratio <= 1.0
        assert self.rated_power_kw > 0
        assert self.rated_energy_kwh > 0
        if self.standby_params is None:
            self.standby_params = StandbyLossParams()
        if self.machine_params is None:
            self.machine_params = MachineParams()
        if self.inverter_params is None:
            self.inverter_params = InverterParams()

    @property
    def soc_min_frac(self) -> float:
        return self.min_speed_ratio ** 2

    @property
    def soc_max_frac(self) -> float:
        return self.max_speed_ratio ** 2  # = 1.0

    def build_efficiency_model(self) -> FESSEfficiencyModel:
        """Construct the combined machine+inverter efficiency model."""
        machine  = MachineEfficiency(self.machine_params,  self.rated_power_kw)
        inverter = InverterEfficiency(self.inverter_params, self.rated_power_kw)
        return FESSEfficiencyModel(machine, inverter)


# ---------------------------------------------------------------------------
# Per-timestep telemetry
# ---------------------------------------------------------------------------

@dataclass
class FESSSnapshot:
    """Complete telemetry for a single simulation timestep."""
    time_h:                    float
    state:                     FESSState
    soc_kwh:                   float
    soc_frac:                  float        # 0-1 relative to rated energy
    speed_ratio:               float        # omega / omega_max
    power_kw:                  float        # +ve=charging, -ve=discharging
    power_available_kw:        float        # max discharge power at current speed

    # Standby losses — mechanical (drain flywheel SoC)
    standby_aero_w:            float        # aerodynamic drag          ∝ sr³
    standby_tmb_eddy_w:        float        # TMB eddy currents         ∝ sr²
    standby_rmb_eddy_sync_w:   float        # RMB synchronous eddy      ∝ sr²
    standby_mechanical_kw:     float        # total mechanical loss (kW)

    # Standby losses — electrical auxiliary (drawn from grid)
    standby_tmb_bias_w:        float        # TMB bias coil             constant
    standby_rmb_bias_w:        float        # RMB bias coil(s)          ∝ sr²
    standby_rmb_eddy_pwm_w:    float        # RMB PWM-ripple eddy       constant
    standby_cooling_w:         float        # water pump + fan          constant
    standby_vacuum_w:          float        # vacuum pump (avg over dt) intermittent
    standby_auxiliary_kw:      float        # total auxiliary load (kW)
    vacuum_pump_running:       bool

    # Electrical conversion losses
    machine_loss_kw:           float        # electrical machine loss this step
    inverter_loss_kw:          float        # inverter/PCS loss this step
    eta_actual:                float        # realised one-way efficiency this step

    # Energy flows
    energy_charged_kwh:        float        # energy added to flywheel shaft
    energy_discharged_kwh:     float        # energy extracted from flywheel shaft
    grid_energy_consumed_kwh:  float        # kWh drawn from grid (charging)
    grid_energy_delivered_kwh: float        # kWh delivered to grid (discharging)

    cumulative_operating_hours: float
    equivalent_full_cycles:    float


# ---------------------------------------------------------------------------
# FESS Unit
# ---------------------------------------------------------------------------

class FESSUnit:
    """
    Single flywheel energy storage unit with physics-based efficiency curves.

    Efficiency is evaluated at each timestep from the current
    (power_pu, speed_ratio) operating point via the component loss models,
    not a fixed scalar.  This captures the variation in efficiency with
    load level, speed, and direction of power flow.

    Cycle counter accumulates equivalent full cycles from total energy
    throughput.  There is no capacity degradation — cycles are tracked
    for operational monitoring only.

    Parameters
    ----------
    params : FESSParams
    unit_id : str
    initial_soc_frac : float
    """

    def __init__(
        self,
        params:           FESSParams,
        unit_id:          str   = "FW-00",
        initial_soc_frac: float = 0.5,
    ):
        self.params     = params
        self.unit_id    = unit_id
        self._eff_model     = params.build_efficiency_model()
        self._standby_model = StandbyLossModel(
            params.standby_params,
            initial_vacuum_phase_s=params.standby_params.vacuum_cycle_period_s / 2,
        )

        # State
        self._soc_kwh          = params.rated_energy_kwh * float(
            np.clip(initial_soc_frac, params.soc_min_frac, params.soc_max_frac)
        )
        self._state            = FESSState.IDLE
        self._current_power_kw = 0.0

        # Cumulative counters
        self.cumulative_operating_hours       = 0.0
        self.cumulative_energy_charged_kwh    = 0.0
        self.cumulative_energy_discharged_kwh = 0.0
        self.cumulative_machine_loss_kwh      = 0.0
        self.cumulative_inverter_loss_kwh     = 0.0
        self.equivalent_full_cycles           = 0.0

        self._sim_time_h = 0.0
        self.history: list[FESSSnapshot] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def soc_kwh(self) -> float:
        return self._soc_kwh

    @property
    def soc_frac(self) -> float:
        return self._soc_kwh / self.params.rated_energy_kwh

    @property
    def speed_ratio(self) -> float:
        """omega/omega_max derived from E = E_rated * (omega/omega_max)^2"""
        return math.sqrt(max(self.soc_frac, 0.0))

    @property
    def power_available_kw(self) -> float:
        """Max discharge power at current speed: P_rated * speed_ratio"""
        return self.params.rated_power_kw * self.speed_ratio

    @property
    def power_headroom_kw(self) -> float:
        """Max additional charge power available (rated minus current discharge capacity)"""
        return self.params.rated_power_kw - self.power_available_kw

    @property
    def state(self) -> FESSState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_soc(self, soc: float) -> float:
        lo = self.params.soc_min_frac * self.params.rated_energy_kwh
        hi = self.params.soc_max_frac * self.params.rated_energy_kwh
        return float(np.clip(soc, lo, hi))

    def _apply_standby_losses(self, dt_hours: float) -> StandbyLossBreakdown:
        """
        Compute all standby losses via the decomposed model and apply the
        mechanical component to SoC.  Returns the full breakdown.
        """
        breakdown = self._standby_model.compute(self.speed_ratio, dt_hours)

        # Only mechanical losses drain the flywheel
        self._soc_kwh -= breakdown.mechanical_loss_kwh
        self._soc_kwh  = self._clamp_soc(self._soc_kwh)

        return breakdown

    def _apply_ramp_limit(self, requested_kw: float, dt_hours: float) -> float:
        dt_sec    = dt_hours * 3600.0
        max_delta = self.params.ramp_rate_kw_per_sec * dt_sec
        delta     = requested_kw - self._current_power_kw
        return self._current_power_kw + float(np.clip(delta, -max_delta, max_delta))

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def step(
        self,
        power_setpoint_kw:  float,
        dt_hours:           float,
        apply_ramp_limit:   bool = True,
        log:                bool = True,
    ) -> FESSSnapshot:
        """
        Advance the FESS unit by one timestep.

        Efficiency at each step is computed from the operating-point
        loss curves, not a fixed scalar.  Machine and inverter losses
        are tracked separately in the snapshot.

        Parameters
        ----------
        power_setpoint_kw : float
            Requested grid-side power.
            +ve = charging, -ve = discharging.
        dt_hours : float
            Timestep size in hours.
        apply_ramp_limit : bool
        log : bool

        Returns
        -------
        FESSSnapshot
        """
        if self._state == FESSState.FAULT:
            return self._zero_snapshot(dt_hours)

        P_rat = self.params.rated_power_kw
        sr    = self.speed_ratio

        # 1. Standby losses (every step, before power exchange)
        sb = self._apply_standby_losses(dt_hours)

        # 2. Ramp rate
        if apply_ramp_limit:
            power_setpoint_kw = self._apply_ramp_limit(power_setpoint_kw, dt_hours)

        # 3. Power exchange
        energy_charged_kwh    = 0.0
        energy_discharged_kwh = 0.0
        grid_consumed_kwh     = 0.0
        grid_delivered_kwh    = 0.0
        actual_power_kw       = 0.0
        machine_loss_kw       = 0.0
        inverter_loss_kw      = 0.0
        eta_actual            = 0.0

        if power_setpoint_kw > 1e-3:
            # ── CHARGING ────────────────────────────────────────────────
            max_charge_kw   = min(P_rat, self.power_headroom_kw)
            actual_power_kw = float(np.clip(power_setpoint_kw, 0.0, max_charge_kw))

            if actual_power_kw > 1e-3:
                p_pu  = actual_power_kw / P_rat
                eta_c = self._eff_model.eta_charge(p_pu, sr)
                eta_actual        = eta_c
                grid_consumed_kwh = actual_power_kw * dt_hours
                energy_charged_kwh = grid_consumed_kwh * eta_c

                # SoC ceiling
                max_storable = self.params.rated_energy_kwh - self._soc_kwh
                if energy_charged_kwh > max_storable:
                    energy_charged_kwh = max_storable
                    grid_consumed_kwh  = energy_charged_kwh / eta_c if eta_c > 0 else 0.0
                    actual_power_kw    = grid_consumed_kwh / dt_hours
                    p_pu               = actual_power_kw / P_rat
                    eta_actual         = self._eff_model.eta_charge(p_pu, sr)

                machine_loss_kw  = self._eff_model.machine.loss_kw(p_pu, sr)
                inverter_loss_kw = self._eff_model.inverter.loss_kw(p_pu)
                self._soc_kwh   += energy_charged_kwh
                self._state      = FESSState.CHARGING

        elif power_setpoint_kw < -1e-3:
            # ── DISCHARGING ──────────────────────────────────────────────
            max_discharge_kw = self.power_available_kw
            actual_power_kw  = float(np.clip(power_setpoint_kw, -max_discharge_kw, 0.0))

            if abs(actual_power_kw) > 1e-3:
                p_pu  = abs(actual_power_kw) / P_rat
                eta_d = self._eff_model.eta_discharge(p_pu, sr)
                eta_actual         = eta_d
                grid_delivered_kwh = abs(actual_power_kw) * dt_hours
                energy_discharged_kwh = (
                    grid_delivered_kwh / eta_d if eta_d > 0 else grid_delivered_kwh
                )

                # SoC floor
                soc_floor = self.params.soc_min_frac * self.params.rated_energy_kwh
                available = self._soc_kwh - soc_floor
                if energy_discharged_kwh > available:
                    energy_discharged_kwh = max(available, 0.0)
                    grid_delivered_kwh    = energy_discharged_kwh * eta_d
                    actual_power_kw       = -grid_delivered_kwh / dt_hours
                    p_pu                  = abs(actual_power_kw) / P_rat
                    eta_actual            = self._eff_model.eta_discharge(p_pu, sr)

                machine_loss_kw  = self._eff_model.machine.loss_kw(p_pu, sr)
                inverter_loss_kw = self._eff_model.inverter.loss_kw(p_pu)
                self._soc_kwh   -= energy_discharged_kwh
                self._state      = FESSState.DISCHARGING

        else:
            self._state = FESSState.IDLE

        # 4. Safety clamp
        self._soc_kwh = self._clamp_soc(self._soc_kwh)

        # 5. Ramp tracking
        self._current_power_kw = actual_power_kw

        # 6. Counters
        throughput = energy_charged_kwh + energy_discharged_kwh
        self.cumulative_operating_hours       += dt_hours
        self.cumulative_energy_charged_kwh    += energy_charged_kwh
        self.cumulative_energy_discharged_kwh += energy_discharged_kwh
        self.cumulative_machine_loss_kwh      += machine_loss_kw  * dt_hours
        self.cumulative_inverter_loss_kwh     += inverter_loss_kw * dt_hours
        self.equivalent_full_cycles           += throughput / (2.0 * self.params.rated_energy_kwh)
        self._sim_time_h                      += dt_hours

        # 7. Snapshot
        snap = FESSSnapshot(
            time_h                    = self._sim_time_h,
            state                     = self._state,
            soc_kwh                   = self._soc_kwh,
            soc_frac                  = self.soc_frac,
            speed_ratio               = self.speed_ratio,
            power_kw                  = actual_power_kw,
            power_available_kw        = self.power_available_kw,
            standby_aero_w            = sb.p_aero_w,
            standby_tmb_eddy_w        = sb.p_tmb_eddy_w,
            standby_rmb_eddy_sync_w   = sb.p_rmb_eddy_sync_w,
            standby_mechanical_kw     = sb.total_mechanical_w / 1000.0,
            standby_tmb_bias_w        = sb.p_tmb_bias_w,
            standby_rmb_bias_w        = sb.p_rmb_bias_w,
            standby_rmb_eddy_pwm_w    = sb.p_rmb_eddy_pwm_w,
            standby_cooling_w         = sb.p_cooling_w,
            standby_vacuum_w          = sb.p_vacuum_pump_w,
            standby_auxiliary_kw      = sb.total_auxiliary_w / 1000.0,
            vacuum_pump_running       = sb.vacuum_pump_running,
            machine_loss_kw           = machine_loss_kw,
            inverter_loss_kw          = inverter_loss_kw,
            eta_actual                = eta_actual,
            energy_charged_kwh        = energy_charged_kwh,
            energy_discharged_kwh     = energy_discharged_kwh,
            grid_energy_consumed_kwh  = grid_consumed_kwh,
            grid_energy_delivered_kwh = grid_delivered_kwh,
            cumulative_operating_hours= self.cumulative_operating_hours,
            equivalent_full_cycles    = self.equivalent_full_cycles,
        )

        if log:
            self.history.append(snap)

        return snap

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self, initial_soc_frac: float = 0.5) -> None:
        self._soc_kwh = self.params.rated_energy_kwh * float(
            np.clip(initial_soc_frac, self.params.soc_min_frac, self.params.soc_max_frac)
        )
        self._state            = FESSState.IDLE
        self._current_power_kw = 0.0
        self.cumulative_operating_hours       = 0.0
        self.cumulative_energy_charged_kwh    = 0.0
        self.cumulative_energy_discharged_kwh = 0.0
        self.cumulative_machine_loss_kwh      = 0.0
        self.cumulative_inverter_loss_kwh     = 0.0
        self.equivalent_full_cycles           = 0.0
        self._sim_time_h       = 0.0
        self._standby_model.reset()
        self.history.clear()

    def summary(self) -> dict:
        charged    = self.cumulative_energy_charged_kwh
        discharged = self.cumulative_energy_discharged_kwh
        rt_eff     = discharged / charged if charged > 1e-6 else None
        sb         = self._standby_model.summary()
        return {
            "unit_id":                         self.unit_id,
            "soc_kwh":                         round(self._soc_kwh, 3),
            "soc_frac":                        round(self.soc_frac, 4),
            "speed_ratio":                     round(self.speed_ratio, 4),
            "power_available_kw":              round(self.power_available_kw, 2),
            "cumulative_operating_hours":       round(self.cumulative_operating_hours, 2),
            "cumulative_energy_charged_kwh":    round(charged,    3),
            "cumulative_energy_discharged_kwh": round(discharged, 3),
            "standby_losses":                  sb,
            "cumulative_machine_loss_kwh":      round(self.cumulative_machine_loss_kwh,  3),
            "cumulative_inverter_loss_kwh":     round(self.cumulative_inverter_loss_kwh, 3),
            "equivalent_full_cycles":           round(self.equivalent_full_cycles, 4),
            "realised_roundtrip_efficiency":    round(rt_eff, 4) if rt_eff else None,
        }

    def operating_point(self) -> dict:
        """Live efficiency breakdown at current operating point."""
        p_pu = abs(self._current_power_kw) / self.params.rated_power_kw
        return self._eff_model.operating_point_summary(p_pu, self.speed_ratio)

    def _zero_snapshot(self, dt_hours: float) -> FESSSnapshot:
        self._sim_time_h += dt_hours
        return FESSSnapshot(
            time_h=self._sim_time_h, state=self._state,
            soc_kwh=self._soc_kwh, soc_frac=self.soc_frac,
            speed_ratio=self.speed_ratio, power_kw=0.0,
            power_available_kw=0.0,
            standby_aero_w=0.0, standby_tmb_eddy_w=0.0,
            standby_rmb_eddy_sync_w=0.0, standby_mechanical_kw=0.0,
            standby_tmb_bias_w=0.0, standby_rmb_bias_w=0.0,
            standby_rmb_eddy_pwm_w=0.0, standby_cooling_w=0.0,
            standby_vacuum_w=0.0, standby_auxiliary_kw=0.0,
            vacuum_pump_running=False,
            machine_loss_kw=0.0, inverter_loss_kw=0.0, eta_actual=0.0,
            energy_charged_kwh=0.0, energy_discharged_kwh=0.0,
            grid_energy_consumed_kwh=0.0, grid_energy_delivered_kwh=0.0,
            cumulative_operating_hours=self.cumulative_operating_hours,
            equivalent_full_cycles=self.equivalent_full_cycles,
        )

    def __repr__(self) -> str:
        return (
            f"FESSUnit(id={self.unit_id!r}, "
            f"SoC={self.soc_frac:.1%}, "
            f"speed={self.speed_ratio:.3f}, "
            f"P_avail={self.power_available_kw:.1f} kW, "
            f"cycles={self.equivalent_full_cycles:.2f}, "
            f"state={self._state.value})"
        )
