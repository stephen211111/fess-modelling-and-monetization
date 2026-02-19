"""
standby_losses.py
=================
Physically decomposed standby (self-discharge) loss model for a FESS unit.

Every loss mechanism is modelled with its correct speed-scaling law derived
from first principles.  The model is intentionally separate from the
electrical efficiency model (efficiency_models.py) because standby losses
drain kinetic energy from the flywheel directly — they are not part of the
charge/discharge electrical path.

Loss inventory
--------------

1. AERODYNAMIC DRAG (gas drag in partial vacuum)
   The flywheel rotor spins in a housing evacuated to ~1-10 Pa.
   Residual gas drag on a cylinder scales as:
       P_aero ∝ ρ_gas × ω³ × r⁵
   In normalised form:
       P_aero = k_aero × speed_ratio³
   At ~1 Pa residual pressure, this is very small (~10 W at rated speed).

2. THRUST MAGNETIC BEARING (TMB) — two sub-components

   2a. TMB bias coil power
       Active magnetic bearings require a constant DC bias current in their
       coils to linearise force response and provide a stable operating point.
       This power is independent of rotor speed.
       P_tmb_bias = constant

   2b. TMB eddy current losses
       The TMB thrust collar rotates through a nominally axisymmetric field.
       Eddy currents are driven by the rotor sweeping past spatial field
       non-uniformities (pole edges, manufacturing asymmetry).
       Induced EMF ∝ ω  →  P_eddy ∝ ω²
       P_tmb_eddy = k_tmb_eddy × speed_ratio²

3. RADIAL MAGNETIC BEARING (RMB) — three sub-components

   3a. RMB bias coil power
       Same linearisation bias as TMB.  Constant with speed.
       P_rmb_bias = constant

   3b. RMB eddy currents — PWM ripple component  (ELECTRICAL auxiliary, not mechanical)
       Power amplifiers driving RMB coils switch at a carrier frequency (10–20 kHz).
       This high-frequency flux ripple induces eddy currents in the rotor journal
       at the carrier frequency, completely independent of rotational speed.
       The energy source is the amplifier power supply (grid-side), NOT the flywheel
       kinetic energy — this is an auxiliary electrical load, correctly classified in
       total_auxiliary_w.  It must not be confused with mechanically driven eddy
       currents (k_rmb_eddy_sync_w) which do drain kinetic energy.
       P_rmb_eddy_pwm = constant  (grid draw)

   3c. RMB eddy currents — synchronous component
       Any rotor eccentricity (always present) and homopolar DC bias flux
       create a flux pattern that is asynchronous in the rotor frame at
       1× running frequency.  The induced eddy current power scales as:
       Induced EMF ∝ ω  →  P_eddy ∝ ω²
       P_rmb_eddy_sync = k_rmb_eddy_sync × speed_ratio²

4. COOLING SYSTEM
   A small water pump + fan for rotor/stator cooling.
   Runs continuously while the unit is operational.
   Realistic values:
       Water pump: 150–300 W (small centrifugal, ~50 LPM)
       Fan:         80–150 W (axial, 0.3–0.5 m³/s)
   → Total: ~400 W, modelled as constant.
   This drains from the GRID (auxiliary load), not from the flywheel.
   Tracked separately in the snapshot.

5. VACUUM PUMP
   Maintains housing vacuum (~1–10 Pa).  Leakage rate is small so the pump
   does not run continuously — it cycles on for ~5-minute bursts every few
   hours to restore vacuum lost through seals and outgassing.
   Realistic values:
       Pump power:    500–1500 W (dry scroll or roots blower)
       Duty cycle:    5 min on / ~4 h off  →  ~2% duty cycle
   → Average power: ~25 W.  Modelled stochastically with configurable
     on-duration and inter-cycle period.
   Also drains from the GRID (auxiliary load), not from the flywheel.

Units convention
----------------
  All powers   : watts (W) for standby losses (they are small)
  Energy loss  : converted to kWh when updating SoC
  Speed ratio  : 0–1 normalised (omega / omega_max)
  Time         : seconds internally, hours at the public interface
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class StandbyLossParams:
    """
    Parameters for the decomposed FESS standby loss model.

    All powers are in WATTS at the operating condition described.
    The separation into flywheel-draining (mechanical) and grid-draining
    (electrical auxiliary) losses is explicit — see category comments below.

    Mechanical losses (drain flywheel kinetic energy)
    -------------------------------------------------
    k_aero_w              : Aerodynamic drag power at rated speed (W)
                            Typical: 8–15 W for a 250 kW unit in ~1 Pa vacuum
    k_tmb_eddy_w          : TMB eddy current loss at rated speed (W)
                            Typical: 5–20 W
    k_rmb_eddy_sync_w     : RMB synchronous eddy current loss at rated speed (W)
                            Typical: 10–30 W

    Electrical auxiliary losses (drain from grid connection)
    --------------------------------------------------------
    p_tmb_bias_w          : TMB bias coil steady-state power (W)
                            Typical: 20–80 W (depends on bearing gap and load)
    p_rmb_bias_w          : RMB bias coil steady-state power (W) — per bearing
                            Typical: 40–120 W per RMB (most machines have 2)
    p_rmb_eddy_pwm_w      : RMB PWM-ripple eddy loss, constant (W)
                            Typical: 15–40 W
    p_cooling_w           : Water pump + fan continuous power (W)
                            Default: 400 W (250 W pump + 150 W fan)
    p_vacuum_pump_w       : Vacuum pump peak power when running (W)
                            Default: 800 W (dry scroll pump)
    vacuum_on_duration_s  : Pump run duration per cycle (seconds)
                            Default: 300 s (5 minutes)
    vacuum_cycle_period_s : Time between pump start events (seconds)
                            Default: 14400 s (4 hours)
    """

    # ── Mechanical (kinetic energy drain, speed-dependent) ──────────────────

    # Aerodynamic: P ∝ ω³ → speed_ratio³
    k_aero_w: float = 10.0          # W at rated speed (very small in vacuum)

    # TMB eddy current: P ∝ ω² → speed_ratio²
    k_tmb_eddy_w: float = 50.0      # W at rated speed

    # RMB synchronous eddy: P ∝ ω² → speed_ratio²
    k_rmb_eddy_sync_w: float = 50.0 # W at rated speed

    # ── Electrical auxiliary (grid-side draw, constant unless noted) ─────────

    # TMB bias coil — constant (supports axial gravity load, RPM-independent)
    p_tmb_bias_w: float = 150.0     # W

    # RMB bias coil — scales as ω² (corrects unbalance forces ∝ ω²)
    # k_rmb_bias_w is the power at rated speed (both radial bearings combined)
    # At low speed: power is near zero; at rated speed: ~160 W typical
    k_rmb_bias_w: float = 160.0     # W at rated speed (sr=1.0)

    # RMB PWM-ripple eddy — constant
    p_rmb_eddy_pwm_w: float = 25.0  # W

    # Cooling (water pump + fan) — constant while running
    p_cooling_w: float = 465.0      # W  (COP=30, rated operating point heat load)

    # Vacuum pump — intermittent
    p_vacuum_pump_w: float = 800.0       # W when running
    vacuum_on_duration_s: float = 300.0  # 5 minutes per cycle
    vacuum_cycle_period_s: float = 14400.0  # 4 hours between cycles

    def __post_init__(self):
        assert self.k_aero_w         >= 0
        assert self.k_tmb_eddy_w     >= 0
        assert self.k_rmb_eddy_sync_w >= 0
        assert self.p_tmb_bias_w     >= 0
        assert self.k_rmb_bias_w     >= 0
        assert self.p_rmb_eddy_pwm_w >= 0
        assert self.p_cooling_w      >= 0
        assert self.p_vacuum_pump_w  >= 0
        assert self.vacuum_on_duration_s  > 0
        assert self.vacuum_cycle_period_s > self.vacuum_on_duration_s


# ---------------------------------------------------------------------------
# Breakdown result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StandbyLossBreakdown:
    """
    Full loss breakdown for a single timestep.

    Mechanical losses (W) — subtract from flywheel kinetic energy.
    Auxiliary loads (W)  — drawn from grid connection.
    """
    # ── Mechanical (flywheel kinetic energy drain) ───────────────────────
    p_aero_w:           float   # Aerodynamic drag           ∝ sr³
    p_tmb_eddy_w:       float   # TMB eddy currents          ∝ sr²
    p_rmb_eddy_sync_w:  float   # RMB synchronous eddy       ∝ sr²
    total_mechanical_w: float   # Sum of above

    # ── Electrical auxiliary (grid draw) ────────────────────────────────
    p_tmb_bias_w:       float   # TMB bias coil              constant
    p_rmb_bias_w:       float   # RMB bias coil(s)           ∝ sr²
    p_rmb_eddy_pwm_w:   float   # RMB PWM-ripple eddy        constant
    p_cooling_w:        float   # Water pump + fan           constant
    p_vacuum_pump_w:    float   # Vacuum pump (if running)   intermittent
    total_auxiliary_w:  float   # Sum of above

    # ── Derived energy quantities (kWh over the timestep) ───────────────
    mechanical_loss_kwh: float  # Kinetic energy drained from flywheel
    auxiliary_kwh:       float  # Grid energy consumed by auxiliaries

    # ── Vacuum pump state ───────────────────────────────────────────────
    vacuum_pump_running: bool


# ---------------------------------------------------------------------------
# Standby Loss Model
# ---------------------------------------------------------------------------

class StandbyLossModel:
    """
    Computes and accumulates all FESS standby losses each timestep.

    Mechanical losses (aerodynamic + bearing eddy currents) drain kinetic
    energy directly from the flywheel and are subtracted from SoC.

    Electrical auxiliary loads (bearing bias coils, cooling, vacuum pump)
    are drawn from the grid connection.  They appear as a separate power
    draw in the plant model, not as SoC reduction.

    Vacuum pump is modelled as a deterministic on/off cycle — the pump
    runs for vacuum_on_duration_s every vacuum_cycle_period_s.  The phase
    is initialised to zero (pump starts at t=0 and first runs at t=0).
    If a different start phase is desired, pass initial_vacuum_phase_s.

    Parameters
    ----------
    params : StandbyLossParams
    initial_vacuum_phase_s : float
        Seconds into the current vacuum cycle at t=0.
        Default 0 means the pump starts running immediately.
        Pass vacuum_cycle_period_s/2 to start mid-cycle (pump off).
    """

    def __init__(
        self,
        params: StandbyLossParams,
        initial_vacuum_phase_s: float = 0.0,
    ):
        self.params = params
        self._vacuum_phase_s = initial_vacuum_phase_s % params.vacuum_cycle_period_s

        # Cumulative accumulators
        self.cumulative_mechanical_loss_kwh: float = 0.0
        self.cumulative_auxiliary_kwh:       float = 0.0
        self.cumulative_aero_loss_kwh:       float = 0.0
        self.cumulative_tmb_eddy_kwh:        float = 0.0
        self.cumulative_rmb_eddy_sync_kwh:   float = 0.0
        self.cumulative_bearing_bias_kwh:    float = 0.0
        self.cumulative_cooling_kwh:         float = 0.0
        self.cumulative_vacuum_kwh:          float = 0.0
        self.total_vacuum_on_seconds:        float = 0.0

    # ------------------------------------------------------------------
    # Vacuum pump state
    # ------------------------------------------------------------------

    def _advance_vacuum_pump(self, dt_seconds: float) -> tuple[bool, float]:
        """
        Advance the vacuum pump cycle by dt_seconds.

        Returns
        -------
        pump_running : bool
            True if pump is ON for the majority of this timestep.
        active_seconds : float
            Seconds the pump was actually running within dt_seconds.
            Used for accurate energy calculation with long timesteps.
        """
        p = self.params
        phase_start = self._vacuum_phase_s
        phase_end   = phase_start + dt_seconds

        # Pump is ON when phase < vacuum_on_duration_s within each cycle
        # For short timesteps (dt << cycle_period) this is simple:
        if dt_seconds <= p.vacuum_cycle_period_s:
            # Count ON seconds within [phase_start, phase_end) modulo cycle
            # Handle the wrap-around at cycle boundary
            on_seconds = 0.0
            t = 0.0
            while t < dt_seconds:
                phase_now = (phase_start + t) % p.vacuum_cycle_period_s
                remaining_dt = dt_seconds - t
                if phase_now < p.vacuum_on_duration_s:
                    # Currently in ON phase
                    time_to_off = p.vacuum_on_duration_s - phase_now
                    on_seconds += min(time_to_off, remaining_dt)
                    t += min(time_to_off, remaining_dt)
                else:
                    # Currently in OFF phase
                    time_to_on = p.vacuum_cycle_period_s - phase_now
                    t += min(time_to_on, remaining_dt)
        else:
            # dt spans multiple cycles — use duty cycle
            duty = p.vacuum_on_duration_s / p.vacuum_cycle_period_s
            on_seconds = dt_seconds * duty

        self._vacuum_phase_s = phase_end % p.vacuum_cycle_period_s
        pump_running = on_seconds > (dt_seconds / 2.0)
        return pump_running, on_seconds

    # ------------------------------------------------------------------
    # Main compute method
    # ------------------------------------------------------------------

    def compute(
        self,
        speed_ratio: float,
        dt_hours: float,
    ) -> StandbyLossBreakdown:
        """
        Compute all standby losses for one timestep.

        Parameters
        ----------
        speed_ratio : float
            Current normalised speed omega/omega_max (0–1).
        dt_hours : float
            Timestep size in hours.

        Returns
        -------
        StandbyLossBreakdown
            Full per-category breakdown.  The caller (FESSUnit) uses
            breakdown.mechanical_loss_kwh to update SoC, and
            breakdown.auxiliary_kwh as additional grid draw.
        """
        sr        = max(speed_ratio, 0.0)
        dt_s      = dt_hours * 3600.0
        p         = self.params

        # ── Mechanical losses (W) ──────────────────────────────────────
        # Aerodynamic drag: residual gas in vacuum housing, P ∝ ω³
        p_aero = p.k_aero_w * sr ** 3

        # TMB eddy currents: spatial field asymmetry, P ∝ ω²
        p_tmb_eddy = p.k_tmb_eddy_w * sr ** 2

        # RMB synchronous eddy: eccentricity + homopolar bias, P ∝ ω²
        p_rmb_eddy_sync = p.k_rmb_eddy_sync_w * sr ** 2

        total_mech_w = p_aero + p_tmb_eddy + p_rmb_eddy_sync

        # ── Electrical auxiliary losses (W) ────────────────────────────
        # TMB bias: constant (axial gravity load, RPM-independent)
        p_tmb_bias    = p.p_tmb_bias_w

        # RMB bias: scales as ω² — corrects unbalance forces ∝ ω²
        p_rmb_bias    = p.k_rmb_bias_w * sr ** 2

        p_rmb_eddy_pwm = p.p_rmb_eddy_pwm_w

        # Cooling: constant
        p_cooling = p.p_cooling_w

        # Vacuum pump: intermittent cycle
        pump_running, pump_on_s = self._advance_vacuum_pump(dt_s)

        # Energy-accurate vacuum power: based on actual on-seconds
        p_vacuum_avg = p.p_vacuum_pump_w * (pump_on_s / dt_s) if dt_s > 0 else 0.0

        total_aux_w = (
            p_tmb_bias + p_rmb_bias + p_rmb_eddy_pwm
            + p_cooling + p_vacuum_avg
        )

        # ── Convert to energy (kWh) ────────────────────────────────────
        w_to_kwh = dt_s / 3_600_000.0   # W·s → kWh

        mech_kwh = total_mech_w * dt_s / 3600.0 / 1000.0  # kWh
        aux_kwh  = total_aux_w  * dt_s / 3600.0 / 1000.0  # kWh

        # ── Accumulate ────────────────────────────────────────────────
        self.cumulative_mechanical_loss_kwh += mech_kwh
        self.cumulative_auxiliary_kwh       += aux_kwh
        self.cumulative_aero_loss_kwh       += p_aero     * w_to_kwh
        self.cumulative_tmb_eddy_kwh        += p_tmb_eddy * w_to_kwh
        self.cumulative_rmb_eddy_sync_kwh   += p_rmb_eddy_sync * w_to_kwh
        self.cumulative_bearing_bias_kwh    += (p_tmb_bias + p_rmb_bias + p_rmb_eddy_pwm) * w_to_kwh
        self.cumulative_cooling_kwh         += p_cooling * w_to_kwh
        self.cumulative_vacuum_kwh          += p_vacuum_avg * w_to_kwh
        self.total_vacuum_on_seconds        += pump_on_s

        return StandbyLossBreakdown(
            p_aero_w            = p_aero,
            p_tmb_eddy_w        = p_tmb_eddy,
            p_rmb_eddy_sync_w   = p_rmb_eddy_sync,
            total_mechanical_w  = total_mech_w,
            p_tmb_bias_w        = p_tmb_bias,
            p_rmb_bias_w        = p_rmb_bias,
            p_rmb_eddy_pwm_w    = p_rmb_eddy_pwm,
            p_cooling_w         = p_cooling,
            p_vacuum_pump_w     = p_vacuum_avg,
            total_auxiliary_w   = total_aux_w,
            mechanical_loss_kwh = mech_kwh,
            auxiliary_kwh       = aux_kwh,
            vacuum_pump_running = pump_running,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self, initial_vacuum_phase_s: float = 0.0) -> None:
        self._vacuum_phase_s = initial_vacuum_phase_s % self.params.vacuum_cycle_period_s
        self.cumulative_mechanical_loss_kwh = 0.0
        self.cumulative_auxiliary_kwh       = 0.0
        self.cumulative_aero_loss_kwh       = 0.0
        self.cumulative_tmb_eddy_kwh        = 0.0
        self.cumulative_rmb_eddy_sync_kwh   = 0.0
        self.cumulative_bearing_bias_kwh    = 0.0
        self.cumulative_cooling_kwh         = 0.0
        self.cumulative_vacuum_kwh          = 0.0
        self.total_vacuum_on_seconds        = 0.0

    def summary(self) -> dict:
        """Cumulative loss summary."""
        total = (
            self.cumulative_mechanical_loss_kwh
            + self.cumulative_auxiliary_kwh
        )
        return {
            "total_standby_kwh":             round(total, 4),
            "mechanical_loss_kwh":           round(self.cumulative_mechanical_loss_kwh, 4),
            "auxiliary_kwh":                 round(self.cumulative_auxiliary_kwh, 4),
            "breakdown_mechanical": {
                "aero_kwh":                  round(self.cumulative_aero_loss_kwh,   5),
                "tmb_eddy_kwh":              round(self.cumulative_tmb_eddy_kwh,    5),
                "rmb_eddy_sync_kwh":         round(self.cumulative_rmb_eddy_sync_kwh, 5),
            },
            "breakdown_auxiliary": {
                "bearing_bias_kwh":          round(self.cumulative_bearing_bias_kwh, 4),
                "cooling_kwh":               round(self.cumulative_cooling_kwh,      4),
                "vacuum_kwh":                round(self.cumulative_vacuum_kwh,       4),
            },
            "vacuum_on_hours":               round(self.total_vacuum_on_seconds / 3600, 3),
        }

    def instantaneous_power_profile(self, speed_ratio: float) -> dict:
        """
        Return instantaneous power for each loss component at a given speed.
        Useful for plotting and teaching — does NOT advance state.
        """
        sr = max(speed_ratio, 0.0)
        p  = self.params

        # Vacuum pump: use average power based on duty cycle
        vac_duty   = p.vacuum_on_duration_s / p.vacuum_cycle_period_s
        vac_avg_w  = p.p_vacuum_pump_w * vac_duty

        p_rmb_bias = p.k_rmb_bias_w * sr ** 2

        mech_total = (
            p.k_aero_w            * sr**3
            + p.k_tmb_eddy_w      * sr**2
            + p.k_rmb_eddy_sync_w * sr**2
        )
        aux_total = (
            p.p_tmb_bias_w + p_rmb_bias
            + p.p_rmb_eddy_pwm_w + p.p_cooling_w + vac_avg_w
        )

        return {
            # Mechanical (W) — speed dependent
            "aero_w":               round(p.k_aero_w            * sr**3, 2),
            "tmb_eddy_w":           round(p.k_tmb_eddy_w        * sr**2, 2),
            "rmb_eddy_sync_w":      round(p.k_rmb_eddy_sync_w   * sr**2, 2),
            "total_mechanical_w":   round(mech_total, 2),
            # Auxiliary (W) — constant or speed-dependent
            "tmb_bias_w":           round(p.p_tmb_bias_w,        2),
            "rmb_bias_w":           round(p_rmb_bias,            2),
            "rmb_eddy_pwm_w":       round(p.p_rmb_eddy_pwm_w,   2),
            "cooling_w":            round(p.p_cooling_w,         2),
            "vacuum_avg_w":         round(vac_avg_w,             2),
            "total_auxiliary_w":    round(aux_total,             2),
            # Grand total
            "grand_total_w":        round(mech_total + aux_total, 2),
        }
