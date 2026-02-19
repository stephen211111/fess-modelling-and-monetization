"""
efficiency_models.py
====================
Physics-grounded efficiency curve models for:

  1. ElectricMachineEfficiency
     Models a permanent-magnet motor/generator (PMSM or PMSG) as used in
     high-speed FESS.  Losses are decomposed into:
       - Copper (I²R) losses        → proportional to torque² (∝ power² at const speed)
       - Iron (core) losses         → proportional to speed² (eddy) + speed (hysteresis)
       - Constant (stray) losses    → fixed offset
     Windage/friction (k_windage) defaults to 0.0 for vacuum AMB machines.
     All bearing drag is accounted for in standby_losses.py to avoid double-counting.

  2. BidirectionalInverterEfficiency
     Models a bidirectional DC/AC power conversion stage (PCS).
     Losses are decomposed into:
       - Switching losses           → proportional to |power| (IGBTs/SiC switches)
       - Conduction losses          → proportional to power²  (resistive, I²R)
       - Fixed (standby) losses     → constant overhead (gate drivers, controls)

Both classes:
  - Are parameterised from nameplate or curve-fit data
  - Return efficiency η ∈ (0, 1] as a function of (power_pu, speed_ratio)
  - Expose a loss_kw() method for direct energy accounting
  - Include a plot_curve() helper for visualisation

Units convention
----------------
  power_pu   : per-unit power  0.0 – 1.0  (fraction of rated power)
  speed_ratio: per-unit speed  0.0 – 1.0  (omega / omega_max)
  loss_kw    : absolute loss in kW given rated_power_kw and operating point
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# 1. Electric Machine Efficiency
# ---------------------------------------------------------------------------

@dataclass
class MachineParams:
    """
    Loss coefficients for a permanent-magnet motor/generator.

    All coefficients are expressed as fractions of rated power so that
    the model is dimensionless and scales correctly with rated_power_kw.

    How to obtain coefficients
    --------------------------
    Option A – from a manufacturer loss breakdown sheet:
        k_copper  = P_copper_rated / P_rated
        k_iron    = P_iron_rated   / P_rated
        k_stray   = P_stray_rated  / P_rated
        k_windage = 0.0  (vacuum AMB machine) or P_windage_rated/P_rated (other)

    Option B – curve-fit from measured efficiency map:
        Use MachineEfficiency.fit_from_data(power_pu, speed_ratio, eta_measured)

    Option C – use these defaults, which represent a well-designed
    high-speed PMSM in vacuum with AMB (η_peak ≈ 97.5% at rated speed, 75% load):
    """
    # Copper (I²R) loss coefficient at rated torque, rated speed
    # Scales as: k_copper * (power_pu / speed_ratio)²
    # Derivation: T ∝ I, loss ∝ I² = (P/ω)² → (P_pu/speed)²
    k_copper: float = 0.018

    # Iron (core) loss coefficient at rated speed
    # Eddy current component scales as speed²; hysteresis as speed.
    # Combined: k_iron_eddy * speed² + k_iron_hyst * speed
    k_iron_eddy:  float = 0.008    # Eddy current (dominant at high speed)
    k_iron_hyst:  float = 0.004    # Hysteresis (more significant at low speed)

    # Windage/friction loss coefficient at rated speed.
    #
    # Default is 0.0 for vacuum-housed active magnetic bearing (AMB) machines:
    #   - Aerodynamic drag is captured by k_aero_w in StandbyLossParams (∝ sr³)
    #   - Bearing eddy and bias losses are captured by k_tmb_eddy_w / k_rmb_eddy_sync_w
    #     and p_tmb_bias_w / k_rmb_bias_w in StandbyLossParams
    # Setting this non-zero with standby bearing terms active creates double-counting.
    # Only set non-zero for non-AMB machines (e.g. ball bearings in air).
    k_windage: float = 0.0

    # Windage speed exponent: 1.0 = friction, 3.0 = gas drag, 1.5 = partial vacuum.
    # Retained for backward compatibility and non-AMB use cases.
    windage_exponent: float = 1.5

    # Stray / miscellaneous losses (constant fraction of rated)
    k_stray: float = 0.002

    # Minimum speed ratio below which the machine is considered stopped
    speed_min: float = 0.05

    def __post_init__(self):
        total_loss_at_rated = (
            self.k_copper + self.k_iron_eddy + self.k_iron_hyst
            + self.k_windage + self.k_stray
        )
        assert total_loss_at_rated < 0.5, (
            f"Loss coefficients sum to {total_loss_at_rated:.3f} — check units."
        )


class MachineEfficiency:
    """
    Efficiency model for a permanent-magnet motor/generator.

    The efficiency at a given operating point is:

        η = P_out / P_in  (motoring / charging)
        η = P_out / P_in  (generating / discharging)

    where total losses = copper + iron_eddy + iron_hyst + windage + stray.

    Loss decomposition (all as fraction of P_rated):
    -------------------------------------------------
        L_copper  = k_copper  * (power_pu / speed_ratio)²
        L_iron    = k_iron_eddy * speed_ratio²  +  k_iron_hyst * speed_ratio
        L_windage = k_windage * speed_ratio ^ windage_exponent  [0.0 for vacuum AMB]
        L_stray   = k_stray

    Total loss fraction:
        L_total = L_copper + L_iron + L_windage + L_stray   (k_windage=0 for FESS default)

    Efficiency:
        η = power_pu / (power_pu + L_total)   [motoring: output/input]
        η = (power_pu - L_total) / power_pu   [generating: output/input]

    Note: both directions use the same loss model but the efficiency
    expression inverts — motoring η < 1 because losses add to input;
    generating η < 1 because losses subtract from output.

    Parameters
    ----------
    params : MachineParams
    rated_power_kw : float
        Rated shaft power (equals rated electrical power at η=1).
    """

    def __init__(self, params: MachineParams, rated_power_kw: float):
        self.params          = params
        self.rated_power_kw  = rated_power_kw

    def _loss_fraction(self, power_pu: float, speed_ratio: float) -> float:
        """
        Total loss as a fraction of rated power at the given operating point.

        Parameters
        ----------
        power_pu    : float  0–1, fraction of rated shaft/electrical power
        speed_ratio : float  0–1, omega/omega_max
        """
        p  = max(abs(power_pu),  1e-6)
        sr = max(speed_ratio,    self.params.speed_min)

        # Copper loss: I²R, torque ∝ current, torque = P/ω
        l_copper = self.params.k_copper * (p / sr) ** 2

        # Iron loss: eddy (ω²) + hysteresis (ω)
        l_iron = (
            self.params.k_iron_eddy * sr ** 2
            + self.params.k_iron_hyst * sr
        )

        # Windage / friction
        l_windage = self.params.k_windage * sr ** self.params.windage_exponent

        # Stray
        l_stray = self.params.k_stray

        return l_copper + l_iron + l_windage + l_stray

    def eta_motoring(self, power_pu: float, speed_ratio: float) -> float:
        """
        Motor (charging) efficiency.
        P_shaft = P_elec * η  →  η = P_pu / (P_pu + L_total)
        """
        if power_pu <= 0:
            return 0.0
        L = self._loss_fraction(power_pu, speed_ratio)
        return float(np.clip(power_pu / (power_pu + L), 0.0, 1.0))

    def eta_generating(self, power_pu: float, speed_ratio: float) -> float:
        """
        Generator (discharging) efficiency.
        P_elec = P_shaft * η  →  η = (P_pu - L_total) / P_pu
        """
        if power_pu <= 0:
            return 0.0
        L = self._loss_fraction(power_pu, speed_ratio)
        return float(np.clip((power_pu - L) / power_pu, 0.0, 1.0))

    def loss_kw(self, power_pu: float, speed_ratio: float) -> float:
        """Absolute machine loss in kW at this operating point."""
        return self._loss_fraction(power_pu, speed_ratio) * self.rated_power_kw

    @classmethod
    def fit_from_data(
        cls,
        power_pu: np.ndarray,
        speed_ratio: np.ndarray,
        eta_measured: np.ndarray,
        rated_power_kw: float,
        mode: str = "motoring",
    ) -> "MachineEfficiency":
        """
        Fit loss coefficients from measured efficiency map data using
        least-squares regression.

        Parameters
        ----------
        power_pu     : 1D array of per-unit power values
        speed_ratio  : 1D array of per-unit speed values (same length)
        eta_measured : 1D array of measured efficiency (same length)
        rated_power_kw : float
        mode         : "motoring" or "generating"

        Returns
        -------
        MachineEfficiency with fitted MachineParams
        """
        from scipy.optimize import curve_fit

        def _eta_model_motoring(X, k_copper, k_iron_eddy, k_iron_hyst,
                                k_windage, k_stray):
            p_pu, sr = X
            p  = np.maximum(np.abs(p_pu), 1e-6)
            sr = np.maximum(sr, 0.05)
            L  = (k_copper * (p / sr)**2
                  + k_iron_eddy * sr**2
                  + k_iron_hyst * sr
                  + k_windage   * sr**1.5
                  + k_stray)
            return p / (p + L)

        def _eta_model_generating(X, k_copper, k_iron_eddy, k_iron_hyst,
                                  k_windage, k_stray):
            p_pu, sr = X
            p  = np.maximum(np.abs(p_pu), 1e-6)
            sr = np.maximum(sr, 0.05)
            L  = (k_copper * (p / sr)**2
                  + k_iron_eddy * sr**2
                  + k_iron_hyst * sr
                  + k_windage   * sr**1.5
                  + k_stray)
            return (p - L) / p

        model_fn = (
            _eta_model_motoring if mode == "motoring"
            else _eta_model_generating
        )
        p0     = [0.018, 0.008, 0.004, 0.006, 0.002]
        bounds = ([0]*5, [0.15]*5)
        X      = (np.asarray(power_pu), np.asarray(speed_ratio))

        popt, _ = curve_fit(
            model_fn, X, np.asarray(eta_measured),
            p0=p0, bounds=bounds, maxfev=10_000
        )
        fitted_params = MachineParams(
            k_copper      = popt[0],
            k_iron_eddy   = popt[1],
            k_iron_hyst   = popt[2],
            k_windage     = popt[3],
            k_stray       = popt[4],
        )
        return cls(fitted_params, rated_power_kw)

    def efficiency_map(
        self,
        power_pu_range: np.ndarray = None,
        speed_range:    np.ndarray = None,
        mode: str = "motoring",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the 2D efficiency map over (speed, power) grid.

        Returns
        -------
        speed_grid, power_grid, eta_grid  — all shape (len(speed), len(power))
        Useful for plotting and for lookup-table optimisation.
        """
        if power_pu_range is None:
            power_pu_range = np.linspace(0.05, 1.0, 40)
        if speed_range is None:
            speed_range = np.linspace(0.2, 1.0, 40)

        P, S = np.meshgrid(power_pu_range, speed_range)
        eta_fn = self.eta_motoring if mode == "motoring" else self.eta_generating

        eta_grid = np.vectorize(eta_fn)(P, S)
        return S, P, eta_grid

    def plot_curve(self, rated_power_kw: float = None, show: bool = True):
        """Plot efficiency vs power at several fixed speeds."""
        import matplotlib.pyplot as plt

        speeds  = [0.3, 0.5, 0.7, 0.9, 1.0]
        p_range = np.linspace(0.05, 1.0, 200)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for sp in speeds:
            eta_m = [self.eta_motoring(p,   sp) for p in p_range]
            eta_g = [self.eta_generating(p, sp) for p in p_range]
            label = f"ω/ωmax = {sp:.1f}"
            axes[0].plot(p_range * 100, np.array(eta_m) * 100,
                         label=label, linewidth=2)
            axes[1].plot(p_range * 100, np.array(eta_g) * 100,
                         label=label, linewidth=2)

        for ax, title in zip(axes, ["Motoring (Charging)", "Generating (Discharging)"]):
            ax.set_xlabel("Power (% of rated)")
            ax.set_ylabel("Efficiency (%)")
            ax.set_title(f"Electric Machine — {title}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(85, 100)

        plt.tight_layout()
        if show:
            plt.savefig("machine_efficiency_curves.png", dpi=150, bbox_inches="tight")
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# 2. Bidirectional Inverter Efficiency
# ---------------------------------------------------------------------------

@dataclass
class InverterParams:
    """
    Loss coefficients for a bidirectional DC/AC power conversion stage.

    Loss decomposition
    ------------------
    Modern SiC-based bidirectional inverters have three main loss mechanisms:

    1. Switching losses (P_sw):
       Proportional to |power| and switching frequency.
       Characterised by the switch energy loss per transition.
       P_sw = k_switch * |power_pu| * rated_power_kw

    2. Conduction losses (P_cond):
       I²R losses through the MOSFET/IGBT on-resistance and inductor ESR.
       Proportional to power² (since I ∝ P at constant voltage).
       P_cond = k_cond * power_pu² * rated_power_kw

    3. Fixed / standby losses (P_fixed):
       Gate drivers, control board, fans, auxiliary supplies.
       Constant regardless of power level.
       P_fixed = k_fixed * rated_power_kw

    How to obtain coefficients
    --------------------------
    From a manufacturer efficiency curve at rated voltage:
        At P=0:    η ≈ 1 - k_fixed          (fixed loss only)
        At P=0.1:  η ≈ solve for k_switch
        At P=1.0:  η_peak ≈ accounts for all three
        Typically: k_switch 0.005-0.012, k_cond 0.003-0.010, k_fixed 0.001-0.004

    These defaults represent a good SiC-based 250 kW PCS:
        η_peak ≈ 98.2% at ~40% load, rated voltage
        η at full load ≈ 97.5%
        η at 10% load  ≈ 96.8%
    """
    k_switch: float = 0.008   # Switching loss coefficient (proportional to |P|)
    k_cond:   float = 0.006   # Conduction loss coefficient (proportional to P²)
    k_fixed:  float = 0.002   # Fixed standby loss coefficient (constant)

    def __post_init__(self):
        total_at_rated = self.k_switch + self.k_cond + self.k_fixed
        assert total_at_rated < 0.3, (
            f"Inverter loss coefficients sum to {total_at_rated:.3f} — check values."
        )


class InverterEfficiency:
    """
    Efficiency model for a bidirectional DC/AC inverter (PCS).

    Works identically in both charge (rectifier) and discharge (inverter)
    directions — bidirectional converters have symmetric loss behaviour
    in modern SiC designs.

    Efficiency expression:
    ----------------------
        L_total = k_fixed + k_switch * |p_pu| + k_cond * p_pu²

        η = p_pu / (p_pu + L_total)   [charging: AC → DC]
        η = p_pu / (p_pu + L_total)   [discharging: DC → AC]
        (same expression — losses increase input or reduce output)

    Note: The inverter model does NOT depend on speed_ratio — it only
    sees AC power magnitude. Speed information is irrelevant to the PCS.

    Parameters
    ----------
    params : InverterParams
    rated_power_kw : float
        Rated AC/DC power of the converter.
    """

    def __init__(self, params: InverterParams, rated_power_kw: float):
        self.params         = params
        self.rated_power_kw = rated_power_kw

    def _loss_fraction(self, power_pu: float) -> float:
        """Total inverter loss as a fraction of rated power."""
        p = max(abs(power_pu), 1e-9)
        return (
            self.params.k_fixed
            + self.params.k_switch * p
            + self.params.k_cond   * p ** 2
        )

    def eta(self, power_pu: float) -> float:
        """
        Inverter efficiency at power_pu (fraction of rated).
        Valid for both charge and discharge directions.

        Parameters
        ----------
        power_pu : float
            Magnitude of power as fraction of rated (0–1).
            Sign is ignored — losses are symmetric.
        """
        if abs(power_pu) < 1e-9:
            return 0.0
        p = abs(power_pu)
        L = self._loss_fraction(p)
        return float(np.clip(p / (p + L), 0.0, 1.0))

    def loss_kw(self, power_pu: float) -> float:
        """Absolute inverter loss in kW at this power level."""
        return self._loss_fraction(abs(power_pu)) * self.rated_power_kw

    @classmethod
    def fit_from_data(
        cls,
        power_pu: np.ndarray,
        eta_measured: np.ndarray,
        rated_power_kw: float,
    ) -> "InverterEfficiency":
        """
        Fit inverter loss coefficients from measured efficiency curve.

        Parameters
        ----------
        power_pu     : 1D array of per-unit power values (0–1)
        eta_measured : 1D array of measured efficiency at those power levels
        rated_power_kw : float

        Returns
        -------
        InverterEfficiency with fitted InverterParams
        """
        from scipy.optimize import curve_fit

        def _eta_model(p_pu, k_fixed, k_switch, k_cond):
            p  = np.maximum(np.abs(p_pu), 1e-9)
            L  = k_fixed + k_switch * p + k_cond * p**2
            return p / (p + L)

        p0     = [0.002, 0.008, 0.006]
        bounds = ([0, 0, 0], [0.05, 0.05, 0.05])
        popt, _ = curve_fit(
            _eta_model, np.asarray(power_pu), np.asarray(eta_measured),
            p0=p0, bounds=bounds, maxfev=5_000,
        )
        fitted_params = InverterParams(
            k_fixed  = popt[0],
            k_switch = popt[1],
            k_cond   = popt[2],
        )
        return cls(fitted_params, rated_power_kw)

    def plot_curve(self, show: bool = True):
        """Plot efficiency and loss breakdown vs power level."""
        import matplotlib.pyplot as plt

        p_range = np.linspace(0.01, 1.0, 300)

        eta_vals    = np.array([self.eta(p) for p in p_range])
        l_fixed     = np.full_like(p_range, self.params.k_fixed * self.rated_power_kw)
        l_switch    = self.params.k_switch * p_range * self.rated_power_kw
        l_cond      = self.params.k_cond   * p_range**2 * self.rated_power_kw

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: efficiency curve
        axes[0].plot(p_range * 100, eta_vals * 100,
                     color="#2196F3", linewidth=2.5)
        axes[0].set_xlabel("Power (% of rated)")
        axes[0].set_ylabel("Efficiency (%)")
        axes[0].set_title("Bidirectional Inverter — Efficiency Curve")
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(90, 100)

        # Mark peak efficiency
        peak_idx = np.argmax(eta_vals)
        axes[0].axvline(p_range[peak_idx] * 100, color="red",
                        linestyle="--", alpha=0.5,
                        label=f"Peak η={eta_vals[peak_idx]:.3%} @ {p_range[peak_idx]:.0%}")
        axes[0].legend(fontsize=9)

        # Right: loss decomposition
        axes[1].stackplot(
            p_range * 100,
            l_fixed, l_switch, l_cond,
            labels=["Fixed (gate/ctrl)", "Switching (IGBT/SiC)", "Conduction (I²R)"],
            colors=["#FF9800", "#F44336", "#9C27B0"],
            alpha=0.75,
        )
        axes[1].set_xlabel("Power (% of rated)")
        axes[1].set_ylabel("Loss (kW)")
        axes[1].set_title("Bidirectional Inverter — Loss Decomposition")
        axes[1].legend(fontsize=8, loc="upper left")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if show:
            plt.savefig("inverter_efficiency_curves.png", dpi=150, bbox_inches="tight")
            plt.show()
        return fig


# ---------------------------------------------------------------------------
# Combined efficiency: machine + inverter in series
# ---------------------------------------------------------------------------

class FESSEfficiencyModel:
    """
    Combined efficiency of the full electrical path:
        Grid ↔ Inverter ↔ Motor/Generator ↔ Flywheel shaft

    Charging  (Grid → Flywheel):
        P_shaft = P_grid * η_inv * η_motor

    Discharging (Flywheel → Grid):
        P_grid = P_shaft * η_gen * η_inv

    This class is the single interface used by FESSUnit — it takes
    the two component models and computes the combined one-way efficiencies
    at any (power_pu, speed_ratio) operating point.

    Parameters
    ----------
    machine : MachineEfficiency
    inverter : InverterEfficiency
    """

    def __init__(self, machine: MachineEfficiency, inverter: InverterEfficiency):
        self.machine  = machine
        self.inverter = inverter

    def eta_charge(self, power_pu: float, speed_ratio: float) -> float:
        """
        Combined charge efficiency: fraction of grid energy stored as
        kinetic energy in the flywheel.

        η_charge = η_inv(P_pu) × η_motor(P_pu, speed)
        """
        return self.inverter.eta(power_pu) * self.machine.eta_motoring(power_pu, speed_ratio)

    def eta_discharge(self, power_pu: float, speed_ratio: float) -> float:
        """
        Combined discharge efficiency: fraction of flywheel kinetic energy
        delivered to the grid.

        η_discharge = η_gen(P_pu, speed) × η_inv(P_pu)
        """
        return self.machine.eta_generating(power_pu, speed_ratio) * self.inverter.eta(power_pu)

    def eta_roundtrip(self, power_pu: float, speed_ratio: float) -> float:
        """Round-trip efficiency at a single operating point."""
        return self.eta_charge(power_pu, speed_ratio) * self.eta_discharge(power_pu, speed_ratio)

    def total_loss_kw(
        self,
        power_kw: float,
        speed_ratio: float,
        rated_power_kw: float,
        charging: bool,
    ) -> float:
        """
        Total combined loss in kW for a given operating point.

        Parameters
        ----------
        power_kw       : Requested power (always positive here)
        speed_ratio    : Current normalised speed
        rated_power_kw : Machine rated power
        charging       : True = motoring direction, False = generating
        """
        p_pu     = power_kw / rated_power_kw
        m_loss   = self.machine.loss_kw(p_pu, speed_ratio)
        inv_loss = self.inverter.loss_kw(p_pu)
        return m_loss + inv_loss

    def operating_point_summary(
        self,
        power_pu: float,
        speed_ratio: float,
    ) -> dict:
        """Return a detailed breakdown of all losses at an operating point."""
        p = abs(power_pu)
        m_loss_m  = self.machine._loss_fraction(p, speed_ratio) * self.machine.rated_power_kw
        inv_loss  = self.inverter._loss_fraction(p)              * self.inverter.rated_power_kw
        return {
            "power_pu":             round(p, 4),
            "speed_ratio":          round(speed_ratio, 4),
            "machine_loss_kw":      round(m_loss_m, 3),
            "inverter_loss_kw":     round(inv_loss, 3),
            "total_loss_kw":        round(m_loss_m + inv_loss, 3),
            "eta_charge":           round(self.eta_charge(p, speed_ratio), 5),
            "eta_discharge":        round(self.eta_discharge(p, speed_ratio), 5),
            "eta_roundtrip":        round(self.eta_roundtrip(p, speed_ratio), 5),
        }

    def plot_roundtrip_map(self, show: bool = True):
        """Plot round-trip efficiency as a 2D heatmap over (speed, power)."""
        import matplotlib.pyplot as plt

        power_range = np.linspace(0.05, 1.0, 60)
        speed_range = np.linspace(0.2,  1.0, 60)
        P, S = np.meshgrid(power_range, speed_range)

        eta_rt = np.vectorize(self.eta_roundtrip)(P, S) * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        cf = ax.contourf(P * 100, S, eta_rt, levels=20, cmap="RdYlGn")
        cs = ax.contour( P * 100, S, eta_rt, levels=10, colors="black",
                         linewidths=0.5, alpha=0.4)
        ax.clabel(cs, fmt="%.1f%%", fontsize=7)
        plt.colorbar(cf, ax=ax, label="Round-trip efficiency (%)")
        ax.set_xlabel("Power (% of rated)")
        ax.set_ylabel("Speed ratio (ω/ωmax)")
        ax.set_title("FESS Round-trip Efficiency Map\n(Machine + Inverter Combined)")
        plt.tight_layout()
        if show:
            plt.savefig("fess_efficiency_map.png", dpi=150, bbox_inches="tight")
            plt.show()
        return fig
