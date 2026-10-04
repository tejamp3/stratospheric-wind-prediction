"""A simplified super-pressure balloon with ballonet altitude control.

SIMULATION. Simplified research model, not flight-certified. Every assumption
is listed here, and anything not listed is not modelled.

Vehicle
  A sealed envelope of fixed volume V (super-pressure: the gas stays above
  ambient pressure, so the envelope keeps its shape and volume). Inside sits a
  ballonet that can hold air. Total mass m = structure + lift gas + ballonet air.

Vertical
  Net buoyancy B = (rho_air * V - m) * g. The balloon moves towards the height
  where rho_air = m / V (its equilibrium density) at the drag-limited terminal
  speed sqrt(2|B| / (rho C_d A)), never overshooting the equilibrium within a
  step. Inertia, added mass and the slow vertical oscillation about equilibrium
  are ignored: they act over minutes, the controller over hours.
  The equilibrium height uses an exponential density profile with the local
  density scale height R_d T / g.

Altitude control
  Pumping outside air into the ballonet raises m and the balloon sinks; venting
  it lets the balloon rise. Pumping costs energy (work against the envelope's
  super-pressure, divided by pump efficiency); venting is free. Both are rate
  limited. Ballonet capacity bounds the reachable altitude band.

Horizontal
  The balloon moves with the wind at its altitude. Its velocity relaxes to the
  wind in minutes, so it is set equal to the wind; there is no thrust.

Energy
  Battery with solar charging proportional to sin(solar elevation), and a
  constant avionics load. With an empty battery the pump cannot run, so the
  balloon cannot descend.

Not modelled: diurnal superheat and its altitude excursion, gas leakage,
radiative heating, gravity waves and turbulence below the ERA5 resolution,
envelope stress limits, ascent and termination.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stratoballoon.atmosphere import G0, R_D, move

H2_TO_AIR_DENSITY = 2.016 / 28.97


@dataclass
class BalloonParams:
    volume_m3: float
    structure_kg: float
    gas_kg: float
    ballonet_max_kg: float
    cd: float = 0.5
    pump_kg_s: float = 0.010
    vent_kg_s: float = 0.020
    pump_efficiency: float = 0.5
    superpressure_pa: float = 200.0
    battery_wh: float = 3000.0
    solar_peak_w: float = 300.0
    base_load_w: float = 60.0

    @property
    def area_m2(self) -> float:
        r = (3 * self.volume_m3 / (4 * np.pi)) ** (1 / 3)
        return float(np.pi * r * r)

    @classmethod
    def design(cls, rho_ceiling: float, rho_floor: float, structure_kg: float = 120.0,
               superpressure_frac: float = 0.1, **kw) -> "BalloonParams":
        """Size the envelope and ballonet to float between two air densities.

        With an empty ballonet the balloon floats where rho = rho_ceiling (the
        top of the band); with a full one, where rho = rho_floor (the bottom).
        Lift gas mass follows from the gas filling V at a little above ambient
        pressure at the ceiling.
        """
        gas_frac = H2_TO_AIR_DENSITY * (1 + superpressure_frac)
        volume = structure_kg / (rho_ceiling * (1 - gas_frac))
        gas = volume * rho_ceiling * gas_frac
        ballonet = volume * rho_floor - structure_kg - gas
        return cls(volume_m3=volume, structure_kg=structure_kg, gas_kg=gas,
                   ballonet_max_kg=ballonet, **kw)


@dataclass
class BalloonState:
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray          # m
    w: np.ndarray            # vertical speed, m/s
    ballonet_kg: np.ndarray
    battery_wh: np.ndarray
    energy_pump_wh: np.ndarray   # cumulative energy spent pumping

    def copy(self) -> "BalloonState":
        return BalloonState(*(getattr(self, f).copy() for f in self.__dataclass_fields__))


def solar_elevation_sin(hours_utc_since_epoch: np.ndarray, lat, lon) -> np.ndarray:
    """sin of the sun's elevation; good to about a degree, enough for power."""
    days = hours_utc_since_epoch / 24.0
    doy = np.mod(days, 365.25)
    decl = np.radians(-23.44) * np.cos(2 * np.pi * (doy + 10) / 365.25)
    solar_time = np.mod(hours_utc_since_epoch, 24) + lon / 15.0
    ha = np.radians(15 * (solar_time - 12))
    la = np.radians(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)


def equilibrium_offset(rho_here, T_here, total_mass, volume):
    """Height change (m) to neutral buoyancy, exponential density profile."""
    h_rho = R_D * T_here / G0
    return h_rho * np.log(rho_here * volume / total_mass)


def ballonet_for_altitude(p: BalloonParams, rho_here, T_here, alt_here, alt_target):
    """Ballonet mass that makes `alt_target` the equilibrium height."""
    h_rho = R_D * T_here / G0
    rho_target = rho_here * np.exp(-(alt_target - alt_here) / h_rho)
    need = rho_target * p.volume_m3 - p.structure_kg - p.gas_kg
    return np.clip(need, 0.0, p.ballonet_max_kg)


def step(p: BalloonParams, s: BalloonState, atm: dict, alt_target: np.ndarray,
         dt: float, sun_sin: np.ndarray, pump_ok: np.ndarray | None = None,
         battery_cap_wh: np.ndarray | float | None = None) -> BalloonState:
    """Advance one step. `atm` holds u, v, T, rho at the current position.

    `pump_ok` and `battery_cap_wh` let fault injection disable the pump or
    shrink the battery per mission.
    """
    s = s.copy()
    cap = p.battery_wh if battery_cap_wh is None else battery_cap_wh
    # --- ballonet inner loop: move ballonet mass towards what the target needs
    want = ballonet_for_altitude(p, atm["rho"], atm["T"], s.alt, alt_target)
    can_pump = (s.battery_wh > 0) if pump_ok is None else (s.battery_wh > 0) & pump_ok
    d_in = np.where(can_pump, np.clip(want - s.ballonet_kg, 0, p.pump_kg_s * dt), 0.0)
    d_out = np.clip(s.ballonet_kg - want, 0, p.vent_kg_s * dt)
    s.ballonet_kg = s.ballonet_kg + d_in - d_out
    # work to push air into the envelope against its super-pressure
    pump_wh = d_in / atm["rho"] * p.superpressure_pa / p.pump_efficiency / 3600
    s.energy_pump_wh = s.energy_pump_wh + pump_wh
    solar_wh = p.solar_peak_w * np.clip(sun_sin, 0, None) * dt / 3600
    s.battery_wh = np.clip(s.battery_wh + solar_wh - p.base_load_w * dt / 3600 - pump_wh,
                           0.0, cap)

    # --- vertical: drag-limited motion towards equilibrium, no overshoot
    m = p.structure_kg + p.gas_kg + s.ballonet_kg
    buoy = (atm["rho"] * p.volume_m3 - m) * G0
    w_term = np.sign(buoy) * np.sqrt(2 * np.abs(buoy) / (atm["rho"] * p.cd * p.area_m2))
    dz_eq = equilibrium_offset(atm["rho"], atm["T"], m, p.volume_m3)
    dz = np.clip(w_term * dt, -np.abs(dz_eq), np.abs(dz_eq))
    s.w = dz / dt
    s.alt = s.alt + dz

    # --- horizontal: drift with the wind
    s.lat, s.lon = move(s.lat, s.lon, atm["u"], atm["v"], dt)
    return s
