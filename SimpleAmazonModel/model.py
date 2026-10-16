"""Simple Amazon Model"""

import logging
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import tree_math
from flax import struct

import xarray as xr
import jax

logger = logging.getLogger(__name__)

ATMOSPHERE_MASS = 1e4 # kg/m^2
PyTree = Any
Carry = PyTree
Diagnostics = PyTree

@tree_math.struct
class State:

    time: jnp.ndarray    
    soil_water_content: jnp.ndarray
    biomass: jnp.ndarray
    atmosphere_specific_humidity: jnp.ndarray

    @classmethod
    def zeros(cls, shape, soil_water_content=None, biomass=None, atmosphere_specific_humidity=None):
        return cls(
            jnp.zeros(()),
            soil_water_content if soil_water_content is not None else jnp.zeros(shape),
            biomass if biomass is not None else jnp.zeros(shape),
            atmosphere_specific_humidity if atmosphere_specific_humidity is not None else jnp.zeros(shape),
        )

@tree_math.struct
class Derived:

    precipitation: jnp.ndarray
    evaporation: jnp.ndarray
    albedo: jnp.ndarray
    skin_temperature: jnp.ndarray

    @classmethod
    def zeros(
        cls,
        shape,
        precipitation=None,
        evaporation=None,
        albedo=None,
        skin_temperature=None,
    ):
        return cls(
            precipitation if precipitation is not None else jnp.zeros(shape),
            evaporation if evaporation is not None else jnp.zeros(shape),
            albedo if albedo is not None else jnp.zeros(shape),
            skin_temperature if skin_temperature is not None else jnp.zeros(shape),
        )

@struct.dataclass
class Parameters:
    
    atmosphere_humidity_relaxation_time: float | jnp.ndarray = 60 * 86400.0
    incoming_atmosphere_specific_humidity: float | jnp.ndarray = 0.02
    atmosphere_windspeed: float | jnp.ndarray = 1.0 # m s^-1
    soil_depth: float | jnp.ndarray = 3.0 # meters
    soil_relaxation_time: float | jnp.ndarray = 5 * 86400.0 # s
    biomass_growth_time: float | jnp.ndarray = 2 * 360 * 86400.0
    biomass_maximum: float | jnp.ndarray = 30.0 # kg / m^2

    @classmethod
    def default(cls) -> "Parameters":
        """Return the default parameters."""
        return cls()

class Model:

    def __init__(
        self,
        ngrid: int = 100,
        grid_length: float = 50e3, # meters
        time_step: float = 86400.0, # seconds
        params: Parameters | None = None,
    ):

        self.ngrid = ngrid
        self.grid_shape = (ngrid,)
        self.grid_length = grid_length
        self.time_step = time_step
        self.params = Parameters.default() if params is None else params

    def initialize(self, params: Parameters | None = None) -> Carry:
        
        shape = self.grid_shape
 
        init_atmosphere_specific_humidity = jnp.zeros(shape)

        state = State.zeros(shape, atmosphere_specific_humidity = init_atmosphere_specific_humidity)
        derived = Derived.zeros(shape)
        params = Parameters.default() if params is None else params
        
        return {
            "state": state,
            "params": params,
        }

    def _interpolate_onto_U_grid(self, q_T):
        
        q_U = jnp.zeros((self.ngrid+1,))
        q_U = q_U.at[1:-1].set((q_T[1:] + q_T[:-1]) / 2)
        q_U = q_U.at[0].set(q_T[0])
        q_U = q_U.at[-1].set(q_T[-1])
        #jax.debug.print("q_U[0] = {x}", x=q_U[0])
        return q_U

    def _ddx_T_to_T(self, q_T):
        
        q_U = self._interpolate_onto_U_grid(q_T)
        dqdx_T = (q_U[1:] - q_U[:-1]) / self.grid_length
        
        return dqdx_T

    def _ddx_T_to_U(self, q_T):
        dqdx_U = jnp.zeros((self.ngrid+1,))
        dqdx_U = dqdx_U.at[1:-1].set( (q_T[1:] - q_T[:-1]) / self.grid_length )
        dqdx_U = dqdx_U.at[0].set(0)
        dqdx_U = dqdx_U.at[-1].set(0)
        return dqdx_U
 
    def _ddx_U_to_T(self, q_U):
        return (q_U[1:] - q_U[:-1]) / self.grid_length 
 
    def step(self, carry: Carry, step: int) -> tuple[Carry, Diagnostics]:
        
        params = carry["params"]
        state = carry["state"]
        dt = self.time_step
        
        new_time = state.time + dt
        flux = - params.atmosphere_windspeed * self._interpolate_onto_U_grid(state.atmosphere_specific_humidity)
        advection_tendency = self._ddx_U_to_T(flux)
        
        """
        x_n+1 - x_n = - x_n+1 * dt / tau
        x_n+1 (1 + dt/tau) = x_n
        x_n+1 = x_n / (1 + dt/tau)
        
        dx_n = x_n+1 - x_n = x_n / (1+dt/tau) - x_n = x_n (1 / (1+dt/tau) - 1)
        x_n - x_n+1 = x_n (1 - (1+dt/tau)**(-1)) 
        """
        
        # Force a 5mm / day precipitation
        precipitation = state.atmosphere_specific_humidity * ATMOSPHERE_MASS * (
            1.0 - 1.0 / (1.0 + dt/params.atmosphere_humidity_relaxation_time)
        ) / dt # m s^-1
 
        new_atmosphere_specific_humidity = state.atmosphere_specific_humidity + dt * (
            advection_tendency
            - precipitation / ATMOSPHERE_MASS 
        )
        new_atmosphere_specific_humidity = new_atmosphere_specific_humidity.at[0].set(params.incoming_atmosphere_specific_humidity)

        new_soil_water_content = (
            state.soil_water_content / ( 1.0 + dt / params.soil_relaxation_time) 
            + precipitation * dt
        )

        new_state = state.replace(
            time = new_time,
            soil_water_content = new_soil_water_content,
            atmosphere_specific_humidity = new_atmosphere_specific_humidity,
        ) 
        derived = Derived.zeros(
            shape=self.grid_shape,
            precipitation = precipitation,
        )

        new_carry = {
            "state": new_state,
            "params": params,
        }
        
        diagnostics = {
            "state": new_state,
            "derived": derived,
        }
        
        return new_carry, diagnostics

    def create_xarray_dataset(self, diagnostics: Diagnostics) -> xr.Dataset:
        
        state = diagnostics["state"]
        derived = diagnostics["derived"]
        time_dim = ("time",)
        gridded_dims = time_dim + ("x",)

        return xr.Dataset(
            data_vars = {
                "soil_water_content": (
                    gridded_dims,
                    state.soil_water_content,
                    {
                        "long_name": "Soil water content",
                        "units": "m",
                    },
                ),
                "precipitation": (
                    gridded_dims,
                    derived.precipitation * 86400.0 * 1000,
                    {
                        "long_name": "Precipitation",
                        "units": "mm day^-1",
                    },
                ),
                "atmosphere_specific_humidity": (
                    gridded_dims,
                    state.atmosphere_specific_humidity,
                    {
                        "long_name": "Atmosphere specific humidity",
                        "units": "kg kg^-1",
                    },
                ),

            },
            coords = {
                "time" : (
                    time_dim,
                    state.time,
                    {
                        "long_name": "Time",
                        "units": "s",
                    },
                ),
            }
        )
