"""Simple Amazon Model"""

import logging
import math
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import tree_math
from flax import struct

import xarray as xr

logger = logging.getLogger(__name__)

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
    atmosphere_windspeed: float | jnp.ndarray = 60 * 86400.0
    soil_depth: float | jnp.ndarray = 3.0 # meters
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
        
        shape = (self.ngrid,)
 
        state = State.zeros(shape)
        derived = Derived.zeros(shape)
        params = Parameters.default() if params is None else params
        
        return {
            "state": state,
            "params": params,
        }

    def step(self, carry: Carry, step: int) -> tuple[Carry, Diagnostics]:
        
        params = carry["params"]
        state = carry["state"]

        new_time = state.time + self.time_step
        new_state = state.replace(
            time = new_time,
        ) 
        
        derived = Derived.zeros(shape=self.grid_shape)

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
