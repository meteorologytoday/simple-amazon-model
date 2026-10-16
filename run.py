from SimpleAmazonModel.model import Model
import jax
import xarray as xr


model = Model()

print("Initializing ...")
init_carry = model.initialize()

print("Running")
final_carray, diagnostics = jax.lax.scan(model.step, init=init_carry, length=5)

ds = model.create_xarray_dataset(diagnostics)

print(ds)
ds.to_netcdf("output.nc")
