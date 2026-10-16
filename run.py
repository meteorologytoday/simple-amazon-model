from SimpleAmazonModel.model import Model, Parameters
import jax
import xarray as xr


SECONDS_OF_A_DAY = 86400.0

record_time_length = SECONDS_OF_A_DAY
domain_total_length = 1000e3 # m
ngrid = 200
grid_length = domain_total_length / ngrid

time_step = 600.0
total_time = 50.0 * SECONDS_OF_A_DAY
params = Parameters(
    soil_relaxation_time = 5.0 * SECONDS_OF_A_DAY,
    atmosphere_windspeed = 1.0,
    atmosphere_humidity_relaxation_time = 1 * SECONDS_OF_A_DAY
)
model = Model(
    time_step = time_step,
    grid_length = grid_length,
    ngrid = ngrid,
)

print("Initializing ...")
init_carry = model.initialize()

print("Running")
final_carray, diagnostics = jax.lax.scan(model.step, init=init_carry, length=int(total_time / time_step))

ds = model.create_xarray_dataset(diagnostics)

print(ds)
output_file="output.nc"

print(f"Output file: {output_file:s}")
ds = ds.isel(time=slice(None, None, int(record_time_length / time_step)))
ds.to_netcdf(output_file)
