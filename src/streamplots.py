import xarray as xr
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd

############################################################################################################
# Data Preprocessing Pipeline
############################################################################################################

# Load the NetCDF dataset
data = xr.open_dataset('../data/oscar_currents_nrt_20210101.nc')  


zonalCurrent = data['u'].isel(time=0)   # shape: (1440, 719)
meridionalCurrent = data['v'].isel(time=0)  # shape: (1440, 719)

# Print shapes for debugging
print(f"Shape of zonal current data: {zonalCurrent.shape}")
print(f"Shape of meridional current data: {meridionalCurrent.shape}")

# Extract latitude and longitude variables
latitudes = data['lat']  # Shape: (719,)
longitudes = data['lon']  # Shape: (1440,)

# Convert to numpy arrays
lat_vals = latitudes.values
lon_vals = longitudes.values


u_vals = zonalCurrent.values.T  # Now shape: (719, 1440)
v_vals = meridionalCurrent.values.T  # Now shape: (719, 1440)

# make latitudes go in ascending order (from south to north)
if lat_vals[0] > lat_vals[-1]:
    lat_vals = lat_vals[::-1]
    u_vals = u_vals[::-1, :]
    v_vals = v_vals[::-1, :]

# transform longitudes to range [-180, 180)
lon_vals_shifted = (lon_vals + 180) % 360 - 180  


sorted_indices = np.argsort(lon_vals_shifted)
lon_vals_shifted = lon_vals_shifted[sorted_indices]
u_vals = u_vals[:, sorted_indices]
v_vals = v_vals[:, sorted_indices]


lon2d, lat2d = np.meshgrid(lon_vals_shifted, lat_vals)

# magnitude
speed = np.sqrt(u_vals**2 + v_vals**2)


speed = ma.masked_invalid(speed)
u_vals = ma.masked_invalid(u_vals)
v_vals = ma.masked_invalid(v_vals)


speed_flat = speed.compressed()
speed_95th = np.percentile(speed_flat, 95)
speed_min = np.nanmin(speed_flat)


# Compute the gradients (partial derivatives) of u and v components
# The gradient function returns derivatives along the first (latitude) and second (longitude) axes
du_dlat, du_dlon = np.gradient(u_vals, lat_vals, lon_vals_shifted, edge_order=2)
dv_dlat, dv_dlon = np.gradient(v_vals, lat_vals, lon_vals_shifted, edge_order=2)


du_dlat = ma.masked_invalid(du_dlat)
du_dlon = ma.masked_invalid(du_dlon)
dv_dlat = ma.masked_invalid(dv_dlat)
dv_dlon = ma.masked_invalid(dv_dlon)

# Identifying critical points where the magnitude of the vector field is minimal
# also define a threshold for the speed to consider as critical points (can change second param to consider more 'critical' points)
speed_threshold = np.percentile(speed.compressed(), 1)  # Lower 1% of speeds
critical_point_mask = speed <= speed_threshold


critical_indices = np.where(critical_point_mask)
critical_points = []

for idx in zip(*critical_indices):
    lat_idx, lon_idx = idx

    # Extract the Jacobian matrix components
    J_elements = [
        du_dlon[lat_idx, lon_idx], du_dlat[lat_idx, lon_idx],
        dv_dlon[lat_idx, lon_idx], dv_dlat[lat_idx, lon_idx]
    ]

    
    if any(ma.is_masked(element) for element in J_elements):
        continue  
    if not np.all(np.isfinite(J_elements)):
        continue  

    # Construct the Jacobian matrix
    J = np.array([
        [du_dlon[lat_idx, lon_idx], du_dlat[lat_idx, lon_idx]],
        [dv_dlon[lat_idx, lon_idx], dv_dlat[lat_idx, lon_idx]]
    ])

    # Compute eigenvalues of the Jacobian matrix
    eigenvalues, _ = np.linalg.eig(J)

    real_parts = eigenvalues.real
    imag_parts = eigenvalues.imag

    # Classify the critical point based on eigenvalues
    if np.all(imag_parts == 0):  # Eigenvalues are real
        if np.all(real_parts > 0):
            cp_type = 'Repelling Node (Source)'
        elif np.all(real_parts < 0):
            cp_type = 'Attracting Node (Sink)'
        elif real_parts[0] * real_parts[1] < 0:
            cp_type = 'Saddle Point'
        else:
            cp_type = 'Degenerate Node'
    elif np.any(imag_parts != 0):  # Eigenvalues are complex
        if np.all(real_parts < 0):
            cp_type = 'Spiral Sink (Attracting Focus)'
        elif np.all(real_parts > 0):
            cp_type = 'Spiral Source (Repelling Focus)'
        elif np.all(real_parts == 0):
            cp_type = 'Center (Elliptic Point)'
        else:
            cp_type = 'Spiral Saddle'
    else:
        cp_type = 'Unknown'

    # Collect critical point data
    critical_point = {
        'latitude': lat2d[lat_idx, lon_idx],
        'longitude': lon2d[lat_idx, lon_idx],
        'eigenvalues': eigenvalues,
        'type': cp_type
    }
    critical_points.append(critical_point)

print(f"[DEBUG] Number of critical points detected: {len(critical_points)}")

############################################################################################################
# Organize and Output Classification Results
############################################################################################################

# Convert the list of critical points to a DataFrame
df_cp = pd.DataFrame(critical_points)

# Print counts of each type of critical point
cp_type_counts = df_cp['type'].value_counts()
print("[DEBUG] Critical Point Counts by Type:")
print(cp_type_counts)

saddle_points = df_cp[df_cp['type'] == 'Saddle Point']
print(f"\n [DEBUG] Saddle Points (Count: {len(saddle_points)}):")


############################################################################################################
# Visualization Pipeline with Streamplot
############################################################################################################

# Define contour levels for plotting speed. Third param is the number of levels, I found 21 to be a nice number
clevs = np.linspace(speed_min, speed_95th, 21)


plt.figure(figsize=(12, 8))

projection = ccrs.Orthographic(central_longitude=-20, central_latitude=20) #using orthographic projection for the globe


ax = plt.axes(projection=projection)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_global()

# plot filled contours of current speed
cf = ax.contourf(
    lon2d, lat2d, speed, levels=clevs,
    transform=ccrs.PlateCarree(), cmap='viridis'
)

# plot streamlines of the current vectors
strm = ax.streamplot(
    lon2d, lat2d, u_vals, v_vals,
    density=2, linewidth=0.7, arrowsize=1,
    color='k', transform=ccrs.PlateCarree()
)

# i still need to make the colorbar more pretty
cb = plt.colorbar(cf, orientation='horizontal', pad=0.05, shrink=0.7)
cb.set_label('Current Speed (m/s)')
plt.title('Surface Ocean Currents with Streamlines')
plt.show()
