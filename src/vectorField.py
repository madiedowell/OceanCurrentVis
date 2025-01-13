import xarray as xr
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

############################################################################################################
# Data Preprocessing Pipeline
############################################################################################################

# Load the NetCDF dataset containing ocean current data
data = xr.open_dataset('../data/oscar_currents_nrt_20210101.nc')  # Replace with your file path

# Select the first time slice of the zonal and meridional currents
zonalCurrent = data['u'].isel(time=0)        # Shape: (1440, 719)
meridionalCurrent = data['v'].isel(time=0)   # Shape: (1440, 719)

# Print shapes for debugging
print(f"[DEBUG] Shape of zonal current data: {zonalCurrent.shape}")
print(f"[DEBUG] Shape of meridional current data: {meridionalCurrent.shape}")

# Extract latitude and longitude variables
latitudes = data['lat']  # Shape: (719,)
longitudes = data['lon']  # Shape: (1440,)

# Convert to numpy arrays
lat_vals = latitudes.values
lon_vals = longitudes.values


u_vals = zonalCurrent.values.T       # Now shape: (719, 1440)
v_vals = meridionalCurrent.values.T  # Now shape: (719, 1440)


if lat_vals[0] > lat_vals[-1]:
    lat_vals = lat_vals[::-1]
    u_vals = u_vals[::-1, :]
    v_vals = v_vals[::-1, :]


lon_vals_shifted = (lon_vals + 180) % 360 - 180  # Now in range [-180, 180)


sorted_indices = np.argsort(lon_vals_shifted)
lon_vals_shifted = lon_vals_shifted[sorted_indices]
u_vals = u_vals[:, sorted_indices]
v_vals = v_vals[:, sorted_indices]


lon2d, lat2d = np.meshgrid(lon_vals_shifted, lat_vals)


speed = np.sqrt(u_vals**2 + v_vals**2)


speed = ma.masked_invalid(speed)
u_vals = ma.masked_invalid(u_vals)
v_vals = ma.masked_invalid(v_vals)

speed_flat = speed.compressed()
speed_95th = np.percentile(speed_flat, 95)
speed_min = np.nanmin(speed_flat)

############################################################################################################
# Critical Point Detection and Classification
############################################################################################################

du_dlat, du_dlon = np.gradient(u_vals, lat_vals, lon_vals_shifted, edge_order=2)
dv_dlat, dv_dlon = np.gradient(v_vals, lat_vals, lon_vals_shifted, edge_order=2)

du_dlat = ma.masked_invalid(du_dlat)
du_dlon = ma.masked_invalid(du_dlon)
dv_dlat = ma.masked_invalid(dv_dlat)
dv_dlon = ma.masked_invalid(dv_dlon)

# i commented more on this on streamplots.py, since the logic is the same
speed_threshold = np.percentile(speed.compressed(), 1)  # Lower 1% of speeds
critical_point_mask = speed <= speed_threshold

# Extract indices of critical points
critical_indices = np.where(critical_point_mask)


critical_points = []

for idx in zip(*critical_indices):
    lat_idx, lon_idx = idx

    # Extract the Jacobian matrix components
    J_elements = [
        du_dlon[lat_idx, lon_idx], du_dlat[lat_idx, lon_idx],
        dv_dlon[lat_idx, lon_idx], dv_dlat[lat_idx, lon_idx]
    ]

    # Check if any elements in J are masked or invalid
    if any(ma.is_masked(element) for element in J_elements):
        continue  # Skip this critical point
    if not np.all(np.isfinite(J_elements)):
        continue  # Skip this critical point

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
cp_type_counts = df_cp['type'].value_counts()
print("[DEBUG] Critical Point Counts by Type:")
print(cp_type_counts)

# [debug] save the results to csv
#df_cp.to_csv('critical_points_classification_results.csv', index=False)

############################################################################################################
# Visualization Pipeline Using Cartopy
############################################################################################################

# First Plot: Global Map of Vector Field of Surface Ocean Currents
# ----------------------------------------------------------------

# Define contour levels for the speed. clevs = contour levels
clevs = np.linspace(speed_min, speed_95th, 21)  


fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Orthographic(central_longitude=-20, central_latitude=20)})
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_global()

# filled contours of current speed
cf = ax.contourf(lon2d, lat2d, speed, levels=clevs, cmap='viridis', transform=ccrs.PlateCarree())

# subsample the data for quiver plot to avoid cluttering the map
# [Vector density] Increase this value to plot fewer vectors. I would initially change this to like 10 or even 15 just
# to check if the program will run properly for you, but if you change it to 3, it will look really cluttered initially.
# you will need to click on the magnifying glass on the plt plot and zoom in to a particular area to see the vectors like
# in the image I sent you.
step = 3   
lon_sub = lon2d[::step, ::step]
lat_sub = lat2d[::step, ::step]
u_sub = u_vals[::step, ::step]
v_sub = v_vals[::step, ::step]

# Plot the quiver plot of the currents (quiver plot is the vector field plot)
q = ax.quiver(lon_sub, lat_sub, u_sub, v_sub, scale=6, width=0.002, color='black', transform=ccrs.PlateCarree())
ax.quiverkey(q, 0.9, -0.1, 0.5, '0.5 m/s', labelpos='E')

cb = plt.colorbar(cf, orientation='horizontal', pad=0.05, shrink=0.7)
cb.set_label('Current Speed (m/s)')
plt.title('Surface Ocean Currents Vector Field')
plt.show()

###########################################################################################################
# Second Plot: Visualizing Critical Points and Vectors Around Them
###########################################################################################################

# Select critical points to plot. This is a subset of the total # of critical points so that we don't
# clutter the map with too many points. You can change this # 
num_critical_points_to_plot = 1000
df_cp_subset = df_cp.sample(n=num_critical_points_to_plot, random_state=42)


fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.Orthographic(central_longitude=-20, central_latitude=20)})

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='white')
ax.add_feature(cfeature.COASTLINE, linewidth=1.5)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
ax.set_global()

# filled contours of current speed
cf = ax.contourf(lon2d, lat2d, speed, levels=clevs, cmap='viridis', transform=ccrs.PlateCarree())


type_colors = {
    'Saddle Point': 'red',
    'Attracting Node (Sink)': 'blue',
    'Repelling Node (Source)': 'green',
    'Spiral Sink (Attracting Focus)': 'purple',
    'Spiral Source (Repelling Focus)': 'orange',
    'Center (Elliptic Point)': 'cyan',
    'Degenerate Node': 'yellow',
    'Unknown': 'black'
}

# iterate through critical points and plot them
for cp_type, group in df_cp_subset.groupby('type'):
    ax.scatter(
        group['longitude'], group['latitude'],
        color=type_colors.get(cp_type, 'black'),
        label=cp_type, s=10, transform=ccrs.PlateCarree()
    )


ax.legend(loc='lower left', fontsize='small')
cb = plt.colorbar(cf, orientation='horizontal', pad=0.05, shrink=0.7)
cb.set_label('Current Speed (m/s)')
plt.title('Critical Points on the Surface Ocean Currents')
plt.show()
