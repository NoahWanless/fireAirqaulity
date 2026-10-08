import xarray as xr
import numpy as np
import argparse
import pandas as pd
import math
import zarr
import logging
import sys
from production_funcs import training_date_ranges,uniquify,merge_in_era5
from pyproj import CRS, Transformer
from scipy.spatial import KDTree

sys.dont_write_bytecode = True
crs_aea_epsg = CRS.from_epsg(5070) #albers eaqual area projection
transformer = Transformer.from_crs("EPSG:4326", crs_aea_epsg, always_xy=True) 
#makes sure the file name we save to in the end is unqiue
X_train_path = uniquify('training_data/X_train.csv')
y_train_path = uniquify('training_data/y_train.csv')
logging_path = uniquify("training_data/training_data.log")
logging.basicConfig(filename=logging_path, level=logging.INFO)
# Example inputs for this program:
'''
python prepare_training_data.py --zarrPath "/home/exouser/labfetch/data/cali1mounthaug25.zarr" --statPath "/home/exouser/fireAirqaulity/station_with_frp.csv" --start '2025-08-03 02:00' --end '2025-08-20 21:00' --offset 1
'''

def main():
    # gets all the arguements we need
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrPath", required=True) #path to the zarr file with the weather data
    parser.add_argument("--statPath", required=True) #path to the station data with the collective frp and pm2.5 data
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--offset", required=True)
    args = parser.parse_args()
    zarrPath = args.zarrPath
    statPath = args.statPath 
    start_time = args.start 
    end_time = args.end
    offset = int(args.offset)
    fire_group = None
    logging.info(f"Zarr path used: {zarrPath}")
    logging.info(f"Station path used: {statPath}")
    logging.info(f"Start time: {start_time}")
    logging.info(f"End time: {end_time}")
    logging.info(f"Offset between between when we are predicting and what info we have: {offset}")
    weather_variables = [
        'u10',
        'v10',
        'tp',
        't2m',
        'sp',
        'd2m'
    ]
    ############### load in the station data ###############
    station_ds = pd.read_csv(statPath) 
    date_hour = station_ds['Date Local'] + ' ' + station_ds['Time Local']
    aqis_id = station_ds['State Code'].astype('str') + ' ' + station_ds['County Code'].astype('str') + ' ' + station_ds['Site Num'].astype('str') + ' ' + station_ds['POC'].astype('str')
    station_ds['aqis_id'] = aqis_id
    station_ds['date_hour'] = date_hour 
    station_ds = station_ds.set_index('date_hour')
    ############### load in the zarr data ###############
    store = zarr.open_group(zarrPath, mode='r')
    available_fires = list(store.group_keys())
    if not available_fires:
        print("No fires found in the Zarr store!")
        exit
    else:
        print(f"Fires available on disk: {available_fires}")
        logging.info(f"Fires available on disk: {available_fires}")
        # MUST use consolidated=False to read the appended stream correctly!
        ds = xr.open_zarr(zarrPath, group=available_fires[0], consolidated=False)
        era5DS = xr.open_dataset(
            "/home/exouser/fireAirqaulity/era5_hourly_precipitation_aug.nc",
            engine="netcdf4" #netcdf4, cfgrib
        )
        #For some reason the total percipatation in aug is really small
        ds = merge_in_era5(ds,era5DS)

        ds.load() 
        
    final_data = []

    ############### prepare dates for gathering the data ###############
    date_r_compare,dates_str_compare,date_r_train,dates_str_train = training_date_ranges(start_date=start_time,end_date=end_time,hour_offset=offset)
    ############### get only stations with frp values to use for training ###############
    station_only_frps = station_ds.dropna(subset=["collective_frp"])
    lat = ds.lat.values
    lon = ds.lon.values
    points = np.column_stack([
        lat.ravel(),
        lon.ravel()
    ])

    station_lat = station_only_frps["Latitude"].values
    station_lon = station_only_frps["Longitude"].values % 360

    station_points = np.column_stack([
        station_lat,
        station_lon
    ])

    tree = KDTree(points)
    _, flat_idx = tree.query(station_points)
    y_idx, x_idx = np.unravel_index(flat_idx, lat.shape) 
    station_only_frps["y_idx"] = y_idx
    station_only_frps["x_idx"] = x_idx  
    

    ############### Main loop ###############
    for r_compare,str_compare,r_train,str_train in zip(date_r_compare,dates_str_compare,date_r_train,dates_str_train): #for each timestamp
        print(f"Processing time step: {r_train}")

        stations_current = station_only_frps.loc[station_only_frps.index == r_train] #gets the station info for that hour
        stations_future = station_only_frps.loc[station_only_frps.index == r_compare] #gets the station info for the future hour
        weather_current = ds.sel(indexers={'time':str_train},method='nearest') #gets the current weather info
        
        x_idxs = np.union1d(stations_current['x_idx'].values,stations_future['x_idx'].values)
        y_idxs = np.union1d(stations_current['y_idx'],stations_future['y_idx'].values)
        # De trending/normalization of the pm2.5
        stations_current["Sample Measurement"] = stations_current["Sample Measurement"] - stations_current["last_years_avg"]
        stations_future["Sample Measurement"] = stations_future["Sample Measurement"] - stations_future["last_years_avg"]

        station_ids = xr.DataArray(
            stations_current["aqis_id"].to_numpy(),
            dims="station",
            name="station"
        )

        station_ids_future = xr.DataArray(
            stations_future["aqis_id"].to_numpy(),
            dims="station",
            name="station"
        )

        x_idxs = xr.DataArray(
            stations_current["x_idx"].values,
            dims="station",
            coords={"station": station_ids}
        )

        y_idxs = xr.DataArray(
            stations_current["y_idx"].values,
            dims="station",
            coords={"station": station_ids}
        )

        data = weather_current.isel(
            y=y_idxs,
            x=x_idxs,
        )[weather_variables]

        cur_pm25 = xr.DataArray(
            stations_current["Sample Measurement"].to_numpy(),
            dims="station",
            coords={"station": station_ids}
        )

        fut_pm25 = xr.DataArray(
            stations_future["Sample Measurement"].to_numpy(),
            dims="station",
            coords={"station":station_ids_future}
        )

        frp = xr.DataArray(
            stations_current["collective_frp"].to_numpy(),
            dims="station",
            coords={"station": station_ids}
        )

        near_fires = xr.DataArray(
            stations_current["num_fires_in_range"].to_numpy(),
            dims="station",
            coords={"station": station_ids}
        )
    
        data["frp"] = frp
        data["cur_pm25"] = cur_pm25
        data["fut_pm25"] = fut_pm25
        data["near_fires"] = near_fires
        data["station_ids_current"] = station_ids
        
        
        column_data = np.column_stack([
            data[var].to_numpy().ravel() for var in data.data_vars
        ])

        
        final_data.append(column_data)

            #! add in the elevation
            #! add the number of fires within the range as its own datapoint
            #! era5 dataset total precipitation, to get that dataset, make a quiick loader for it to get it in
            #! classification for high threshold instead of the other thing
            #! also double check the collect_frp has the correct longitude thing

    final_data = np.concatenate(final_data, axis=0)    
    np.savetxt(X_train_path, final_data, delimiter=",", fmt="%s")
    print("Finished looping over the entries")
    # logging.info("Now that we have finished processing the data, we shall give you some basic statistics on the training set")
    # logging.info(f"Target pm2.5 mean:{final_data[:,5].mean()}")
    # logging.info(f"Target pm2.5 max:{final_data[:,5].max()}")
    # logging.info(f"Target pm2.5 min:{final_data[:,5].min()}")
    # logging.info(f"Target pm2.5 std:{final_data[:,5].std()}")
    



if __name__ == "__main__":
    main()