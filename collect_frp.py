import warnings
import zarr
import sys
sys.dont_write_bytecode = True
import numpy as np
import csv
import argparse
import logging
import pandas as pd
import xarray as xr
from collections import deque
from scipy.spatial import KDTree
from production_funcs import make_date_ranges,get_collective_frp,uniquify,aggragate_frp_temporal
from pyproj import CRS, Transformer # Transformers for projections 
import os

crs_aea_epsg = CRS.from_epsg(5070) #Albers eaqual area projection
transformer = Transformer.from_crs("EPSG:4326", crs_aea_epsg, always_xy=True) 

new_path = uniquify('station_with_frp.csv')
new_filename, extension = os.path.splitext(new_path)
logging.basicConfig(filename=new_filename+".log", level=logging.INFO)


def load_data(zarrPath,statPath):
    store = zarr.open_group(zarrPath, mode='r')
    available_fires = list(store.group_keys())
    if not available_fires:
        print("No fires found in the Zarr store!")
        raise ImportError("Could not find any fires to look at!")
    else:
        print("Fires successfully found!")
        logging.info("Fires successfully found!")
        logging.info(f"Fires available on disk: {available_fires}")
        print(f"Fires available on disk: {available_fires}")
        fire_id = available_fires[0] 
        # MUST use consolidated=False to read the appended stream correctly!
        weather_fire_ds = xr.open_zarr(zarrPath, group=fire_id , consolidated=False) # Dont load the whole array onto disk, keep lazy loaded
    print("Zarr file loaded and ready")
    logging.info("Zarr file loaded and ready")
    ################ Loading the station data ################
    station_ds = pd.read_csv(statPath) 
    # Within a timestep, a station is uniqued identified by the state code, county code, site num, and POC num
    date_hour = station_ds['Date Local'] + ' ' + station_ds['Time Local']
    aqis_id = station_ds['State Code'].astype('str') + ' ' + station_ds['County Code'].astype('str') + ' ' + station_ds['Site Num'].astype('str') + ' ' + station_ds['POC'].astype('str')
    station_ds['aqis_id'] = aqis_id
    station_ds['date_hour'] = date_hour 
    station_ds = station_ds.set_index('date_hour')
    print("Station data loaded and prepared")
    return station_ds, weather_fire_ds



# Example inputs for this program:
# start_date: str = '2025-01-07 20:00',
# end_date: str = '2025-01-07 20:00'
# python collect_frp.py --zarrPath "/home/exouser/labfetch/data/cali1mounthaug25.zarr" --statPath "/home/exouser/pmdata/hourly_88101_2025.csv" --distance 50000 --safe "False" --start "2025-08-07 20:00" --end "2025-08-29 20:00"

#python collect_frp.py --zarrPath "/home/exouser/labfetch/data/cali1mounthaug25.zarr" --statPath "/home/exouser/pmdata/hourly_88101_2025.csv" --distance 50000  --start "2025-08-01 00:00" --end "2025-08-31 23:00" --reachBack 40
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrPath", required=True)
    parser.add_argument("--statPath", required=True)
    parser.add_argument("--distance", required=True)
    parser.add_argument("--reachBack", required=True )
    parser.add_argument("--verbose", required=False)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    reach_back_length = int(args.reachBack)
    zarrPath = args.zarrPath
    statPath = args.statPath 
    radius_distance = int(args.distance)
    start_time = args.start 
    end_time = args.end
    verbose = args.verbose

    ################ Loading the zarr data ################
    station_ds, weather_fire_ds = load_data(zarrPath,statPath)
    ################ Prepare other variables ################
    date_r,dates_str = make_date_ranges(start_time,end_time)
    all_frps = []
    all_ind = []
    all_time = []
    all_raw_frps = []
    all_near_fires = []
    single_frps = []
    frp_cache = {}
    ################################ The main loop ################################
    for date_hour,date_hour_extended in zip(date_r,dates_str): #for each timestamp
        print(f"Processing time step: {date_hour}")
        logging.info(f"Beginning the processing of time step: {date_hour}")
        stations_date_hour = station_ds.loc[station_ds.index == date_hour] 
        weather_frp_region = weather_fire_ds.sel(indexers={'time':date_hour_extended}, method='nearest')
        fire_mask = weather_frp_region.where((weather_frp_region.rave_frp > 0).compute(), drop=True) # nans out values nans that dont have a rave_frp
        skip_fires = False 
        if np.isnan(fire_mask.rave_frp.values).all(): 
            print(f'No fires for this time step: {date_hour}, moving on')
            logging.info('No fires for this time step, moving on')
            skip_fires = True
        mask_x, mask_y = transformer.transform(fire_mask.lon.values.flatten() , fire_mask.lat.values.flatten() ) # Tansforms the fire locations
        masked_frp = fire_mask.rave_frp.values.flatten() 
        masked_u10 = fire_mask.u10.values.flatten()
        masked_v10 = fire_mask.v10.values.flatten()

        tree = KDTree(np.column_stack((mask_x, mask_y))) 

        for _, sta in stations_date_hour.iterrows(): 
            ind = sta["aqis_id"]
            frp = 0 # A default of 0 frp per station per timestep is needed to insure the agragation through time is accurate
            near_fires = 0
            logging.info(f"For time step {date_hour} considering station: {ind}")
            if ind not in frp_cache:
                frp_cache[ind] = deque(maxlen=reach_back_length)

            if skip_fires:
                frp_cache[ind].append(0)

            x_coord, y_coord = transformer.transform(sta['Longitude'], sta['Latitude']) # Transformers the Stations Lat Longs
            indices = tree.query_ball_point(
                [x_coord,y_coord],
                radius_distance
            )
            # index = tree.query(
            #     [x_coord,y_coord]
            # )
            raw_frp_sum = 0
            #single_frp = 0
            if indices: 
                frp_near = masked_frp[indices]
                #single_frp = masked_frp[index]
                u10_near = masked_u10[indices]
                v10_near = masked_v10[indices]
                lon_near = mask_x[indices]
                lat_near = mask_y[indices]
    
                valid = ~np.isnan(frp_near) & (frp_near > 0)
    
                frp_near = frp_near[valid]
                u10_near = u10_near[valid]
                v10_near = v10_near[valid]
                lon_near = lon_near[valid]
                lat_near = lat_near[valid]            
                near_fires = lat_near.shape[0]
                raw_frp_sum = frp_near.sum()
                safe = False
                if lat_near.size != 0:
                    frp = get_collective_frp(lon_near,lat_near,x_coord,y_coord,u10_near,v10_near,frp_near,safe)
                    
            
            logging.info(f"T: {date_hour} Stat: {ind}, {near_fires} nearby fires, frp: {frp}")
            frp_cache[ind].append(frp)

            frp = aggragate_frp_temporal(frp_cache[ind])
            all_near_fires.append(near_fires)
            all_raw_frps.append(raw_frp_sum) #sum of all frps in region NOT agragated
            #single_frps.append(single_frp) #the frp for the cell the station is in
            all_frps.append(frp)
            all_ind.append(ind)
            all_time.append(date_hour)
            
    print("Done collecting FRP's")
    logging.info("Done collecting FRP's")

    ############ Final appending of data ############
    results = pd.DataFrame({
        "aqis_id": all_ind,
        "date_hour": all_time,
        "num_fires_in_range": all_near_fires,
        "collective_frp": all_frps,
        "sum_raw_frps": all_raw_frps
    })
    print("dataframe made")
    results['aqis_id'] = results['aqis_id'].astype('str')
    print("type changed")
    df_combined = pd.merge(station_ds, results, on=['aqis_id', 'date_hour'], how='inner')
    print("df's merged")
    ########### Saving the file ###########
    
    df_combined.to_csv(new_path)
    print("File saved")
    logging.info("File saved")
    logging.info("In total, the following information was found and saved:")
    logging.info(f"For the time range: {start_time} to {end_time}")
    logging.info(f"{len(all_frps)} different collective frp values were found and saved")
    logging.info(f"The mean collective frp was: {np.array(all_frps).mean()}")
    logging.info(f"With a max value of: {np.array(all_frps).max()}")
    logging.info(f"And a min value of: {np.array(all_frps).min()}")
    logging.info(f"We also used a distance of {radius_distance} meters for how fair fires could be to be considered")
    logging.info("Process complete")

if __name__ == "__main__":
    main()



