import warnings
import zarr
import sys
sys.dont_write_bytecode = True
import numpy as np
import argparse
import logging
import pandas as pd
import xarray as xr
from production_funcs import make_date_ranges,get_collective_frp,get_close_data,uniquify
from pyproj import CRS, Transformer#these are the transformers for projections 
import os

crs_aea_epsg = CRS.from_epsg(5070) #albers eaqual area projection
transformer = Transformer.from_crs("EPSG:4326", crs_aea_epsg, always_xy=True) 

new_path = uniquify('station_with_frp.csv')#makes sure the file name we save to in the end is unqiue
new_filename, extension = os.path.splitext(new_path)
logging.basicConfig(filename=new_filename+".log", level=logging.INFO)

# Example inputs for this program:
# start_date: str = '2025-01-07 20:00',
# end_date: str = '2025-01-07 20:00'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarrPath", required=True)
    parser.add_argument("--statPath", required=True)
    parser.add_argument("--distance", required=True)
    parser.add_argument("--safe", required=False)
    parser.add_argument("--verbose", required=False)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    zarrPath = args.zarrPath
    statPath = args.statPath 
    radius_distance = int(args.distance)
    start_time = args.start 
    end_time = args.end
    verbose = args.verbose
    if args.safe=="False":
        safe = False
    else:
         safe = True 

    ################ Loading the zarr data ################
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
        fire_id = available_fires[3] 
        # MUST use consolidated=False to read the appended stream correctly!
        weather_fire_ds = xr.open_zarr(zarrPath, group=fire_id , consolidated=False) 
        #fire_name = weather_fire_ds.attrs.get('Fire_Name', '')
    weather_fire_ds.load() #turns the files from dask to array format
    print("Zarr file loaded and ready")
    logging.info("Zarr file loaded and ready")

    ################ Loading the station data ################
    df = pd.read_csv(statPath) 
    station_ds = df.to_xarray()

    ###### Prepare station data: ######
    # Adding new coords for indexing purposes
    date_hour = station_ds['Date Local'] + ' ' + station_ds['Time Local']
    station_ds = station_ds.assign_coords({'lat':station_ds['Latitude'],'long':station_ds['Longitude'],'date_hour':date_hour,})
    print("Station data loaded and prepared")
    ################ Prepare other variables ################
    date_r,dates_str = make_date_ranges(start_time,end_time)
    safe=False
    if safe == False:
            warnings.warn("This is the unsafe option where we will try and correct the angle when calcuating it, this is not recommended and should only be used for testing")
            logging.info("This is the unsafe option where we will try and correct the angle when calcuating it, this is not recommended and should only be used for testing")
    all_frps = []
    all_ind = []
    all_time = []
    ################################ The main loop ################################
    for date_hour,date_hour_extended in zip(date_r,dates_str): #for each timestamp
        print(f"Processing time step: {date_hour}")
        logging.info(f"Beginning the processing of time step: {date_hour}")
        stations_date_hour = station_ds.sel(indexers={'date_hour':date_hour}) #gets the station info for that hour
        weather_frp_region = weather_fire_ds.sel(indexers={'time':date_hour_extended},method='nearest')
        masked = weather_frp_region.where(weather_frp_region.rave_frp > 0, drop=True)#this nans out things that dont meet the conditions (ie only want valid fires)
        if np.isnan(masked.rave_frp.values).all(): #if no valid fires
            print(f'No fires for this time step: {date_hour}, moving on')
            logging.info('No fires for this time step, moving on')
            continue
        mask_lon = masked.lon #gets the all the lat lon values for the weather fire data
        mask_lat = masked.lat
        mask_frp = masked.rave_frp
        for i,thing in enumerate(stations_date_hour.index): #for each station by index(we can use the index to build other things)
            ind = thing.index.values #gets this stations index 
            logging.info(f"For time step {date_hour} considering station of index: {ind}")
            sta = stations_date_hour.sel(indexers={'index':ind}) #get that specifc station by its index
            #NOTE, what we pass in: 'weather_frp_region' is masked such that it should only have valid fires, however with how the 'drop=True' functionality works, what it
            # drops is rows and columns of the dataset that are all naned out, HOWEVER, if a row or column still has regular values, but a nan or two in there, those nans will stay.
            # The net result is that while a var in 'weather_frp_region' may have dimension 11 x 10, it doesnt have 110 non nan values. This explains why you need two checks later on.
            # One is needed to ensure all values are within the distance, and whether they are valid fires.
            fires_near, mask_x,mask_y,dist_mask,x_coord, y_coord,station_weather_near = get_close_data(sta,weather_frp_region,mask_lon,mask_lat,masked,dist_away=radius_distance)
            if dist_mask.sum() == 0: #then there is no fires within the distance and move on
                continue
            frp = fires_near.rave_frp.values.flatten() #now we pull out all the relavent data we need and clean it
            lat_rel = fires_near.lat.values.flatten()
            lon_rel = fires_near.lon.values.flatten()
            u10 = fires_near.u10.values.flatten()
            v10 = fires_near.v10.values.flatten()
            clean_frp = frp[~np.isnan(frp)] #this removes the nan values 
            clean_u10 = u10[~np.isnan(u10)]
            clean_v10 = v10[~np.isnan(v10)]
            lat_clean = lat_rel[~np.isnan(frp)]
            lon_clean = lon_rel[~np.isnan(frp)]
            x_relavent, y_relavent = transformer.transform(lon_clean, lat_clean)
            # Second sanity check: make sure that the fires_near outputed at least 1 valid fires
            if x_relavent.shape[0] == 0: 
                continue
            logging.info(f"Time: {date_hour} Station: {ind} has {x_relavent.shape[0]} nearby fires")
            frp = get_collective_frp(x_relavent,y_relavent,x_coord,y_coord,clean_u10,clean_v10,clean_frp,safe)
            all_frps.append(frp)
            all_ind.append(ind.item())
            all_time.append(date_hour)
    print("Done collecting FRP's")
    logging.info("Done collecting FRP's")
    ############ Final appending of data ############
    # we need to make the data into a dataset to merge them together
    frp_temp = xr.Dataset(
        data_vars={"collective_frps": (["index"], np.array(all_frps)),},
        coords={
            "index": ("index", all_ind),
            "date_hour": ("index", all_time),
        }
    )
    combined = xr.merge([station_ds, frp_temp])
    ########### Saving the file ###########
    df_combined = combined.to_dataframe()
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