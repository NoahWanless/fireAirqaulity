import xarray as xr
import numpy as np
import argparse
import pandas as pd
import math
import zarr
import logging
import sys
from production_funcs import training_date_ranges,uniquify
from pyproj import CRS, Transformer

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
python prepare_training_data.py --zarrPath "/media/volume/firebackup/med_test/20250801_0000-20250831_2300_WF.zarr" --statPath "/home/exouser/code/labfetch/station_with_frp_2.csv" --start '2025-08-01 02:00' --end '2025-08-31 21:00' --offset 1
python prepare_training_data.py --zarrPath "/media/volume/firebackup/med_test/20250801_0000-20250831_2300_WF.zarr" --statPath "/home/exouser/code/labfetch/station_with_frp_2.csv" --start '2025-08-01 02:00' --end '2025-08-31 21:00' --offset 1
'''
# start_date: str = '2025-01-07 20:00',
# end_date: str = '2025-01-07 20:00'
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

    ############### load in the station data ###############
    df = pd.read_csv(statPath) 
    station = df.to_xarray()
    date_hour = station['Date Local'] + ' ' + station['Time Local']
    new_id = station['County Code'].astype(str) + ' ' + station['Site Num'].astype(str) + ' ' + station['State Code'].astype(str) + ' ' + station['POC'].astype(str)
    station = station.assign_coords({'lat':station['Latitude'],'long':station['Longitude'],'date_hour':date_hour,'new_id':new_id})

    ############### load in the zarr data ###############
    store = zarr.open_group(zarrPath, mode='r')
    available_fires = list(store.group_keys())

    if not available_fires:
        print("No fires found in the Zarr store!")
        exit
    else:
        print(f"Fires available on disk: {available_fires}")
        logging.info(f"Fires available on disk: {available_fires}")
        #fire_id = available_fires[3]    

    ## If we only want one fire group then we set it accordingly ##
    if fire_group != None:
        available_fires = available_fires[fire_group]

    Y_temp = []
    X_temp = []
    
    for fire_id in available_fires: #for each fire group 
        # MUST use consolidated=False to read the appended stream correctly!
        ds = xr.open_zarr(zarrPath, group=fire_id , consolidated=False)  # group=fire_id 
        fire_name = ds.attrs.get('Fire_Name', '')
        ds.load()

        ############### prepare dates for gathering the data ###############
        date_r_compare,dates_str_compare,date_r_train,dates_str_train = training_date_ranges(start_date=start_time,end_date=end_time,hour_offset=offset)

        ############### get only stations with frp values to use for training ###############
        station_only_frps = station.dropna(dim="index", subset=['collective_frps'])
        skip_nans = True #this means as we iterate through all the values, we skip ones that nan, or have no collective frp value

        data_len = station_only_frps.collective_frps.values.shape[0]
        
        
        ############### Main loop ###############
        for r_compare,str_compare,r_train,str_train in zip(date_r_compare,dates_str_compare,date_r_train,dates_str_train): #for each timestamp
            try:
                stations_current = station_only_frps.sel(indexers={'date_hour':r_train}) #gets the current station info
                stations_future = station.sel(indexers={'date_hour':r_compare}) #gets the future station info
            except KeyError:
                continue

            weather_current = ds.sel(indexers={'time':str_train},method='nearest') #gets the current weather info
            
            if stations_current['new_id'].ndim > 0: #if the time series in question has more then one entry
                iterator = enumerate(stations_current['new_id'])    
                for i,stat in iterator: #for each station by index of the current timestep
                    site_id = stat.values
                    try:
                        stat_current = stations_current.sel(indexers={'new_id':site_id}) #get that specifc station by its index (current)
                        stat_future = stations_future.sel(indexers={'new_id':site_id}) #(future)
                    except KeyError:
                        print("There was some sort of error when loading in these sites, skipping to the next one")
                        continue
                    col_frp = stat_current.collective_frps.values.flatten().item() #collective frp
                    if math.isnan(col_frp): #skip if the frp collected is nan 
                        col_frp = 0
                        if skip_nans == True:
                            continue
                    
                    dist = (weather_current.lat - stat_current.lat.values)**2 + (weather_current.lon - stat_current.long.values)**2
                    min_idx = dist.argmin(dim=["y", "x"]) # Find the (y, x) index of the minimum distance  
                    #this gets the weather info for the cell closest to the station
                    weather_near_station = weather_current.isel(y=min_idx["y"].values, x=min_idx["x"].values)
                    
                    cur_pm = stat_current['Sample Measurement'].values.flatten().item() #gets the pm2.5 values, both current and future
                    frps = stat_current.collective_frps.values.flatten().item()
                    fut_pm = stat_future['Sample Measurement'].values.flatten().item()
                    u10 = weather_near_station.u10.values.flatten().item() #wind component
                    v10 = weather_near_station.v10.values.flatten().item()
                    t2m = weather_near_station.t2m.values.flatten().item() #2 meter temp
                    sp = weather_near_station.sp.values.flatten().item() #surface pressure
                    d2m = weather_near_station.d2m.values.flatten().item() #dew point temp

                    x = np.array([cur_pm,frps,u10,v10,t2m,sp,d2m])
                    y = fut_pm
                    Y_temp.append(y)
                    
                    X_temp.append(x)
                    
            else:
                print('found a single station')
                stat_current = stations_current
                
                site_id = stations_current.new_id.values
                stat_future = stations_future.sel(indexers={'new_id':site_id})
                col_frp = stat_current.collective_frps.values.flatten().item() #collective frp
                if math.isnan(col_frp): #skip if the frp collected is nan 
                    col_frp = 0
                    if skip_nans == True:
                        continue
                
                dist = (weather_current.lat - stat_current.lat.values)**2 + (weather_current.lon - stat_current.long.values)**2
                min_idx = dist.argmin(dim=["y", "x"]) # Find the (y, x) index of the minimum distance  
                    #this gets the weather info for the cell closest to the station
                weather_near_station = weather_current.isel(y=min_idx["y"].values, x=min_idx["x"].values)
                    
                cur_pm = stat_current['Sample Measurement'].values.flatten().item() #gets the pm2.5 values, both current and future    
                frps = stat_current.collective_frps.values.flatten().item()
                fut_pm = stat_future['Sample Measurement'].values.flatten().item()
                u10 = weather_near_station.u10.values.flatten().item() #wind component
                v10 = weather_near_station.v10.values.flatten().item()
                t2m = weather_near_station.t2m.values.flatten().item() #2 meter temp
                sp = weather_near_station.sp.values.flatten().item() #surface pressure
                d2m = weather_near_station.d2m.values.flatten().item() #dew point temp
                
                x = np.array([cur_pm,frps,u10,v10,t2m,sp,d2m])
                y = fut_pm
                
                Y_temp.append(y)
                
                X_temp.append(x)
                
                
                    
    Y = np.array(Y_temp)
    X = np.array(X_temp)
    print(X)
    print(Y)
    print(X.shape)
    print(Y.shape)
    np.savetxt(X_train_path, X, delimiter=",")
    np.savetxt(y_train_path, Y, delimiter=",")
    print("Finished looping over the entries")
    logging.info("Now that we have finished processing the data, we shall give you some basic statistics on the training set")
    logging.info(f"Target pm2.5 mean:{Y.mean()}")
    logging.info(f"Target pm2.5 max:{Y.max()}")
    logging.info(f"Target pm2.5 min:{Y.min()}")
    logging.info(f"Target pm2.5 std:{Y.std()}")
    

        

if __name__ == "__main__":
    main()