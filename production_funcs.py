import xarray as xr
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import warnings
from pyproj import CRS, Transformer#these are the transformers for projections 
crs_aea_epsg = CRS.from_epsg(5070) #albers eaqual area projection
transformer = Transformer.from_crs("EPSG:4326", crs_aea_epsg, always_xy=True) 
import os



#plots the data, assumes that there are the 7 variables passed in, bc it only displays those first 6
# data_x: the model input data
# data_y: the model target data
def plotPMvsData(data_x,data_y):
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))

    axs[0,0].scatter(data_x[:,1],data_y,s=.7)
    axs[0,0].set_xlabel("Variable 1: col_frp")
    axs[0,0].set_ylabel("PM2.5")

    axs[0,1].scatter(data_x[:,2],data_y,s=.7)
    axs[0,1].set_xlabel("Variable 1: u10")
    axs[0,1].set_ylabel("PM2.5")

    axs[0,2].scatter(data_x[:,3],data_y,s=.7)
    axs[0,2].set_xlabel("Variable 2: v10")
    axs[0,2].set_ylabel("PM2.5")

    axs[1,0].scatter(data_x[:,4],data_y,s=.7)
    axs[1,0].set_xlabel("Variable 3: t2m")
    axs[1,0].set_ylabel("PM2.5")

    axs[1,1].scatter(data_x[:,5],data_y,s=.7)
    axs[1,1].set_xlabel("Variable 4: sp")
    axs[1,1].set_ylabel("PM2.5")

    axs[1,2].scatter(data_x[:,6],data_y,s=.7)
    axs[1,2].set_xlabel("Variable 5: d2m")
    axs[1,2].set_ylabel("PM2.5")
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(data_x[:,2], data_x[:,3], data_y, c=data_y, cmap='viridis', marker='o')
    ax.set_xlabel('Wind component u10')
    ax.set_ylabel('Wind component v10')
    ax.set_zlabel('PM2.5')
    ax.set_title('PM2.5 vs Wind')
    plt.show()






#This makes sure that files names are unique as we save them:
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


# Gets a series of date ranges in the two formats I need in the required, although strange, formats
def make_date_ranges(start_date='2025-01-07 20:00',end_date='2025-01-07 20:00'):
    dates = pd.date_range(start=start_date, end=end_date,freq='h') #this is the date range we are interested in
    date_r = dates.strftime('%Y-%m-%d %H:%M')
    dates_str = dates.astype(str)
    dates_str = dates_str.str.replace(' ','T')
    dates_str =dates_str.str.cat([".000000000"]*len(dates_str))
    return date_r,dates_str

# Gets a data range of both formats that i need
# but this time, makes two versions, for for training, and one to compare to because we are trying to predict a hour ahead after all
# so its like this:
# 123456
#  123456 (with the first 1, and last 6 being removed and only using the overlaping sections)
# this can be done with any offset we give it, defaults to 1
def training_date_ranges(start_date='2025-01-07 20:00',end_date='2025-01-07 20:00',hour_offset=1):
    dates = pd.date_range(start=start_date, end=end_date,freq='h') #this is the date range we are interested in
    dates_compare=dates[hour_offset:len(dates)]
    dates_train=dates[0:len(dates)-hour_offset]

    #makes the date ranges for compariasion
    date_r_compare = dates_compare.strftime('%Y-%m-%d %H:%M')
    dates_str_compare = dates_compare.astype(str)
    dates_str_compare = dates_str_compare.str.replace(' ','T')
    dates_str_compare =dates_str_compare.str.cat([".000000000"]*len(dates_str_compare))

    #makes the date ranges for training
    date_r_train = dates_train.strftime('%Y-%m-%d %H:%M')
    dates_str_train = dates_train.astype(str)
    dates_str_train = dates_str_train.str.replace(' ','T')
    dates_str_train =dates_str_train.str.cat([".000000000"]*len(dates_str_train))
    return date_r_compare,dates_str_compare,date_r_train,dates_str_train


############ get_collective_frp ############
# x_coord,y_coord: the projected x,y coordinate of the station
# x_relavent,y_relavent: the 2 LISTS, of projected x coords, and projected y coords for each fire
# clean_frp: the list of frp values, a list in parellel with x_relavent,y_relavent
# clean_u10,clean_v10: the 2 LISTS, of u and v components of wind vectors at the fires, a list in parellel with x_relavent,y_relavent
# safe: the safety device that stops the evaluation of this frp value if a invalid angle is found, turning this on should only be used for testing
############ Returns: ############
# the collective frp for the sation within the range
def get_collective_frp(x_relavent,y_relavent,x_coord,y_coord,clean_u10,clean_v10,clean_frp,safe=True):
    fire_wind = np.array([clean_u10,clean_v10]).T #TODO: CHECK IF THIS IS THE CORRECT ORDER
    fires_xy = np.array([x_relavent,y_relavent]) #array of fire positions (row 0 is the x coords, row 1 is the y coords)
    station_temp = np.tile([x_coord,y_coord], (fires_xy.shape[1], 1)).T
    
    fires_to_station = (station_temp - fires_xy).T#these are the station to fire vectors (tranpose for convenice later)
    #this order ensures that fire vec + distance vec points to the station(what we want)
    ############### Now we will calcuate the wind angle to direction to the station for each fire ###############
    wind_dist_dot = np.einsum('ij,ij->i', fires_to_station, fire_wind)#this calcuates the dot product for each pair of vectors across the two lists (ie for each fire)
    wind_norm = np.linalg.norm(fire_wind,axis=1) #norm of wind vectors 
    dist_norm = np.linalg.norm(fires_to_station,axis=1) #norm of the distance from fire to 
    wind_dist_prod = np.multiply(wind_norm,dist_norm) #element wise multiplication o fthe norms
    interal_value = np.divide(wind_dist_dot,wind_dist_prod) 
    #sanity check 
    if interal_value.max() > 1 or interal_value.min() < -1:
        #TODO: implelment some system such that this fire is ignored and the system can contine that is better then the current version
        #TODO: implelment some system such that this fire is ignored and the system can contine that is better then the current version
        if safe:
            raise ValueError(f"When calcuating the angle a value was found that would give a invalid angle when run through arccos, mainly a value greater then 1 or less then -1. min:{interal_value.min()}, max:{interal_value.max()}")
        else: #this is the unsafe option, where we will try and correct the angle. This is not recommended but is a work around for testing reasons
            toobig_vecs = np.where(interal_value > 1)
            toosmall_vecs = np.where(interal_value < -1)
            interal_value[toobig_vecs] = 1
            interal_value[toosmall_vecs] = -1
    #* there is a chance something in here is -1, or 1, if the vectors are in perfect alignment or agasint
    inline_vecs = np.where(interal_value == 1)
    opposite_vecs = np.where(interal_value == -1)
    angles = np.arccos(interal_value) 
    
    angles[inline_vecs] == 0
    angles[opposite_vecs] == 2*np.pi #manuely update those values to what they should be
    #^ BIG ASSUMPTION RIGHT HERE, WE ARE ASSUMING THAT THE LENGTH OF THE WIND VECTORS CORRESPONDS TO THE SPEED OF THE WIND ^#
    
    step1 = np.multiply(np.absolute(wind_norm),(np.cos(np.absolute(angles))+1)) #this is the forumla itself
    step2 = np.multiply(clean_frp,step1)
    step3 = 4*np.pi*((dist_norm/1000)**2)
    frps = np.divide(step2,step3)
    collective_frp_value = frps.sum()
    return collective_frp_value


########### get_close_data ###########
# sta: the station itself we are using
# weather_frp_region: the weather and frp, we use this to get the cell closest to the station
# mask_lon: the lon values of the fire cells (these are still xarray datasets form )
# mask_lat: the lon values of the fire cells
# masked: the masked weather and frp data to being only fires
# dist_away: how far away fires can be to be considered
########### Returns: ###########
# fires_near: the dataset that was masked to be just the fires in range
# mask_x, mask_y: the transformed x,y coords of the fires
# dist_mask: the mask of fires that are within
# x_coord, y_coord: the x,y coords (projected) of the station
# station_weather_near: the weather cell nearest the station
def get_close_data(sta,weather_frp_region,mask_lon,mask_lat,masked,dist_away=10000000):
    ###### find the closest weather cell of the station
    ##### This uses base euclidean distance but because the distances will be rather small itll be fine i think #####
    #####TODO: change this to use the porjected coords you calcuated eariler :TODO#####
    #####TODO: change this to use the porjected coords you calcuated eariler :TODO#####
    dist = (weather_frp_region.lat - sta.lat.values)**2 + (weather_frp_region.lon - sta.long.values)**2
    min_idx = dist.argmin(dim=["y", "x"]) # Find the (y, x) index of the minimum distance 
    #this gets the weather info for the cell closest to the station
    station_weather_near = weather_frp_region.isel(y=min_idx["y"].values, x=min_idx["x"].values)
    
    ########## now we will find all the fires within the region of the station ##########
    ########## this uses the alber equal area projection formula ##########
    x_coord, y_coord = transformer.transform(sta.long.values, sta.lat.values) #this is the station lat longs now transformed into 2d coords 
    mask_x,mask_y = transformer.transform(mask_lon, mask_lat)
    x_diff = mask_x - x_coord #gets the difference between each point 
    y_diff = mask_y - y_coord
    diff = np.array([x_diff,y_diff])
    dist = np.linalg.norm(diff,axis=0) #now calcuates the distance between each point (now of shape 14x43 or the orginal shape of mask_lon and etc)
    dist_mask = dist < dist_away #makes a mask of all points that are within the needed distance dist_away
    dist_mask_np = xr.DataArray(dist_mask,dims=["y", "x"]) #makes the mask a xarray
    fires_near = masked.where(dist_mask_np) 
    
    return fires_near, mask_x,mask_y,dist_mask,x_coord, y_coord,station_weather_near