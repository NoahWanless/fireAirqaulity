import xarray as xr
import matplotlib.pyplot as plt
import warnings
import numpy as np
import pandas as pd
import xesmf as xe
import warnings
import math
from pyproj import CRS, Transformer#these are the transformers for projections 
crs_aea_epsg = CRS.from_epsg(5070) #albers eaqual area projection
transformer = Transformer.from_crs("EPSG:4326", crs_aea_epsg, always_xy=True) 
import os



def merge_in_era5(hrrrDS,era5DS):
    era5DS = era5DS.reset_coords("number",drop=True).reset_coords("expver",drop=True).rename({"valid_time":"time"})
    
    
    regridder = xe.Regridder(
        era5DS,
        hrrrDS,
        "bilinear"
    )
    era5DS_on_hrrr = regridder(era5DS)
    hrrrDS = hrrrDS.merge(era5DS_on_hrrr)
    
    return hrrrDS



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

############ reach_back ############
# date: The current date we are considering
# dates: The list of all dates at our disposal
# reach_back: How many hours back we want (but may not be able) to reach to get
############ Returns: ############
# a list of the dates we got (including the one we passed in)
def reach_back(date,dates,reach_back=0):
    if isinstance(dates, pd.Index) != True:
        raise ValueError("Dates list must be a pandas index object")
    if reach_back == 0:
        return date
    else:
        index = dates.get_loc(date)
        starting_index = 0 if reach_back > index else index - reach_back
        return dates[starting_index:index+1]

#this is a sin function with y intercept 1 and a period manipluated such that the first time it crosses the x axies is slightly after the time_factor.
# this makes it so time values closer to the current time have more weight
#! IN FUTURE MAKE THIS A LOG FUNCTION, exponetial decreasing no sin
def frp_time_scale(time_factor,reach_back):
    return -math.sin(.25*(2*math.pi*time_factor)/(reach_back+.5) - math.pi/2)

#works under the assumtion that the first element is the most recent
#this could be expanded to be a better function
def aggragate_frp_temporal(frps):
    total = 0
    for index, frp in enumerate(frps):
        total = total + frp * frp_time_scale(index,len(frps))
    return total


############ get_collective_frp ############
# x_coord,y_coord: the projected x,y coordinate of the station
# x_relavent,y_relavent: the 2 LISTS, of projected x coords, and projected y coords for each fire
# clean_frp: the list of frp values, a list in parellel with x_relavent,y_relavent
# clean_u10,clean_v10: the 2 LISTS, of u and v components of wind vectors at the fires, a list in parellel with x_relavent,y_relavent
# safe: the safety device that stops the evaluation of this frp value if a invalid angle is found, turning this on should only be used for testing
############ Returns: ############
# the collective frp for the sation within the range
def get_collective_frp(x_relavent,y_relavent,x_coord,y_coord,clean_u10,clean_v10,clean_frp,safe=True):
    fire_wind = np.array([clean_u10,clean_v10]).T 
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
