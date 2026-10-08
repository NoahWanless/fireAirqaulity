import xarray as xr
import numpy as np
import pandas as pd
import argparse

# example newPath: "/home/exouser/fireAirqaulity/station_with_frp_avg.csv"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--statPath", required=True)
    parser.add_argument("--newPath", required=True)
    args = parser.parse_args()
    statPath = args.statPath
    newPath = args.newPath
    print("Loading in the Dataset")
    station_df = pd.read_csv(statPath)
    print("Processing the PM 2.5 values")
    grouped = station_df.groupby("aqis_id")
    for group_name, station in grouped:
        station_df.loc[station_df["aqis_id"] == group_name, "last_years_avg"] = station["Sample Measurement"].mean()

    station_df.to_csv(newPath)
    print("Finished!")

if __name__ == "__main__":
    main()