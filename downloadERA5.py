import cdsapi
c = cdsapi.Client()
# Define the ERA5 hourly request parameters
c.retrieve(
    #'reanalysis-era5-single-levels',
    'reanalysis-era5-single-levels-timeseries',
    {
        'product_type': 'reanalysis',
        'format': 'nc',                   # Options: 'grib', 'netcdf', or 'grib_zip'
        'variable': [
            'boundary_layer_height'       # 'total_precipitation',
        ],
        'year': ['2025'],
        'month': [
            '08'
        ],
        'day': [
            '01', '02', '03', '04', '05', '06',
            '07', '08', '09', '10', '11', '12',
            '13', '14', '15', '16', '17', '18',
            '19', '20', '21', '22', '23', '24',
            '25', '26', '27', '28', '29', '30', '31'
        ],
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'
        ],
        # Optional: Sub-select a geographical box [North, West, South, East]

        'area': [42.009518, -124.409591, 32.534156, -114.131211], 
    },
    'era5_hourly_boundary_aug.nc'           # Output filename
)