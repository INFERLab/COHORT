import forecastio
import datetime
import csv
import time
import pandas as pd

# Temps is a list of dictionary
def MakeSeries(Temps):
    temp = []
    idx = []
    for dct in Temps:
        # Adjust for Time Zone
        idx.append(dct['Time']-datetime.timedelta(hours = 4))
        temp.append(dct['Temp'])
    # pad one more hour
    idx.append(idx[-1]+datetime.timedelta(hours = 1))
    temp.append(dct['Temp'])
    return pd.Series(temp, index = idx, name = "T_out")

def getWeather(start_time = None, duration = 2):
    '''link to the Forecast.io API: https://developer.forecast.io/
        Forecast.io Python Wrapper: https://github.com/ZeevG/python-forecast.io'''
    api_key = "c0764c24b7fba4308aac35fc72c36110"

    '''Locations for weather forecast
        If this list is updated, remember to update the dict keys as well
    '''

    locations = ['Pittsburgh']
    lats = [40.442899,]
    lngs = [ -79.942850]

    ''' Location Key
    Pittsburgh
    '''

    ''' Get the current time to add to the forecast record '''
    now = datetime.datetime.now()
    ForecastTime = now.strftime("%Y-%m-%d %H:%M")

    ''' Set the start date to either a specific day or a specific lag '''
    if start_time is None:
        a = datetime.datetime.today()
    else:
        a = start_time
    numdays = duration# Set the number of days to pull in the future

    '''Set the Date array'''
    dateList = []
    for x in range (0, numdays):
        dateList.append(a + datetime.timedelta(days = x)) # Can change the '+' to '-' to pull historical
    
    '''Loop through the locations and dates'''
    temps = []

    for i in range (len(locations)):
        for dates in dateList:
            forecast = forecastio.load_forecast(api_key, lats[i], lngs[i], dates)

            byHour = forecast.hourly()

            for hourlyData in byHour.data:
                try:
                    temps.append({
                            'Time':hourlyData.time,
                            'Temp':hourlyData.temperature,
                            })
                except:
                    pass
    ts = MakeSeries(temps)
    return ts.resample("15T").interpolate(method = "linear")

