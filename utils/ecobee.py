import requests
import json
import datetime

def readJson(fileName):
    with open(fileName) as file:
        data = json.load(file)
    return data

def saveAsJson(fileName, data):
    with open(fileName, 'w') as file:
        json.dump(data, file)


def Dict2Str(dictionary):
    string = '{'
    for key, value in dictionary.items():
        string+=key
        string+=":"
        if type(value)==dict:
            string+=Dict2Str(value)
        else:
            string+=value
        string+=','
    return string[:-1]+"}"

class Ecobee():
    ## Refer to the documentation of ecobee API to generate the API key and refresh token for your thermostat.
    def __init__(self, file = 'utils/ecobee_info'):
        self.file = file
        self.info = readJson(self.file)
        self.url = self.info['url']
        self.apiKey = self.info['apiKey']
        self.valid_until = None
        self.access_token = None
        self.refresh_token = self.info['refresh_token']
        self.headers = None
        if self.access_token is None:
            self.refreshToken()
        
    def refreshToken(self):
        params = {"grant_type":"refresh_token", 
                  "code":self.refresh_token,
                  "client_id":self.apiKey}
        r = requests.post(self.url+"token", params=params) 
        data = r.json()
        
        if r.status_code == requests.codes.ok:
            self.access_token = data["access_token"]
            self.refresh_token = data["refresh_token"]
            self.headers = {'Authorization': 'Bearer '+self.access_token}
            
            self.info['refresh_token'] = self.refresh_token
            saveAsJson(self.file, self.info)
            
            self.valid_until = datetime.datetime.now() + datetime.timedelta(minutes = 45)
        else:
            r.raise_for_status()
             
    # Get Current Data;    
    def getData(self):
        if datetime.datetime.now() > self.valid_until:
            self.refreshToken()
        bodyDict = {'"selection"':
                    {'"selectionType"':'"registered"',
                     '"selectionMatch"':'""',
                     '"includeRuntime"':"true",
                     '"includeEquipmentStatus"':"true"
                     #'"includeSettings"':"true"
                    }}
        params = {'format' : 'json', 'body': Dict2Str(bodyDict)}
        
        r = requests.get(self.url+"1/thermostat", headers = self.headers, params = params)
        data = r.json()
        
        if r.status_code == requests.codes.ok:    
            #print(data)
            pass
        else:
            r.raise_for_status()
        return data['thermostatList'][0]
    
    ## Note: The time in the request is interpreted as UTC
    ## Return: In local time
    def getHistorical(self, start_time, end_time): #, variables
        if datetime.datetime.now() > self.valid_until:
            self.refreshToken()
        
        bodyDict = {
            '"startDate"':start_time.strftime("%Y-%m-%d"),
            '"endDate"':end_time.strftime("%Y-%m-%d"),
            '"columns"':'"hvacMode,compCool1,fan,outdoorTemp,zoneAveTemp,zoneCoolTemp,zoneOccupancy"',
            '"selection"':
            {'"selectionType"':'"thermostats"',
             '"selectionMatch"':'"521788610260"'}
        }
        params = {'format' : 'json', 'body': Dict2Str(bodyDict)}
        r = requests.get(self.url+"1/runtimeReport", headers = self.headers,  params = params)
        #print(r.url)
        data = r.json()
        if r.status_code == requests.codes.ok:    
            pass
        else:
            r.raise_for_status()
        return data
        
        
    # Set Setpoint
    def setHold(self, heatHoldTemp = 750, coolHoldTemp = 700):
        if datetime.datetime.now() > self.valid_until:
            self.refreshToken()
        dataDict = {
          "selection": {
            "selectionType":"registered",
            "selectionMatch":""
          },
          "functions": [
            {
              "type":"setHold",
              "params":{
                "holdType":"nextTransition",
                "heatHoldTemp":heatHoldTemp,
                "coolHoldTemp":coolHoldTemp
              }
            }
          ]
        }     
        r = requests.post(self.url+"1/thermostat?format=json", headers = self.headers, json = dataDict)
        if r.status_code == requests.codes.ok: 
            print("Success")
        else:
            r.raise_for_status()
