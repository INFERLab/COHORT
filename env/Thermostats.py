import numpy as np
import pandas as pd
import pickle
import datetime, time

import os
import sys
main_path = os.path.abspath(os.path.join(__file__, '..', '..'))
sys.path.insert(0, main_path)
from utils import weather, pjm, ecobee

def roundTime(time):
    return datetime.datetime(year = time.year, month = time.month, day = time.day, hour = time.hour, minute = (time.minute//15)*15)

## Simulation environment for virtual thermostats
class Thermostat():
    def __init__(self, params, cur_time, timestep = 15, exog_var = ["T_out", "Occupied_lp", "Sleep"]):
        ## Index in the Metadata file
        self.building_idx = params["idx"]
        
        self.params = params
        self.p = len(self.params['a'])
        self.m = len(self.params['bu']) # how many u_prev to consider
        self.n_dist = len(self.params['bd']) # Other exog variables
        self.timestep = timestep
        self.Pm = self.params['Pm']
        
        ## Current states
        self.cur_time = cur_time
        self.x = None
        self.u = 0
        
        ## Save previous states and actions
        self.idx_list = []
        self.x_list = []
        self.u_list = []
        
        ## Disturbances
        self.disturbance = None
        self.sp = 75
        self.exog_var = exog_var
        
        ## deadband
        self.Delta = 1.8

        ## Only used for determining what value to pad when len(x_prev)<p
        self.x_init = None
        
    def reset(self, x_init = None, u_init = None):
        if x_init is not None:
            self.x = x_init
        else:
            self.x =  75 # self.sp.loc[self.cur_time]
        if u_init is not None:
            self.u = u_init
        else:
            self.u = 0
        
        self.idx_list.append(self.cur_time)
        self.x_list.append(self.x)
        self.u_list.append(self.u)
        
        self.x_init = 75 #self.sp.loc[self.cur_time]
        return self.x, self.u
        
    def step(self, u, normalized = False, k = -1):
        if normalized:
            u *= self.Pm
        x_prev = self.x_list[-self.p:]
        self.u_list.append(u)
        u_prev = self.u_list[-self.m:]
        d = self.disturbance[self.exog_var].loc[self.cur_time].values
        
        self.x = self.calculate(np.array(x_prev), np.array(u_prev), d)
        
        self.x_list.append(self.x)
        self.cur_time += datetime.timedelta(minutes = self.timestep)
        self.idx_list.append(self.cur_time)
        return self.x, u
    
    def calculate(self, x_prev, u_prev, d):
        # Note: x and u in [x_{t-T+1}, ..., x_t]
        if len(x_prev) < self.p:
            x_prev = np.pad(x_prev, (self.p-len(x_prev), 0), mode = 'constant', constant_values = self.x_init)
        if len(u_prev) < self.m:
            u_prev = np.pad(u_prev, (self.m-len(u_prev), 0), mode = 'constant')
            
        return self.params['a'].dot(np.flip(x_prev)) + self.params['bu'].dot(np.flip(u_prev)) / self.Pm + self.params['bd'].dot(d) + self.params['c']
    
    
    def pullDist(self, start_time, end_time):
        x_prev = self.x_list[-self.p:]
        if len(x_prev) < self.p:
            x_prev = np.pad(x_prev, (self.p-len(x_prev), 0), mode = 'constant', constant_values = self.x_init)
        x_lower = (self.sp.loc[start_time:end_time] - self.Delta).clip(None, 72)
        x_upper = self.sp.loc[start_time:end_time] + self.Delta
        
        dist = self.disturbance[self.exog_var].loc[start_time:end_time]
        d = dist.dot(self.params['bd']) + self.params['c']
        if len(self.u_list)>0:
            d.iloc[0] += self.params['bu'][1] * self.u_list[-1]
        return x_prev, d.values, x_lower.values, x_upper.values
        
    def updateDist(self, weather, filePath = "/Users/chenbq/Desktop/CMU/Research 101/Grid/BuildSys'20/ecobee"):
        occupancy = pd.read_pickle("{}/exog/exog_{}".format(filePath, self.building_idx))
        self.disturbance = pd.concat([weather.to_frame(), occupancy], axis = 1, join = 'inner')
        self.sp = self.disturbance["T_stp_cool"]

    
class myApartment(Thermostat):
    def __init__(self, params, cur_time, limit = 4.5):
       super().__init__(params, cur_time, exog_var = ["T_out", "Home", "Sleep"])
       self.ecobee = ecobee.Ecobee()
       
       ## For PWM
       self.cum_error = 0
       self.limit = limit
       self.dt = 1 # check every minute
       
       self.onoff = 0 #{0, 1}
       
       ## For saving,
       self.u_record = []
       self.command_record = []
       self.idx_record = []
       
    ## Note: Inherited Methods:  pullDist
    #        Method Modified: reset, step, updateDist,
    #        New Method: PWM
    #        To be implemented: updateModel
    #        Method Not Used: calculate
    
    def reset(self, x_init = None, u_init = None):
        obs = self.ecobee.getData()
        self.x = obs['runtime']['actualTemperature'] * 0.1
        
        if 'compCool1' in obs["equipmentStatus"].split(","):
            self.onoff = 1
        else:
            self.onoff = 0
            
        self.u = self.onoff*self.Pm
        
        self.idx_list.append(self.cur_time)
        self.x_list.append(self.x)
        self.u_list.append(self.u)
        
        self.x_init = 75 #self.sp.loc[self.cur_time]
        return self.x, self.u
    
    def PWM(self, signal):
        ## Delta-Sigma Method
        # Make decision based on predicted error!
        pred_error = self.cum_error + (signal - self.onoff*self.Pm)*self.dt
        if pred_error > self.limit:
            self.onoff = 1
            switched = True
        elif pred_error < -self.limit:
            self.onoff  = 0
            switched = True
        else:
            switched = False
                
        return self.onoff * self.Pm, switched
        
    ## Be in sync with the real system time
    def syncTime(self, target_time):
        real_time = datetime.datetime.now()
        if target_time > real_time:
            delta = target_time - real_time
            time.sleep(delta.seconds)
    
    def getObs(self, target_time):
        # Pull Current Observation
        obs = self.ecobee.getData()
        # Check Target Time is close to Thermostat time
        thermostatTime = datetime.datetime.strptime(obs["thermostatTime"], "%Y-%m-%d %H:%M:%S")
        time_diff = thermostatTime - target_time
        if abs(time_diff.total_seconds())<60:
            pass
        else:
            print("Discrepancy between Thermostat Time and Target Time", time_diff.total_seconds())
        '''
        if 'compCool1' in obs["equipmentStatus"].split(","):
            self.onoff = 1
        else:
            self.onoff = 0
        '''
        return obs['runtime']['actualTemperature'] * 0.1
        
    def step(self, u, normalized = False, k = -1):
        self.syncTime(self.cur_time)
        self.x = self.getObs(self.cur_time)
        
        self.idx_list.append(self.cur_time)
        self.x_list.append(self.x)
        self.u_list.append(u)
        print("{} x={}, on={}, u={}".format(self.cur_time, self.x, self.onoff, u))
        
        options = [800, 700]
        
        ## Translate u to setpoint: PWM -> SP
        control_time = self.cur_time
        tmp = []
        
        if k==1:
            on_for = (self.cum_error + u * self.timestep) // self.Pm
            on_for = max(0, on_for)
            print("{} On For = {} Minutes".format(self.cur_time, on_for))
        
        # Turn off the AC in the case of on_for = 15
        if (k==2):
            self.onoff = 0
            sp = options[int(self.onoff)]
            self.ecobee.setHold(coolHoldTemp = sp, heatHoldTemp = sp)
            
        for t in range(self.timestep//self.dt):
            ## Standard PWM
            if k!=1:
                u_discrete, switched = self.PWM(u)
                if switched:
                    print("Switch to {}".format(self.onoff))
                    sp = options[int(self.onoff)]
                    self.ecobee.setHold(coolHoldTemp = sp, heatHoldTemp = sp)
            ## My Heuristics
            else:
                if (t==0) & (u>0.1) & (on_for>0):
                    ## Turn On
                    self.onoff = 1
                    sp = options[int(self.onoff)]
                    self.ecobee.setHold(coolHoldTemp = sp, heatHoldTemp = sp)
                elif t == on_for:
                    # Turn Off
                    self.onoff = 0
                    sp = options[int(self.onoff)]
                    self.ecobee.setHold(coolHoldTemp = sp, heatHoldTemp = sp)
                u_discrete = self.onoff * self.Pm
                
            tmp.append(u_discrete)
            
            self.idx_record.append(control_time)
            self.u_record.append(u)
            self.command_record.append(u_discrete)
            
            error = u - u_discrete
            self.cum_error += error * self.dt
            print("{} x={} On/Off={}; cum error={}".format(control_time, self.x, self.onoff, self.cum_error))
            
            control_time += datetime.timedelta(minutes = self.dt)
            self.syncTime(control_time)
            
            ## Manual Override
            if t%4==0:
                self.x = self.getObs(control_time)
                
                if (self.x>77.5)|(self.x<72.5):
                    sp = 750
                    self.ecobee.setHold(coolHoldTemp = sp, heatHoldTemp = sp)
                    if self.x > 77.5:
                        self.onoff = 1
                    elif self.x < 72.5:
                        self.onoff = 0
                        
        self.cur_time += datetime.timedelta(minutes = self.timestep)
        return self.x, np.mean(tmp)
        
    def updateDist(self, weather):
        ## Schedule-based Occupancy
        home = pd.Series([1 if (time_idx.hour)>8 else 0 for  time_idx in weather.index], index = weather.index, name = 'Home')
        sleep = pd.Series([0 if (time_idx.hour)>8 else 1 for time_idx in weather.index], index = weather.index, name = 'Sleep')
        self.disturbance = pd.concat([weather.to_frame(), home.to_frame(), sleep.to_frame()], axis = 1)
        
        self.sp = pd.Series(np.ones(len(self.disturbance)) * 75, index = self.disturbance.index)
        
        # Reset
        self.u_record = []
        self.command_record = []
        self.idx_record = []
        
    def load(self, startOfDay, filePath = "/Users/chenbq/Desktop/CMU/Research 101/Grid/BuildSys'20/results"):
        try:
            data = pd.read_pickle("{}/ApartmentRecord-{}".format(filePath, startOfDay.strftime("%Y-%m-%d")))
            self.u_record = list(data[0])
            self.command_record = list(data[1])
            self.idx_record = list(data.index.to_pydatetime())
        except:
            pass
    
    def save(self, startOfDay, filePath = "/Users/chenbq/Desktop/CMU/Research 101/Grid/BuildSys'20/results"):
        data = pd.DataFrame(np.vstack([np.array(self.u_record), np.array(self.command_record)]).transpose(1, 0), index = self.idx_record)
        # data = data.loc[self.startOfDay:]
        data.to_pickle("{}/ApartmentRecord-{}".format(filePath, startOfDay.strftime("%Y-%m-%d")))
        
    ##TODO: Update Model Based on New Observations
    def updateModel(self):
        data = pd.read_pickle()
        raw = self.ecobee.getHistorical()
        
class Thermostat_Cluster():
    def __init__(self, params_list, myParams = None, timestep = 15, start_time = None, save_every = 4, saveName = ""):
        self.TCL_list = []
        
        self.timestep = timestep
        if start_time is None:
            self.cur_time = roundTime(datetime.datetime.now())
        else:
            self.cur_time = start_time
        self.startOfDay = None
        self.endOfDay = None
        
        self.updateDist(start_time = self.cur_time)
        
        ## Instantiate the TCLs
        for params in params_list:
            self.TCL_list.append(Thermostat(params, self.cur_time))
        
        if myParams is not None:
            self.TCL_list.append(myApartment(myParams, self.cur_time))
        
        self.n_agent = len(self.TCL_list)
        
        ## For saving record
        self.idx_list = [self.cur_time]
        self.x_record = []
        self.u_record = []
        self.p_record = []
        
        self.count = 0
        self.save_every = save_every
        self.saveName = saveName
        
    def reset(self, x_init = None, u_init = None):
        self.updateDist(self.cur_time)
        
        ## Reload Data if the loop was terminated prematurely
        if self.saveName == "":
            self.load()
        
        x_list = []
        u_list = []
        for i, item in enumerate(self.TCL_list):
            if x_init is None:
                if len(self.x_record)>0:
                    x, u = item.reset(self.x_record[-1][i], self.p_record[-1][i])
                else:
                    x, u = item.reset()
            else:
                x, u = item.reset(x_init = x_init[i], u_init = u_init[i])
                assert x == x_init[i]
                assert u == u_init[i]
            x_list.append(x)
            u_list.append(u)
        
        x = np.array(x_list)
        u = np.array(u_list)
        
        if len(self.x_record)>0:
            self.idx_list.append(self.cur_time)
        
        self.x_record.append(x)
        self.p_record.append(u)
        self.u_record.append(u)
            
        self.count += 1
        return x, u
    
    # A all sim version
    def step(self, u, normalized = False, k = -1):
        if self.count % self.save_every == 0:
            self.save()
        
        x_list = []
        p_list = []
        for i, item in enumerate(self.TCL_list):
            assert self.cur_time == item.cur_time
            x, p = item.step(u[i], normalized, k=k)
            x_list.append(x)
            p_list.append(p)
            
        self.cur_time += datetime.timedelta(minutes = self.timestep)
        
        if self.cur_time == self.endOfDay:
            self.save()
            self.x_record = []
            self.p_record = []
            self.u_record = []
            self.idx_list = []
            self.count = 0
            
            self.updateDist(start_time = self.cur_time)
            
        x = np.array(x_list)
        p = np.array(p_list)

        self.idx_list.append(self.cur_time)
        self.x_record.append(x)
        self.p_record.append(p)
        self.u_record.append(u)
        self.count += 1
        return x, p
     
    def pullDist(self, T, start_time = None):
        if start_time is None:
            start_time = self.cur_time
        end_time = start_time + datetime.timedelta(minutes = self.timestep*(T-1))
        print("pullDist", start_time, end_time)
        x_prev_list = []
        d_list = []
        x_lower_list = []
        x_upper_list = []
        for TCL in self.TCL_list:
            x_prev, d, x_lower, x_upper = TCL.pullDist(start_time, end_time)
            x_prev_list.append(x_prev)
            d_list.append(d)
            x_lower_list.append(x_lower)
            x_upper_list.append(x_upper)
        return x_prev_list, d_list, x_lower_list, x_upper_list
        
    # Get Up-to-Data Exog variables at the beginning of each day
    def updateDist(self, start_time = None, duration = 2):
        weather_forecast = weather.getWeather(start_time, duration)
        self.startOfDay = datetime.datetime(year = self.cur_time.year, month = self.cur_time.month, day = self.cur_time.day)
        self.endOfDay = datetime.datetime(year = self.cur_time.year, month = self.cur_time.month, day = self.cur_time.day+1)
        for TCL in self.TCL_list:
            TCL.updateDist(weather_forecast)
    
   
    def load(self, filePath = "/Users/chenbq/Desktop/CMU/Research 101/Grid/BuildSys'20/results"):
        try:
            X = pd.read_pickle("{}/X{}-{}".format(filePath, self.saveName, self.startOfDay.strftime("%Y-%m-%d")))
            self.x_record = list(X.values)
            self.idx_list = list(X.index.to_pydatetime())
            print("Load Prev Data")
        except:
            pass
            
        try:
            P = pd.read_pickle("{}/P{}-{}".format(filePath, self.saveName, self.startOfDay.strftime("%Y-%m-%d")))
            self.p_record = list(P.values)
        except:
            pass
            
        try:
            U = pd.read_pickle("{}/U{}-{}".format(filePath, self.saveName, self.startOfDay.strftime("%Y-%m-%d")))
            self.u_record = list(U.values)
        except:
            pass
            
        try:
            self.TCL_list[-1].load(self.startOfDay)
        except:
            pass
            
    def save(self, filePath = "/Users/chenbq/Desktop/CMU/Research 101/Grid/BuildSys'20/results"):
        X = pd.DataFrame(np.array(self.x_record), index = self.idx_list)
        X.to_pickle("{}/X{}-{}".format(filePath, self.saveName, self.startOfDay.strftime("%Y-%m-%d")))
        P = pd.DataFrame(np.array(self.p_record), index = self.idx_list)
        P.to_pickle("{}/P{}-{}".format(filePath, self.saveName, self.startOfDay.strftime("%Y-%m-%d")))
        U = pd.DataFrame(np.array(self.u_record), index = self.idx_list)
        U.to_pickle("{}/U{}-{}".format(filePath, self.saveName, self.startOfDay.strftime("%Y-%m-%d")))
        try:
            self.TCL_list[-1].save(self.startOfDay)
        except:
            pass
