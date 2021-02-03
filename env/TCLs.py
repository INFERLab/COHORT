import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

  
def local_reward(x, sp, u, Delta=1):
    if np.abs(x-sp)>Delta:
        return -(x-sp)**2 #- u
    else:
        return 0
    
class RBC():
    def __init__(self, N, theta, Delta, sign = -1):
        self.T_upper = theta+Delta
        self.T_lower = theta-Delta
        # + for heating;
        # - for cooling
        self.sign = sign
        
        self.u = np.random.randint(2, size = N)
        
    def OnOff(self, x):
        if self.sign > 0:
            # Heating
            self.u[x<=self.T_lower] = 1
            self.u[x>=self.T_upper] = 0
        else:
            # Cooling
            self.u[x>=self.T_upper] = 1
            self.u[x<=self.T_lower] = 0
        return self.u
    
class TCL_Cluster():
    def __init__(self, N, R=2, C=2, Pm=5.6, eta=2.5, theta=22.5, Delta=0.3, sign = -1, eps = 0.1):
        '''
        N: number of TCLs
        Default nominal value from Hao et al., 2014
        '''
        self.N = N
        self.TCL_list = []
        
        # Sample R and C from a uniform distribution from the nominal value
        if isinstance(R, int) | isinstance(R, float):
            self.R = np.random.uniform((1-eps*0.5)*R, (1+eps*0.5)*R, N)
        else:
            assert len(R) == N
            self.R = R
            
        if isinstance(C, int) | isinstance(C, float):
            self.C = np.random.uniform((1-eps*0.5)*C, (1+eps*0.5)*C, N)
        else:
            assert len(C) == N
            self.C = C
            
        if isinstance(Pm, int) | isinstance(Pm, float):
            self.Pm = np.random.uniform((1-eps*0.5)*Pm, (1+eps*0.5)*Pm, N)
        else:
            assert len(Pm) == N
            self.Pm = Pm
        
        # Assume everything else is constant for the time being
        if isinstance(eta, int) | isinstance(eta, float):
            self.eta = np.ones(N)*eta
        else:
            assert len(eta) == N
            self.eta = eta

        if isinstance(theta, int) | isinstance(theta, float):
            self.theta = np.ones(N)*theta
        else:
            assert len(theta) == N
            self.theta = theta
            
        if isinstance(Delta, int) | isinstance(Delta, float):
            self.Delta = np.ones(N)*Delta
        else:
            assert len(Delta) == N
            self.Delta = Delta
        
        ## Instantiate the TCLs
        for i in range(self.N):
            self.TCL_list.append(TCL(self.R[i], self.C[i], self.Pm[i], self.eta[i], self.theta[i], self.Delta[i], sign))
            
    def reset(self, x_init = None, u_init = None):
        x_list = []
        u_list = []
        for i, item in enumerate(self.TCL_list):
            if x_init is None:
                x, u = item.reset()
            else:
                x, u = item.reset(x_init = x_init[i], u_init = u_init[i])
                assert x == x_init[i]
                assert u == u_init[i]
            x_list.append(x)
            u_list.append(u)
        return np.array(x_list), np.array(u_list), False
        
    def step(self, u, Ta = 32, dt = 1, normalized = True, continuous = False):
        x_list = []
        r_list = [] # local reward for agents
        P_list = []
        for i, item in enumerate(self.TCL_list):
            x, r, P = item.step(u[i], Ta, dt, normalized, continuous)
            x_list.append(x)
            r_list.append(r)
            P_list.append(P)
        return np.array(x_list), np.array(r_list), np.array(P_list), False

class TCL():
    def __init__(self, R, C, Pm, eta, theta, Delta, sign, reward_fn = local_reward, Ta_bar = 32, limit = None):
        '''
        R: thermal resistance (C/kW)
        C: thermal capacitance (kWh/C)
        Pm: rated electrical power (kW);
        eta: COP
        theta: temperature Setpoint (C)
        Delta: temperature deadband (C)
        sign: (+) for heating and (-) for cooling
        '''
        self.R = R
        self.C = C
        self.Pm = Pm
        self.eta = eta
        self.theta = theta
        self.Delta = Delta
        self.sign = sign
        
        self.x = None
        self.reward_fn = reward_fn
        
        ## Intersective Method
        '''
        T_on = self.R * self.C * np.log((self.theta + self.Delta-Ta_bar+self.R * self.Pm * self.eta) / (self.theta - self.Delta-Ta_bar+self.R * self.Pm * self.eta))
        T_off = self.R * self.C * np.log((self.theta - self.Delta - Ta_bar) / (self.theta + self.Delta - Ta_bar) )
        self.T_pwm =  0.75 * (T_on + T_off) 
        # print(self.T_pwm)
        # The cumulative time
        self.t = np.random.uniform() * self.T_pwm
        '''
        ##
        self.out = np.random.choice(2) * self.Pm
        self.cum_error = 0
        
        if limit is not None:
            self.limit = limit
        else:
            self.limit = 0.1 #np.random.uniform(0.2, 0.4)*self.Pm
        
    def reset(self, x_init = None, u_init = None):
        # Initialize State
        if x_init is None:
            self.x = np.random.uniform(self.theta-self.Delta, self.theta+self.Delta)
            self.out = np.random.choice(2, p = (0.64, 0.36)) * self.Pm
        else:
            self.x = x_init
            self.out = u_init
        return self.x, self.out
        
    def getParameters(self):
        param = dict()
        param["R"] = self.R
        param["C"] = self.C
        param["Pm"] = self.Pm
        param["eta"] = self.eta
        param["theta"] = self.theta
        param["Delta"] = self.Delta
        param["sign"] = self.sign
        return param
    
    def PWM(self, signal, dt):
        # The intersective method:
        '''
        if u > self.Pm * self.t / self.T_pwm:
            return self.Pm
        else:
            return 0
        '''
        ## Delta-Sigma Method
        # Make decision based on predicted error!
        pred_error = self.cum_error + (signal - self.out)*dt
        if pred_error > self.limit:
            self.out = self.Pm
        elif pred_error < -self.limit:
            self.out = 0
        return self.out
            
   
    ## system dynamics: T_{k+1} = aT_k+(1-a)(Ta + sign * b u)
    def step(self, u, Ta, dt, normalized, continuous):
        # x: temperature
        ######################
        # u (Power) in {0, Pm}
        ######################
        # dt: timestep in hr
        # Normalized: A flag for if action is normalized
        
        a = np.exp(-dt/(self.R*self.C))
        b = self.eta * self.R
        
        if normalized:
            u *= self.Pm
        
        # Make the continous action discrete with PWM
        if continuous:
            u_discrete = self.PWM(u, dt)
            error = u - u_discrete
            self.cum_error += error * dt
            u = u_discrete
        else:
            self.out = u
            
        self.x = a * self.x + (1-a)*(Ta + self.sign * b * u)
        r = self.reward_fn(self.x, self.theta, u, Delta = self.Delta)
        #self.t += dt
        #self.t = self.t % self.T_pwm
        return self.x, r, u
