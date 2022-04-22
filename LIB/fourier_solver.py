#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 15:13:06 2022

@author: pkrah
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 11:27:56 2022

@author: pkrah
"""
import numpy as np
from numpy import pi, reshape, meshgrid, sin, cos
from numpy.fft import fft,ifft,fft2,ifft2
from time import perf_counter


class params_class:
    def __init__(self, pde = "advection",dim = 2 ,L=[1,1] ,N = [2**7, 2**6], T=0.1, Nt=300 , case="bunsen_flame"):
        # init geometry
        self.geom.L = np.asarray(L) # domain size
        self.geom.N = np.asarray(N) # number of spatial points in each dimension
        self.time.T = T
        self.time.dt = T/Nt
        self.geom.dX = self.geom.L/self.geom.N
        self.geom.X = [np.arange(0, self.geom.L[d],self.geom.dX[d]) for d in range(dim)]
        self.geom.K = [2*np.pi*np.fft.fftfreq(self.geom.N[k],d=self.geom.dX[k]) for k in range(dim)]
        self.geom.Xgrid = meshgrid(*self.geom.X)
        # init time
        self.time.t = np.arange(0,self.time.T,self.time.dt)
        #self.advection_speed= 0.1#0.1 # velocity of moving vortex
        self.advection_speed = 0.1 # 0.1 # velocity of moving vortex
        #self.w0 = [45, 45]   # initial vorticity
        self.w0 = [30, 30]  # initial vorticity
        self.r0 = 0.005    # initial size of vortex
        self.decay_constant = 0.2#0.2 # decay of the initial vorticity
        # init advection reaction diffusion
        self.diffusion_const = 1e-4# adjust the front width ( larger value = larger front)
        self.reaction_const = 10    # adjust the propagation of the flame (larger value = faster propagation)
        self.dim = dim
        self.shape = [*self.geom.N] + [len(self.time.t)]
        self.fom_size = np.prod(self.geom.N)
        self.pde = pde
        self.diff_operators=self.diff_operators(self.geom.N,self.geom.dX, dim = dim)
        self.case = case
        self.info_dict ={}
        self.inicond = self.set_inicond(case)
                
        if pde == "advection":
            if dim == 2:
                self.rhs = lambda qvals, time: rhs_advection2D_periodic(self, qvals, time)
            else:
                self.rhs = lambda qvals, time: rhs_advection1D_periodic(self, qvals)
        elif pde == "burgers":
            if dim == 2:
                self.rhs = lambda qvals, time: rhs_burgers2D_periodic(self, qvals)
            else:
                self.rhs = lambda qvals, time: rhs_burgers1D_periodic(self, qvals)
        elif pde == "react":
            if dim == 2:
                self.rhs = lambda qvals, time: rhs_advection_reaction_diffusion_2D_periodic(self, qvals, time)
            else:
                self.rhs = lambda qvals, time: rhs_advection_reaction_diffusion_1D_periodic(self, qvals)

    class material:
                schmidt_number = 1
                reynolds_number = 200
                gamma = 1.197
                R = 8.314462
                W = 1
                cp= 1004.7029
                reaction_const =  4 # adjust the propagation of the flame (larger value = faster propagation) Y
                reaction_const_h =  0.1 # adjust energy transfer to NS  (larger value = faster propagation)
                mu0 = 3.6e-7*1950**0.7#3.6e-0                 # dynamic viscosity
                W1 = 1
                dh1 = 3.5789e6
                preexponent = 2.8e9
                activation_temperature = 2.0130e4
                Le = 1                       # Lewis Number
                prandtl_number = 0.720  # Prandtl Number
    class geom:
            L = np.asarray([1.0, 1.0])
            N = np.asarray([2**7, 2**7])
    class rom:
            pass
    class odeint:
            timestep = 0.05
    class time:
            T = 1
            dt = 0.05
            
    class diff_operators:
        def __init__(self, Ngrid, dX, dim):
            self.K = [2*np.pi*np.fft.fftfreq(Ngrid[k],d=dX[k]) for k in range(dim)]
            self.Kgrid = meshgrid(*self.K)
            if dim == 1:
                self.Dx = lambda q_fft: self.Dx_1D( q_fft)
                self.Dxx = lambda q_fft: self.Dxx_1D( q_fft)
            else:
                self.Dx = lambda q_fft: self.Dx_2D( q_fft)
                self.Dxx = lambda q_fft: self.Dxx_2D( q_fft)
                self.Dy = lambda q_fft: self.Dy_2D( q_fft)
                self.Dyy = lambda q_fft: self.Dyy_2D( q_fft)
  
        def Dx_1D(self, qhat):
            """
            derivative in x, assuming qhat is a 2D array already transformed
            in fourier space using qhat = fft2(q)
            """
            kx = self.K[0]
            # first derivative
            dq_dx = np.real(ifft2(kx * qhat * (1j)))
        
            return dq_dx
        
        def Dxx_1D(self, qhat):
            """
            derivative in x, assuming qhat is a 2D array already transformed
            in fourier space using qhat = fft2(q)
            """
            kx  = self.K[0]
            # second derivative
            ddq_ddx = np.real(ifft2(-kx**2 * qhat ))
            
            return ddq_ddx
        
        
        def Dx_2D(self, qhat):
            """
            derivative in x, assuming qhat is a 2D array already transformed
            in fourier space using qhat = fft2(q)
            """
            [kx, ky] = self.Kgrid
            # first derivative
            dq_dx = np.real(ifft2(kx * qhat * (1j)))
        
            return dq_dx
        
        def Dxx_2D(self, qhat):
            """
            derivative in x, assuming qhat is a 2D array already transformed
            in fourier space using qhat = fft2(q)
            """
            [kx, ky] = self.Kgrid
            # second derivative
            ddq_ddx = np.real(ifft2(-kx**2 * qhat ))
            
            return ddq_ddx
        
        def Dy_2D(self, qhat):
            """
            derivative in x, assuming qhat is a 2D array already transformed
            in fourier space using qhat = fft2(q)
            """
            [kx, ky] = self.Kgrid
            # first derivative
            dq_dy = np.real(ifft2(ky * qhat * (1j)))
            
            return dq_dy
        
        def Dyy_2D(self, qhat):
            """
            derivative in x, assuming qhat is a 2D array already transformed
            in fourier space using qhat = fft2(q)
            """
            [kx, ky] = self.Kgrid
            # second derivative
            ddq_ddy = np.real(ifft2(-ky**2 * qhat))
           
            return ddq_ddy



    def velocity_field(self, t, sampleMeshIndices = None):

        # dipole vortex benchmark
        c = self.advection_speed
        tau = self.decay_constant
        [X, Y] = self.geom.Xgrid
        L = self.geom.L
        T = self.time.T
        we = np.asarray(self.w0)
        r0 = self.r0
        ri = lambda xi, yi: (X - xi) ** 2 + (Y - yi) ** 2
        if self.case == "pacman":
            (x1, y1) = (0.6 - c * t, 0.49 * L[1])
            (x2, y2) = (0.6 - c * t, 0.51 * L[1])
            distance1 = (ri(x1, y1)/r0)**2
            distance2 = (ri(x2, y2)/r0)**2
            ux = -we[0] * (Y - y1) * np.exp(-(distance1)) + we[1] * (Y - y2) * np.exp(-(distance2))
            uy = we[0] * (X - x1) * np.exp(-(distance1)) - we[1] * (X - x2) * np.exp(-(distance2) )
            ux *= np.exp(-(t / tau) ** 2)
            uy *= np.exp(-(t / tau) ** 2)
        elif self.case == "bunsen_flame":
            inv_2pi = 1/(2*np.pi)
            my_sawtooth = lambda x: (x*inv_2pi)%1
            (x1, y1) = (0.1 * L[0] + 0.8*L[0] * my_sawtooth(2*pi/T*t * 5), 0.46*L[1])
            (x2, y2) = (0.1 * L[0] + 0.8*L[0] * my_sawtooth(2*pi/T*t * 5), 0.54*L[1])
            ux = + we[0] * (Y - y1) * np.exp(-(ri(x1, y1) / r0) ** 2) - we[1] * (Y - y2) * np.exp(-(ri(x2, y2) / r0) ** 2)
            uy = - we[0] * (X - x1) * np.exp(-(ri(x1, y1) / r0) ** 2) + we[1] * (X - x2) * np.exp(-(ri(x2, y2) / r0) ** 2)
            #ux *= sin(2*pi/T*t*4)**2#np.exp(-(t / tau) ** 2)
            #uy *= sin(2*pi/T*t*4)**2#np.exp(-(t / tau) ** 2)

            smoothwidth = self.geom.dX[0] * 1
            softstep = lambda x, d: (1 + np.tanh(x / d)) * 0.5
            ux += (0.05)*(softstep(Y-0.3*self.geom.L[1], 2*smoothwidth)-softstep(Y-0.7*self.geom.L[1], 2*smoothwidth))*softstep(X-0.05*self.geom.L[0], smoothwidth)#*params.penalisation.weights
            #ux += c
        else:
            pass

        ux = np.reshape(ux, -1)
        uy = np.reshape(uy, -1)
        if sampleMeshIndices is not None:
            ux = np.take(ux, sampleMeshIndices, axis=0)
            uy = np.take(uy, sampleMeshIndices, axis=0)

        return np.reshape(ux,-1), np.reshape(uy,-1)

    def set_inicond(self,case=""):
        dx = np.min(self.geom.dX)
        L  = np.min(self.geom.L)
        delta = 0.005 * L
        self.front = lambda x: 1/(1 + np.exp(-x/delta))
        self.forward=lambda x: np.sigmoid(x/delta)
        if self.dim == 2:
            if case== "pacman":
                [X,Y] = self.geom.Xgrid
                phi0 = np.sqrt((X-L*(0.4))**2+(Y-L*(0.5))**2) - 0.2*L
                q0 = self.front(phi0)
                return 1-reshape(q0,-1)
            if case == "bunsen_flame":
                return np.reshape((1-self.penalisation.weights),-1)
            else:
                return np.reshape(np.zeros(self.geom.N),-1)
        else:
            if case == "reaction1D":
                self.front = lambda x: 0.5 * (1 - np.tanh(x / 2))
                return self.front(2*(np.abs(self.geom.X[0] - self.geom.L[0] * 0.5)-2)/self.reaction_const)
            else:
                return self.front(self.geom.X[0]-self.geom.L[0]*0.05)-self.front(self.geom.X[0]-self.geom.L[0]*0.15)

###############################################################################
# RIGHT HAND SIDES
###############################################################################
def rhs_advection_reaction_diffusion_2D_periodic(self, q, t):
    a = self.diffusion_const
    b = self.reaction_const
    #chi = self.penalisation.eta
    #weights = self.penalisation.weights
    #q_target = self.penalisation.q_target
    # to fourier space
    q = reshape(q, self.geom.N)
    qhat = fft2(q)
    # first derivative
    dq_dx = self.diff_operators.Dx(qhat)
    dq_dy = self.diff_operators.Dy(qhat)
    # second derivative
    ddq_ddx = self.diff_operators.Dxx(qhat)
    ddq_ddy = self.diff_operators.Dyy(qhat)

    ux, uy = self.velocity_field(t)
    ux, uy = reshape(ux,self.geom.N), reshape(uy,self.geom.N)
    # right hand side
    rhs = - ux * dq_dx -  uy * dq_dy + a * (ddq_ddx + ddq_ddy) - b * q**2*(q-1) #-chi * weights * (q-q_target)

    return reshape(rhs,-1)


def rhs_advection_reaction_diffusion_1D_periodic(self, q, t):
    """
    This function computes the rhs
    :param q: state at time t
    :param t: time
    :param dq: direction of the derivative dq
    :return: drhs(q,t)/dq * dq
    """

    delta = self.reaction_const
    # to fourier space
    qhat = fft(q)
    # second derivative
    ddq_ddx = self.diff_operators.Dxx(qhat)
    # right hand side
    rhs = ddq_ddx + 8 / delta ** 2 * q ** 2 * ( 1 -  q )

    return reshape(rhs, -1)


def rhs_burgers1D_periodic(self,q):

    nu = self.diffusion_const
    K = self.geom.K[0]

    # spectral derivatives
    qhat = fft(q)
    dq = ifft(K * qhat * (1j))
    ddq = ifft(- K**2 *qhat)

    # right hand side
    rhs = -q * dq + nu * ddq
    return np.real( rhs)

def rhs_burgers2D_periodic(self,q):

    q = reshape(q, self.geom.N)
    nu = self.diffusion_const
    K = self.geom.K
    [kx, ky] = meshgrid(K[0], K[1])

    # spectral derivatives
    qhat = fft2(q)
    # first derivative
    dq_dx = np.real(ifft2(kx * qhat * (1j)))
    dq_dy = np.real(ifft2(ky * qhat * (1j)))
    # second derivative
    ddq_ddx = np.real(ifft2(-kx**2 * qhat ))
    ddq_ddy = np.real(ifft2(-ky**2 * qhat))

    # right hand side
    rhs = -q * dq_dx - q * dq_dy + nu * (ddq_ddx + ddq_ddy)

    return reshape(rhs,-1)


def rhs_advection1D_periodic(self, q):
        c = self.advection_speed
        K = self.geom.K[0]
        qhat = fft(q)
        qhat_x = K * qhat * (1j)
        return -c*np.real(ifft(qhat_x))


def rhs_advection2D_periodic(self,q,time):
        q = reshape(q,self.geom.N)
        L = self.geom.L
        K = self.geom.K

        v_x = L[0]* 10 * ( -sin(2 * pi * time))
        v_y = L[1] * 10 * (  cos(2 * pi * time))
        [kx,ky] = meshgrid(K[0],K[1]) 
        qhat = fft2(q)
        dq_dx = np.real(ifft2(kx * qhat * (1j)))
        dq_dy = np.real(ifft2(ky * qhat * (1j)))
        return reshape(- (v_x*dq_dx + v_y *dq_dy),-1)



###############################################################################
# solver
###############################################################################
def solve_FOM(params):
    from scipy.integrate import solve_ivp as ode45
    print("Solving Full Order Model")
    q0 =params.inicond

    time_odeint = perf_counter() # save timing
    #q = odeint(params.rhs,q0,params.time.t).transpose()
    rhs = lambda state, t: params.rhs(t,state)
    time = params.time.t
    #q = timesteper(rhs, [time[0], time[-1]], q0, t_vec=time, tau=params.odeint.timestep).T
    #nrhs = 4*len(time)
    ret = ode45(rhs, [time[0], time[-1]], q0, method="RK45",rtol=1e-9 , t_eval=time)
    q = ret.y
    nrhs = ret.nfev
    time_odeint = perf_counter()-time_odeint
    print("t_cpu = %1.3f #rhs calls: %d"%(time_odeint,nrhs))
    params.info_dict["num_rhs_calls"] = nrhs
    # reshape to data field!

    q = reshape(q, [*params.geom.N[:params.dim], -1])

    return params, q

def solve_pacman(Ngrid = [2**9, 2**9]):
    # choose pde [burgers,advection]
    pde = "react"
    case = "pacman"
    T = 3
    params = params_class(case=case, pde=pde, N=Ngrid, T = T)
    params,q = solve_FOM(params)
    
    return q

if __name__ == "__main__":
    
    # choose pde [burgers,advection]
    pde = "react"
    case = "pacman"
    Ngrid = [2**9,2**9]
    T = 2
    params = params_class(case=case, pde=pde, N=Ngrid, Nt = 200, T = T)
    params,q = solve_FOM(params)
    
    #%%
    import matplotlib.pyplot as plt
    plt.pcolormesh(params.geom.X[0],params.geom.X[1],q[...,-1])
    plt.colorbar()