from math import pi, sqrt
from numpy import linspace, zeros, concatenate, maximum, array, arange, exp, sinh, sin, log, gradient, asarray
from numpy.random import rand
from scipy.integrate import ode, trapz
from time import time

el = 15 # Lenght of habitat
w_0 = 0.01   # Cost per unit of fishing effort
w_1 = 0.001 # cost rate
p = 1
N = 120 # number of nodes
x = linspace(-el/2,el/2,N) # Habitat for PDE
h = el/N # distance between nodes

beta = 10 # very large values will trigger overflow

# Divided second order finite differences

def div_diff2(g): 
    return (g[:-2] + g[2:] - 2*g[1:-1])/h**2

# Metrics

def metric(u): # example: if u population density -> metric(u) = total biomass in the habitat
    return trapz(u, x = x) # approximate integral via trapezoidal rule

def reserve_length(f): # length of reserve areas (where f = 0)
    l = 0
    for z in f:
        if abs(z) < 1e-16:
            l += h
    return l

# Tourism density functions

# a_0: tourism-fishing relative price
# u_0: biomass density threshold value

def g(u,u_0):
    return 1/(1+exp(-beta*(u-u_0)))
              
def z_t(u,a_0,u_0):
    return a_0*g(u,u_0)

# Optimal tourism effort density

def f_so(u,lt,gamma):
    # lt is lambda
    fstar = (p*u - w_0 - lt*(u + gamma*u*u))/w_1/2 # minus sign in lambda because of its definition below
    return maximum(fstar,zeros(fstar.size))

# Temporal derivative + Dirichlet boundary conditions
# y = (u,lambda,y3,y4) y3, y4 for better convergence, at equilibrium they are constants (Holly's Matlab code)

def fun_z(t,y,gamma,a_0,u_0):
    u = y[:N] #u
    lt = y[N:2*N] #lambda
    y3 = y[2*N:3*N]
    y4 = y[3*N:4*N]
    u_res = u[1:-1] #interior nodes
    lt_res = lt[1:-1] #interior nodes
    f = f_so(u,lt,gamma)
    f_res = f[1:-1] #interior nodes
    du = (y3[1:-1] - y4[1:-1]*u_res)*u_res - f_res*u_res + div_diff2(u) #temporal derivative at interior nodes
    dlambda = p*f_res + lt_res*(y3[1:-1] - 2*u_res*y4[1:-1] - f_res)  + div_diff2(lt) + a_0*beta*g(u_res,u_0)*(1-g(u_res,u_0)) #temporal derivative at interior nodes; note derivative of tourism function z'(u)
    dy3 = 1/2*(1 - y3)
    dy4 = 1/2*(1 + gamma*f - y4)
    return concatenate([[0],du,[0],[0],dlambda,[0],dy3,dy4]) # now we impose Dirichlet boundary conditions (0 at boundary)

# Solver, using Python integrator

def ode_solver_z(tf,dt,init,gamma,a_0,u_0,bol = False):
    # choose something like dt ~ h^2
    solver = ode(fun_z)
    solver.set_integrator('dopri5') # This is Runge-Kutta of order 4 with adaptive step size (can try others... vode, dop853)
    solver.set_f_params(gamma,a_0,u_0)
    solver.set_initial_value(init,0)
    print('a_0 = %.3f' % a_0)
    print('u_0 = %.3f' % u_0)
    print('beta = %.1f' % beta)
    print('gamma = %.2f' % gamma)
    t = arange(1,tf,dt)
    tic = time()
    while solver.successful() and solver.t < tf:
          solver.integrate(solver.t + dt)
    print('Solution in %.2f s' % (time()-tic) )
    sols = solver.y
    if bol:
        uu = sols[:N]
        ff =  f_so(uu,sols[N:2*N],gamma)
        print('Total fishing revenue = %.2f' % (metric( p*uu*ff - (w_0 + w_1*ff)*ff ) ))
        print('Total tourism revenue = %.2f' % (metric( z_t(uu,a_0,u_0) ) ))
    return sols

# random initial conditions (it is usually better if you can provide a good guess of IC)

def ic():
    u0 = 0.5*(1+(2*rand(x.size-2)-1))
    lt0 = 0.5*(1+(2*rand(x.size-2)-1))
    y30 = zeros(x.size)
    y40 = zeros(x.size)
    y30[:] = 0.01
    y40[:] = 0.01
    print("Initial conditions randomized")
    return concatenate([[0],u0,[0],[0],lt0,[0],y30,y40])
