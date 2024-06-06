import numpy as np
import matplotlib.pyplot as plt
import firedrake as df
import os
from scipy.ndimage import gaussian_filter
import torch

# Necessary for multipool execution
os.environ['OMP_NUM_THREADS'] = '1'
df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 4
df.parameters['allow_extrapolation'] = True


# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

# This quadrature scheme is not high order (but still works).  Thanks to Mauro P. for point this out.
from numpy.polynomial.legendre import leggauss
def half_quad(order):
    points,weights = leggauss(order)
    points=points[(order-1)//2:]
    weights=weights[(order-1)//2:]
    weights[0] = weights[0]/2
    return points,weights

class SpecFO(object):
    """ A class for solving the ansatz spectral in vertical-CG in horizontal Blatter-Pattyn equations """
    def __init__(self, nx=100, ny=100, dx=150e3, dy=150e3):

        # Resolution        
        self.nx = nx
        self.ny = ny
        # Domain size
        self.dx = dx
        self.dy = dy
        mesh = self.mesh = df.RectangleMesh(nx, ny, dx, dy)


        # Funcion space for velocity components
        E_cg = df.FiniteElement('CG', mesh.ufl_cell(), 1)
        self.E_V = df.MixedElement([E_cg, E_cg, E_cg, E_cg])
        self.V = df.FunctionSpace(mesh, self.E_V)
        # Standard CG function space
        self.Q_cg = df.FunctionSpace(mesh, 'CG', 1)

        self.one = df.Function(self.Q_cg)
        self.one.assign(1.0)
        self.area = df.assemble(self.one*df.dx)

        self.build_variables()
        self.build_forms()
        self.get_map_indexes()


    # Compute indexes to map from images to DOF's
    def get_map_indexes(self):
        x = df.SpatialCoordinate(self.mesh)
        x0 = df.interpolate(x[0], self.Q_cg).dat.data
        y0 = df.interpolate(x[1], self.Q_cg).dat.data
        self.indexes = np.lexsort((x0, y0))


    def build_variables(
            self,
            n=3.0,  # Glen's exponent
            g=9.81,  # gravity
            rho_i=917.,  
            rho_w=1000.,  
            eps_reg=1e-5,
            Be=464158, # Ice softness
            #Be=500000.,
            p=1, 
            q=1, 
            beta2=1e-1,
        ):

        self.n = df.Constant(n)                    
        self.g = df.Constant(g)                    
        self.rho_i = df.Constant(rho_i)            
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.Be = df.Constant(Be)                  
        self.p = df.Constant(p)                
        self.q = df.Constant(q)            

        # Traction coefficient
        self.beta2 = df.Function(self.Q_cg)  
        self.beta2.dat.data[:] = beta2
        # Thickness
        self.H = df.Function(self.Q_cg)
        # Bed 
        self.B = df.Function(self.Q_cg)
        # Surface
        self.S = self.B + self.H
        # Effective pressure
        self.N = df.Function(self.Q_cg)

        self.U = df.Function(self.V)
        self.ubar = self.U[0]               
        self.vbar = self.U[1]
        self.udef = self.U[2]
        self.vdef = self.U[3]

        # Test functions
        self.Lambda = df.TestFunction(self.V)
        self.lamdabar_x = self.Lambda[0]  
        self.lamdabar_y = self.Lambda[1]
        self.lamdadef_x = self.Lambda[2]
        self.lamdadef_y = self.Lambda[3]

        # TEST FUNCTION COEFFICIENTS
        coef = [lambda s:1.0, lambda s:1./(n+1)*((n+2)*s**(n+1) - 1)]  
        dcoef = [lambda s:0, lambda s:(n+2)*s**n]           
 
        u_ = [self.ubar,self.udef]
        v_ = [self.vbar,self.vdef]
        lamda_x_ = [self.lamdabar_x,self.lamdadef_x]
        lamda_y_ = [self.lamdabar_y,self.lamdadef_y]

        self.u = VerticalBasis(u_,coef,dcoef)
        self.v = VerticalBasis(v_,coef,dcoef)
        self.lamda_x = VerticalBasis(lamda_x_,coef,dcoef)
        self.lamda_y = VerticalBasis(lamda_y_,coef,dcoef)

        self.U_b = df.as_vector([self.u(1),self.v(1)]) 
 

    def build_forms(self):
        # Assemble FEM forms
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        p = self.p
        q = self.q
        beta2 = self.beta2

        u = self.u
        v = self.v
        lamda_x = self.lamda_x
        lamda_y = self.lamda_y
        H = self.H
        S = self.S


        # For now just assume 0 water pressure
        N = self.N = 0.1*rho_i*g*H 

        def dsdx(s):
            return 1./H*(S.dx(0) - s*H.dx(0))

        def dsdy(s):
            return 1./H*(S.dx(1) - s*H.dx(1))

        def dsdz(s):
            return -1./H 

        # 2nd INVARIANT STRAIN RATE
        def epsilon_dot(s):
            return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                        +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                        +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                        +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                        + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                        + eps_reg)

        # VISCOSITY
        def eta_v(s):
            return Be/2.*epsilon_dot(s)**((1.-n)/(2*n))

        # MEMBRANE STRESSES
        def membrane_xx(s):
            return (lamda_x.dx(s,0) + lamda_x.ds(s)*dsdx(s))*H*(eta_v(s))*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        def membrane_xy(s):
            return (lamda_x.dx(s,1) + lamda_x.ds(s)*dsdy(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yx(s):
            return (lamda_y.dx(s,0) + lamda_y.ds(s)*dsdx(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yy(s):
            return (lamda_y.dx(s,1) + lamda_y.ds(s)*dsdy(s))*H*(eta_v(s))*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        # SHEAR STRESSES
        def shear_xz(s):
            return dsdz(s)**2*lamda_x.ds(s)*H*eta_v(s)*u.ds(s)

        def shear_yz(s):
            return dsdz(s)**2*lamda_y.ds(s)*H*eta_v(s)*v.ds(s)

        # DRIVING STRESSES
        def tau_dx(s):
            return rho_i*g*H*S.dx(0)*lamda_x(s)

        def tau_dy(s):
            return rho_i*g*H*S.dx(1)*lamda_y(s)

        # GET QUADRATURE POINTS (THIS SHOULD BE ODD: WILL GENERATE THE GAUSS-LEGENDRE RULE 
        # POINTS AND WEIGHTS OF O(n), BUT ONLY THE POINTS IN [0,1] ARE KEPT< DUE TO SYMMETRY.
        points,weights = half_quad(9)

        # INSTANTIATE VERTICAL INTEGRATOR
        vi = VerticalIntegrator(points,weights)

        # Basal shear stress (note minimum effective pressure ~1m head)
        tau_bx = -beta2*N*u(1)
        tau_by = -beta2*N*v(1)

        # weak form residuals for BP approximation.
        R_u_body = (- vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) + tau_bx*lamda_x(1) - vi.intz(tau_dx))*df.dx
        R_v_body = (- vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) + tau_by*lamda_y(1) - vi.intz(tau_dy))*df.dx

        # Residual
        self.R = (R_u_body + R_v_body)

        # Set up the solver
        self.problem = df.NonlinearVariationalProblem(self.R, self.U)
        self.solver = df.NonlinearVariationalSolver(self.problem)

        self.dU = df.Function(self.V)
        

    """
    Takes in a CG field and a 2d array of values, setting the CG field dofs appropriately.
    """
    def set_field(self, field, vals):
        field.dat.data[self.indexes] = vals.flatten() 

    """
    Converts CG dofs to a 2D image array. 
    """
    def get_field(self, field):
        return field.dat.data[self.indexes].reshape(self.nx+1, self.ny+1)
    
    """
    Sets all velocity components from a tensor.
    """
    def set_velocity(self, U):
        self.set_field(self.U.sub(0), U[0,:,:])
        self.set_field(self.U.sub(1), U[1,:,:])
        self.set_field(self.U.sub(2), U[2,:,:])
        self.set_field(self.U.sub(3), U[3,:,:])

    """
    Get the velocity in a nice format for plotting.
    """
    def get_velocity(self):
        # Vertically averaged velocity
        ubar0 = self.get_field(self.U.sub(0))
        ubar1 = self.get_field(self.U.sub(1))

        udef0 = self.get_field(self.U.sub(2))
        udef1 = self.get_field(self.U.sub(3))

        # Surface velocity
        s0 = ubar0-(5./4.)*udef0
        s1 = ubar1-(5./4.)*udef1

        return ubar0, ubar1, udef0, udef1, s0, s1
    

    """
    Get the gradient of loss w.r.t. velocity
    """
    def get_dU(self):
        du0 = self.get_field(self.dU.sub(0))
        du1 = self.get_field(self.dU.sub(1))
        du2 = self.get_field(self.dU.sub(2))
        du3 = self.get_field(self.dU.sub(3))
        
        dU = np.stack([du0, du1, du2, du3])[np.newaxis,:,:,:]
        dU = torch.tensor(dU, dtype=torch.float32)
        return dU
    
    """
    Generates geometry for an ideal ice sheet with bumpy bed
    """
    def get_geometry(
        self,
        S0 = 2000, # Ice sheet peak elevation
        B0 = 400, # Scales size of bed bumps
        sigma = 5.,
        mid_offset = np.array([0.,0.])
    ):

        xs = np.linspace(0., self.dx, self.nx+1)
        ys = np.linspace(0., self.dy, self.ny+1)
        xx, yy = np.meshgrid(xs, ys)

        # Surface elevation
        p_mid = np.array([xs.mean(), ys.mean()]) + mid_offset
        d = np.linalg.norm(np.stack([xx, yy]) - p_mid[:,np.newaxis, np.newaxis], axis=0)
        S = S0*np.sqrt(np.maximum(1. - d/55e3, np.zeros_like(d)))

        # Smooth some noise
        noise = np.random.randn(self.nx+1, self.ny+1)
        B = gaussian_filter(noise, sigma, axes=(0,1))

        # Remap values
        B = B0*((B - B.min()) / (B.max() - B.min()))

        # Thickness
        H = S-B
        H[H < 10.] = 10.

        return B, H
    
class PINNLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, U, B, H, model):

        ctx.model = model
        ctx.U = U 
        ctx.B = B
        ctx.H = H

        model.set_velocity(U[0])
        model.set_field(model.B, B)
        model.set_field(model.H, H)

        df.assemble(model.R, tensor=model.dU)
        ctx.dU = model.dU.dat.data[:]
        
        r = np.sqrt(np.array(model.dU.dat.data)**2).sum() / model.area
        r = torch.tensor(r, dtype=torch.float32) 
     
        return r
    

    @staticmethod
    def backward(ctx, grad_output):

        model = ctx.model  
        dU = ctx.dU

        model.dU.dat.data[0][:] = dU[0]
        model.dU.dat.data[1][:] = dU[1]
        model.dU.dat.data[2][:] = dU[2]
        model.dU.dat.data[3][:] = dU[3]

        dU = model.get_dU()

        return dU*grad_output, None, None, None