import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt
from model import SpecFO

"""
Generate randomized synthetic ice sheets and solve for velocity.
"""

#data = np.load('project/solution_data/data.npy')

nx=75
ny=75
dx=125e3
dy=125e3

model = SpecFO(nx, ny, dx, dy)

N = 3000
S0 = np.random.uniform(1500., 4000., N)
sigmas = np.random.uniform(5., 15., N)
mid_offsets = 5000.*np.random.randn(N,2)

plot = False
data = []

for i in range(N):
    print(i)

    B, H = model.get_geometry(
        B0 = 1000.,
        S0 = S0[i],
        sigma=sigmas[i],
        mid_offset=mid_offsets[i]
    )
    model.set_field(model.B, B)
    model.set_field(model.H, H)

    model.solver.solve()

    ubar0 = model.get_field(model.U.sub(0))
    ubar1 = model.get_field(model.U.sub(1))
    udef0 = model.get_field(model.U.sub(2))
    udef1 = model.get_field(model.U.sub(3))

    X = np.stack([
        B, H, ubar0, ubar1, udef0, udef1
    ])
    data.append(X)

    if plot:
        plt.subplot(3,2,1)
        plt.imshow(B)
        plt.colorbar()

        plt.subplot(3,2,2)
        plt.imshow(H)
        plt.colorbar()

        plt.subplot(3,2,3)
        plt.imshow(ubar0)
        plt.colorbar()

        plt.subplot(3,2,4)
        plt.imshow(ubar1)
        plt.colorbar()

        plt.subplot(3,2,5)
        plt.imshow(udef0)
        plt.colorbar()

        plt.subplot(3,2,6)
        plt.imshow(udef1)
        plt.colorbar()

        plt.show()

    model.U.assign(model.U*0.)

data = np.array(data)
print(data.shape)
np.save('data.npy', data)