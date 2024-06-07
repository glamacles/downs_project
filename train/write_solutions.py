import os
import sys
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../model/')
import numpy as np
import firedrake as fd
from model import SpecFO

"""
Generate randomized synthetic ice sheets and solve for velocity.
This script is used to generate training data for the emulator. 
"""

model = SpecFO()

# Generate 3000 examples
N = 4000
# Max elevation ranges from 1500 - 4000 m
S0 = np.random.uniform(2000., 4000., N)
# Random length scale parameter for the bed
sigmas = np.random.uniform(2.5, 25., N)
# This randomizes the location of the ice sheet in the domain
mid_offsets = 7500.*np.random.randn(N,2)

plot = False
data = []

for i in range(N):
    print(i)

    # Generate random ice sheet geometry
    B, H = model.get_geometry(
        B0 = 1500.,
        S0 = S0[i],
        sigma=sigmas[i],
        mid_offset=mid_offsets[i]
    )

    # Solve for velocity
    model.set_field(model.B, B)
    model.set_field(model.H, H)
    model.solver.solve()

    ubar0 = model.get_field(model.U.sub(0))
    ubar1 = model.get_field(model.U.sub(1))
    udef0 = model.get_field(model.U.sub(2))
    udef1 = model.get_field(model.U.sub(3))

    # Save inputs / outputs
    X = np.stack([
        B, H, ubar0, ubar1, udef0, udef1
    ])
    data.append(X)

    # Optionally plot
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

    # Reset velocity so velocity solver converges
    model.U.assign(model.U*0.)

# Save all the training data
data = np.array(data)
print(data.shape)
np.save('solution_data/data.npy', data)