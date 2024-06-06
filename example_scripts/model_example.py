import os
import sys
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../model/')
import numpy as np
import firedrake as fd
from model import SpecFO

# Model defaults to a 125 x 125 km domain with 1.25 km res
model = SpecFO()

# Generate random ice sheet geometry
B, H = model.get_geometry(
    B0 = 1000.,
    S0 = 3000,
)

# Solve for velocity
model.set_field(model.B, B)
model.set_field(model.H, H)
model.solver.solve()

ubar0, ubar1, udef0, udef1, s0, s1 = model.get_velocity()

plt.figure(figsize=(12,8))
plt.subplot(4, 2, 1)
plt.title('Bed (m)')
plt.imshow(B)
plt.colorbar()

plt.subplot(4, 2, 2)
plt.title('Surface (m)')
plt.imshow(B+H)
plt.colorbar()

plt.subplot(4,2,3)
plt.title('X-component Depth Averaged Velocity (m/a)')
plt.imshow(ubar0)
plt.colorbar()

plt.subplot(4,2,4)
plt.title('Y-component Depth Averaged Velocity (m/a)')
plt.imshow(ubar1)
plt.colorbar()

plt.subplot(4,2,5)
plt.title('X-component Deformational Velocity (m/a)')
plt.imshow(s0)
plt.colorbar()

plt.subplot(4,2,6)
plt.title('Y-component Deformational Velocity (m/a)')
plt.imshow(s1)
plt.colorbar()

plt.subplot(4,2,7)
plt.title('Depth Avg. Velocity (m/a)')
plt.imshow(np.sqrt(ubar0**2 + ubar1**2))
plt.colorbar()

plt.subplot(4,2,8)
plt.title('Surface Velocity (m/a)')
plt.imshow(np.sqrt(s0**2 + s1**2))
plt.colorbar()

plt.tight_layout()
plt.show()