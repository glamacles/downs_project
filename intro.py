import os
import sys
import matplotlib.pyplot as plt
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('model/')
sys.path.append('emulator/')
import numpy as np
import firedrake as fd
from model import SpecFO
from emulator import FNOEmulator
import torch

# FEM Model
model = SpecFO()

# FNO Emulator
emulator = FNOEmulator().cuda()
checkpoint = torch.load('emulator/weights.pt')
emulator.load_state_dict(checkpoint)
emulator.eval()

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


with torch.no_grad():
    x = np.stack([
        B, H
    ])[np.newaxis, :, :, :]
    x = torch.tensor(x, dtype=torch.float32).cuda()

    y = emulator(x).cpu()

    extent=[0., model.dx/1e3, 0., model.dy/1e3]
    plt.subplot(3,2,1)
    plt.title('Bed')
    plt.imshow(B, extent=extent)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    cbar = plt.colorbar()
    cbar.set_label('m', rotation=270)


    plt.subplot(3,2,2)
    plt.title('Surface')
    plt.imshow(B+H, extent=extent)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    cbar = plt.colorbar()
    cbar.set_label('m', rotation=270)

    plt.subplot(3,2,3)
    plt.title(f'Ubar0 (FEM)')
    plt.imshow(ubar0,  extent=extent)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    cbar = plt.colorbar()
    cbar.set_label('m/a', rotation=270)


    plt.subplot(3,2,4)
    plt.title(f'Ubar0 (FNO)')
    plt.imshow(y[0,0,:,:],  extent=extent)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    cbar = plt.colorbar()
    cbar.set_label('m/a', rotation=270)

    plt.subplot(3,2,5)
    plt.title(f'Ubar1 (FEM)')
    plt.imshow(ubar1,  extent=extent)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    cbar = plt.colorbar()
    cbar.set_label('m/a', rotation=270)

    plt.subplot(3,2,6)
    plt.title(f'Ubar1 (FNO)')
    plt.imshow(y[0,1,:,:],  extent=extent)
    plt.xlabel('x (km)')
    plt.ylabel('y (km)')
    cbar = plt.colorbar()
    cbar.set_label('m/a', rotation=270)

    plt.tight_layout()
    plt.show()
