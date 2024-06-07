import torch
from modulus.models.fno import FNO
import torch.nn as nn

"""
FNO emulator.
 Input Features: B, H
 Outputs: Ubar0, Ubar1, Udef0, Udef1
"""
class FNOEmulator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fno = FNO(
            in_channels=3,
            out_channels=4,
            decoder_layers=1,
            decoder_layer_size=32,
            dimension=2,
            latent_channels=32,
            num_fno_layers=4,
            num_fno_modes=12,
            padding=9
        ).cuda()

    def forward(self, x):

        """
        x contains just B and H. Gradients are computed
        as additional features.
        """
       
        B = x[:,0,:,:]
        H = x[:,1,:,:]
        S = B + H
        Sx = torch.gradient(S, axis=2)[0]
        Sy = torch.gradient(S, axis=1)[0]

        x = torch.stack([
            H / 100., Sx / 100., Sy / 100.
        ], dim=1)

        y = 100. * self.fno(x)
        return y


