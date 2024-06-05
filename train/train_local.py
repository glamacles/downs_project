import firedrake as fd
from model import SpecFO
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from modulus.models.fno import FNO
from model import PINNLoss
import torch.nn as nn
from modulus.launch.utils import load_checkpoint, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

ice_model = SpecFO()
data = np.load('data.npy')

# Create input features
B = data[:,0,:,:]
H = data[:,1,:,:]
S = B+H
S_x = np.gradient(S, axis=2)
S_y = np.gradient(S, axis=1)
X = np.stack([B, H, S_x, S_y], axis=1)

# Output features 
Y = data[:,2:,:,:]

# Create datsets
N_train = 2000
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)
x_scale = X.std(axis=(0,2,3))[np.newaxis,:,np.newaxis,np.newaxis]
y_scale = Y.std(axis=(0,2,3))[np.newaxis,:,np.newaxis,np.newaxis]


train_dataset = TensorDataset(X[0:N_train], Y[0:N_train])
validation_dataset = TensorDataset(X[N_train:], Y[N_train:])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

class Model(nn.Module):
            def __init__(self, x_scale, y_scale):
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
                    padding=9,
                ).cuda()

                self.x_scale = x_scale.cuda()
                self.y_scale = y_scale.cuda()

            def forward(self, x):
                x = x / self.x_scale
                y = self.fno(x[:,1:,:,:])
                y = y*self.y_scale
                return y


fno = Model(torch.tensor(x_scale).cuda(), torch.tensor(y_scale))  
optimizer = torch.optim.Adam(fno.parameters(), lr=5e-6)


ckpt_args = {
    "path": f"./checkpoints",
    "optimizer": optimizer,
    "models": fno,
}
#loaded_epoch = 0
loaded_epoch = load_checkpoint(**ckpt_args)


# Number of epochs
epochs = 1000
pinn_loss = PINNLoss().apply

for epoch in range(epochs):
    avg_loss = 0.
    num = 0

    fno.train()
    for i, (x,y) in enumerate(train_loader):

        B = x[0,0,:,:]
        H = x[0,1,:,:]

        x = x.cuda()
        y = y.cuda()
    
        optimizer.zero_grad()
        y_emulator = fno(x).cpu()

        loss = pinn_loss(y_emulator, B, H, ice_model)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        num += 1

    avg_loss /= num
    print(epoch, 'avg_loss', avg_loss)
    writer.add_scalar("Loss/train", avg_loss, epoch)

    if epoch % 20 == 0:
        save_checkpoint(**ckpt_args, epoch=epoch)

    # Validation 
    if epoch % 10 == 0:
        with torch.no_grad():
            validation_loss = 0.
            num = 0
            for i, (x,y) in enumerate(validation_loader):
                x = x.cuda()
                y = y.cuda()
                y_fno = fno(x)
                loss = torch.mean(torch.sqrt((y - y_fno)**2 + 1e-10))
                validation_loss += loss.item()

                if i == 0:
                    #plt.close("all")
                    fig = plt.figure(figsize=(8,12))

                    y = y.cpu()
                    y_fno = y_fno.cpu()
                    plt.subplot(4,2,1)
                    plt.imshow(y_fno[0,0,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,2)
                    plt.imshow(y[0,0,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,3)
                    plt.imshow(y_fno[0,1,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,4)
                    plt.imshow(y[0,1,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,5)
                    plt.imshow(y_fno[0,2,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,6)
                    plt.imshow(y[0,2,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,7)
                    plt.imshow(y_fno[0,3,:,:])
                    plt.colorbar()

                    plt.subplot(4,2,8)
                    plt.imshow(y[0,3,:,:])
                    plt.colorbar()

                    plt.tight_layout()
                    plt.savefig(f'logs/test_{epoch}.png')

                num += 1

            validation_loss /= num
            writer.add_scalar("Loss/validation", avg_loss, epoch)
            print('validation_loss', validation_loss / num)
   
writer.flush()