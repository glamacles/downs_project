import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from modulus.launch.utils import load_checkpoint, save_checkpoint
import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.append('../model/')
sys.path.append('../emulator/')
from model import SpecFO
from emulator import FNOEmulator
writer = SummaryWriter()

ice_model = SpecFO()
data = np.load('solution_data/data1.npy')

# Create input features
B = data[:,0,:,:]
H = data[:,1,:,:]
X = np.stack([B, H], axis=1)

# Output features 
Y = data[:,2:,:,:]

# Create datsets
N_train = 3500
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

train_dataset = TensorDataset(X[0:N_train], Y[0:N_train])
validation_dataset = TensorDataset(X[N_train:], Y[N_train:])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

# Emulator model 
fno = FNOEmulator()
fno.cuda()

optimizer = torch.optim.Adam(fno.parameters(), lr=1e-5)

ckpt_args = {
    "path": f"./checkpoints",
    "optimizer": optimizer,
    "models": fno,
}

# Number of epochs
epochs = 1000

for epoch in range(epochs):
    avg_loss = 0.
    num = 0

    # Training
    fno.train()
    for i, (x,y) in enumerate(train_loader):

        B = x[:,0,:,:]
        H = x[:,1,:,:]

        x = x.cuda()
        y = y.cuda()
    
        optimizer.zero_grad()
        y_fno = fno(x)

        loss = torch.mean(torch.sqrt((y - y_fno)**2 + 1e-10))

        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        num += 1

    avg_loss /= num
    print(epoch, 'avg_loss', avg_loss)
    writer.add_scalar("Loss/train", avg_loss, epoch)

    # Checkpoint
    if epoch % 50 == 0:
        save_checkpoint(**ckpt_args, epoch=epoch)

    # Validation 
    if epoch % 25 == 0:
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
                    plt.close("all")
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