from __future__ import print_function
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms
import pickle
from VAE_model import VAE, loss_function


batch_size = 100

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

model = VAE(x_dim=784, h_dim=400, z_dim=20)


optimizer = optim.Adam(model.parameters())
global_step = 0
loss_list = []

def train(epoch, model):
    model.train()
    train_loss = 0
    global global_step
    print('epoch: ', epoch, ', batch size:', batch_size)
    for batch_idx, (data, _) in enumerate(train_loader):
        # data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logVar = model(data)
        loss = loss_function(recon_batch, data, mu, logVar)
        # print(loss)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


        if batch_idx % 100 == 0:
            print('Train Epoch: {}, step: {}, [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    global_step,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(data)
                )
            )
            loss_list.append(loss.item() / len(data))
            torch.save(model.state_dict(), 'checkpoints/checkpoint_'+str(global_step))
            global_step += 1


    print('====> Epoch: {} Average loss: {:.6f}'.format(
            epoch,
            train_loss / len(train_loader.dataset)
        )
    )


def test(epoch, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            # data = Variable(data)
            recon_batch, mu, logVar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logVar).item()

    test_loss /= len(test_loader.dataset)
    print('Epoch', epoch, '====> Test set loss: {:.6f}\n'.format(test_loss))


for epoch in range(0, 50):
    train(epoch, model)
    test(epoch, model )

torch.save(model.state_dict(), 'VAE_model_withMSELoss')
with open('MSE_LossList', 'wb') as f:
    pickle.dump(loss_list, f)
    f.close()
