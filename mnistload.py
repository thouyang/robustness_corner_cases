import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as Data
import scipy.io as scio
import pathlib

def data_pre(batch_size=128, retrain=False):
    # batch_size=128

    ##mnist
    train_data = datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./mnist', train=False, download=True, transform=transforms.ToTensor())

    # TrainD = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0,shuffle=True)
    TestD = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=0,shuffle=False)
    D0 = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=0, shuffle=False)  #

    if not retrain:
        trdata0 = train_data
    else:
    # #### deleting outliers
        D1 = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), num_workers=0, shuffle=False)  #
        dataiter = iter(D1)
        images, labels = dataiter.next()
        file = pathlib.Path("results/nc_idx.mat")
        if not file.exists ():
            print ("\033[91m"+"Corner case data index does not exist, please detect corner cases firstly"+"\033[0m")
        else:
            print("\033[92m"+"Corner case data has been detected"+"\033[0m")
            index=scio.loadmat(file)
            ind=index['nc'][0]
            trdata0=Data.TensorDataset(images[ind],labels[ind])

    TrainD=torch.utils.data.DataLoader(trdata0, batch_size=batch_size,num_workers=0,shuffle=True)#

    return TrainD, TestD, D0




