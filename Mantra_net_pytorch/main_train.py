import torch
from network import MantraNet
from torch import optim, nn
import glob
import os
import re

def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


save_dir = './nets'
data_dir=''

if __name__ == "__main__":
    model = MantraNet()
    initial_epoch = findLastCheckpoint(save_dir=save_dir)
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model = torch.load(os.path.join(
            save_dir, 'model_%03d.pth' % initial_epoch))
    model.train()

    batch_size = 64
    batches = 1000
    num_epochs = 100
    lr = 1e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        
