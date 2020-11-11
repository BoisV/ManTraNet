from skimage import io
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []
        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        image_path = self.root_dir + self.names_list[index].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exit!')
            return None
        image = io.imread(image_path)
        label = int(self.names_list[index].split(' ')[1])
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


if __name__ == "__main__":
    train_dataset = MyDataset(root_dir='./dataset/train',
                              names_file='./dataset/train/train.txt',
                              transform=None)
    print(len(train_dataset))
