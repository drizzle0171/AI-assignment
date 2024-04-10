import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MNIST(Dataset):
    """ MNIST dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Substract mean of 0.1307, and divide by std 0.3081
            - These preprocessing can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset = os.listdir(data_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], 
                                  std=[0.3081]),
            transforms.Resize((32, 32))
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        data_pth = os.path.join(self.data_dir, data)
        img = Image.open(data_pth)
        img = self.transform(img)
        label = torch.Tensor([int(data.split('_')[-1].split('.')[0])]).long().squeeze()
        return img, label

if __name__ == '__main__':
    dataset = MNIST('/data/MNIST/data/train')
    print(dataset[0])