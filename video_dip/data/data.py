import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch import Tensor
from torchvision.io import read_video
from torchvision import transforms
import tqdm 

class DIPDataset(Dataset):
    """
    
    """
    def __init__(self, video_path, target_resolution=None):
        self.video_path = video_path
        self.target_resolution = target_resolution

        # load video as series of frames
        self.frames = read_video(video_path)[0]

        if target_resolution:
            self.resize_transform = transforms.Resize((target_resolution[0], target_resolution[1]))

    def __getitem__(self, idx):
        permuted_frame = self.frames[idx].permute(2,0,1)
        
        if self.target_resolution is not None:
            return self.resize_transform(permuted_frame)

        return permuted_frame

    

def create_datset(video_path, target_resolution = None):
    dipdataset = DIPDataset(video_path=video_path, target_resolution=target_resolution)
    return dipdataset

def visualize_frame(tensor:Tensor):
    """
    
    C,H,W
    
    """

    tensor = tensor.permute(1, 2, 0).to("cpu")
    # Convert the tensor to numpy array
    tensor = tensor.numpy()
    # Normalize the tensor to [0, 1]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    plt.imshow(tensor)
    plt.axis('off')  # Turn off axis labels
    plt.show()
    return tensor