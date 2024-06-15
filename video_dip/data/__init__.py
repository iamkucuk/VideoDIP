from data import create_datset, DIPDataset, visualize_frame
from torch.utils.data import Dataset, DataLoader

dipdataset = create_datset("video_dip/data/sora.mp4", target_resolution = (100,100))
data_loader = DataLoader(dipdataset, batch_size=1, num_workers=8)
img = next(iter(dipdataset))
print(img.shape)
visualize_frame(img)