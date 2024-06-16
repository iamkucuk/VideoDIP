import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from PIL import Image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']
VID_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv']

class VideoDIPDataset(Dataset):
    """
    A custom dataset class for loading video frames or images for VideoDIP.

    Args:
        path (str): Path to the video file or directory containing images.
        target_path (str): Path to the directory containing the target frames or images.
        flow_path (str): Path to the directory containing the optical flow frames.
        transforms (torchvision.transforms.Compose): Transforms to be applied to the frames or images.       

    Raises:
        FileNotFoundError: If the specified path does not exist.
        ValueError: If the specified path is not a valid video file or does not contain any images.

    Attributes:
        frames (list): List of paths to the frames or images.

    Methods:
        default_transforms: Returns the default transforms to be applied to the frames or images.
    """

    def __init__(self, input_path, target_path = None, flow_path = None, transforms=None) -> None:
        assert input_path is not None, "Path to video file or images folder is required"
        assert isinstance(input_path, str), "Path must be a string"

        self.input_frames = self._get_frames(input_path)
        self.target_frames = self._get_frames(target_path) if target_path is not None else None
        self.optical_flow_frames = glob(os.path.join(flow_path, "*.npy")) if flow_path is not None else None
        if self.optical_flow_frames is not None:
            self.optical_flow_frames.sort()

        self.transforms = transforms if transforms is not None else self.default_transforms()

    def _get_frames(self, path):
        """
        Returns the paths to the frames or images in the specified directory.

        Args:
            path (str): Path to the directory containing the frames or images.

        Returns:
            list: List of paths to the frames or images.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path {path} does not exist")
        elif os.path.isdir(path):
            frames = []
            for ext in IMG_EXTENSIONS:
                frames.extend(glob(os.path.join(path, f"*{ext}")))
                if len(frames) > 0:
                    break
            if len(frames) == 0:
                raise ValueError(f"Directory {path} does not contain images")
        elif os.path.isfile(path):
            if path.endswith(tuple(VID_EXTENSIONS)):
                frames = self._dump_video(path)
            else:
                raise ValueError(f"File {path} is not a valid video file")
        # sort the frames
        frames.sort()
        return frames       

    def _dump_video(self, path):
        """
        Dumps the frames of the video into a temporary directory.

        Args:
            path (str): Path to the video file.

        Returns:
            list: List of paths to the dumped frames.
        """
        import cv2
        import tempfile

        temp_dir = tempfile.mkdtemp()
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(temp_dir, f"{len(frames)}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
        cap.release()
        return frames

    def _load_image(self, path):
        """
        Loads an image from the specified path.

        Args:
            path (str): Path to the image file.

        Returns:
            PIL.Image.Image: The loaded image.
        """
        return Image.open(path).convert("RGB")

    @staticmethod
    def default_transforms():
        """
        Returns the default transforms to be applied to the frames or images.

        Returns:
            torchvision.transforms.Compose: The default transforms.
        """
        from torchvision import transforms

        # TODO: Probably, we want to normalize the input image respecting the VGG training 
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((480, 856)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def default_flow_transforms():
        """
        Returns the default transforms to be applied to the optical flow frames.

        Returns:
            torchvision.transforms.Compose: The default transforms.
        """
        from torchvision import transforms

        return transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        if self.optical_flow_frames is None:
            return len(self.input_frames)
        return len(self.input_frames) - 2 # Omit the first two frames
    
    def __getitem__(self, idx):
        datum = {}
        if self.optical_flow_frames is not None:
            idx += 2 # Skip the first two frames
            # flow_frame = self._load_image(self.optical_flow_frames[idx])
            # prev_flow_frame = self._load_image(self.optical_flow_frames[idx - 1])
            # datum["flow"] = self.transforms(flow_frame)
            # datum["prev_flow"] = self.transforms(prev_flow_frame)
            datum['flow'] = torch.tensor(np.load(self.optical_flow_frames[idx - 1]))
            datum['prev_flow'] = torch.tensor(np.load(self.optical_flow_frames[idx - 2]))

        frame = self._load_image(self.input_frames[idx])
        datum.update({"input": self.transforms(frame), "filename": os.path.basename(self.input_frames[idx])})
        if self.target_frames is not None:
            target_frame = self._load_image(self.target_frames[idx])
            datum["target"] = self.transforms(target_frame)
        return datum

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import PIL
    dipdataset = VideoDIPDataset("video_dip/data/sora.mp4")
    data_loader = DataLoader(dipdataset, batch_size=1, num_workers=8)
    img = next(iter(dipdataset))['input']

    print(img.shape)
    
    PIL.Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).show()

    dipdataset = VideoDIPDataset("datasets/input/pair1", "datasets/GT/pair1")
    data_loader = DataLoader(dipdataset, batch_size=1, num_workers=8)
    batch = next(iter(dipdataset))
    
    img = batch['input']
    print(img.shape)

    PIL.Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).show()

    img = batch['target']
    print(img.shape)

    PIL.Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).show()

