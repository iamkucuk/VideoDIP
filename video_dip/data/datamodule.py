import pytorch_lightning as pl

try:
    from video_dip.data.dataset import VideoDIPDataset
    from video_dip.models.optical_flow import Farneback, RAFT, RAFTModelSize
    
except ImportError:
    import sys
    import os
    # Add parent of parent directory to path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

    from video_dip.data.dataset import VideoDIPDataset
    from video_dip.models.optical_flow import Farneback, RAFT, RAFTModelSize

from torch.utils.data import DataLoader

class VideoDIPDataModule(pl.LightningDataModule):
    def __init__(self, input_path, batch_size, num_workers, flow_model = Farneback(), target_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_path = input_path
        self.target_path = target_path

        self.flow_model = flow_model

    def dump_optical_flow(self, path=None):
        """
        Dump the optical flow to a folder. 
        """
        import os
        from tqdm.auto import tqdm
        from torchvision.utils import save_image

        flow_folder = path if path is not None else os.path.join(self.input_path, "flow")
        if not os.path.exists(flow_folder):
            os.makedirs(flow_folder)
        
        dataset = VideoDIPDataset(
            self.input_path,
            transforms=VideoDIPDataset.default_flow_transforms()
        )
        
        for i in tqdm(range(1, len(dataset))):
            img1 = dataset[i - 1]['input']
            img2 = dataset[i]['input']
            base_name = dataset[i]['filename']
            flow = self.flow_model(img1, img2)
            save_image(flow / 255, os.path.join(flow_folder, base_name))

    def setup(self, stage=None):
        self.dump_optical_flow()
        if stage == 'fit' or stage is None:
            self.train_dataset = VideoDIPDataset()
            self.val_dataset = VideoDIPDataset()
        if stage == 'test' or stage is None:
            self.test_dataset = VideoDIPDataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
if __name__ == '__main__':

    module = VideoDIPDataModule("datasets/GT/pair1", batch_size=2, num_workers=8, flow_model=RAFT(RAFTModelSize.LARGE))
    module.dump_optical_flow('flow_outputs')
    