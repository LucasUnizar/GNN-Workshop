from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from src.dataloader.dataset import  MPNN_GraphDataset, Poisson_GraphDataset
from pathlib import Path


class GraphDataModule(LightningDataModule):
    def __init__(self, dataset_dir, batch_size=8, num_workers=0, ratio=1., dataset_type='poisson'):
        super().__init__()
        self.data_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ratio = ratio
        self.dataset_type = dataset_type
        
        # Map dataset type strings to their corresponding classes
        self.dataset_classes = {
            'gnn': MPNN_GraphDataset,
            'poisson': Poisson_GraphDataset
        }
        
        # Validate dataset type
        if self.dataset_type not in self.dataset_classes:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be one of {list(self.dataset_classes.keys())}")

    def _get_dataset(self, path, **kwargs):
        """Helper method to get the appropriate dataset instance"""
        return self.dataset_classes[self.dataset_type](path, **kwargs)

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = self._get_dataset(self.data_dir / 'train', ratio=self.ratio)
            self.val_dataset = self._get_dataset(self.data_dir / 'valid')
            self.valid_rollout_dataset = self._get_dataset(self.data_dir / 'valid', rollout=True)
            self.valid_rollout_dataloader = DataLoader(
                self.valid_rollout_dataset, 
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False
            )
        if stage == 'test':
            self.test_dataset = self._get_dataset(self.data_dir / 'test', rollout=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            num_workers=1, 
            shuffle=False
        )

    def plot_first_and_last_rollout(self, file_index=0, stage='valid'):
        """Plot the first and last data of a rollout sequence."""
        if stage == 'valid':
            if hasattr(self, 'valid_rollout_dataset'):
                self.valid_rollout_dataset.plot_first_and_last(file_index)
            else:
                print("Rollout dataset not initialized. Run setup('fit') first.")
        elif stage == 'test':
            if hasattr(self, 'test_dataset'):
                self.test_dataset.plot_first_and_last(file_index)
            else:
                print("Test dataset not initialized. Run setup('test') first.")
        else:
            raise ValueError("Stage must be 'valid' or 'test'")
