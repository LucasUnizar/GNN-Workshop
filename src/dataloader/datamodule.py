from src.dataloader.dataset import MGN_GraphDataset, MPNN_GraphDataset, Poisson_GraphDataset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

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
            'meshgraph': MGN_GraphDataset,
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
            # Load datasets from disc into RAM memory
            self.train_dataset = self._get_dataset(self.data_dir / 'train', ratio=self.ratio)
            self.val_dataset = self._get_dataset(self.data_dir / 'valid')

            # set rollout for validation split
            self.valid_rollout_dataset = self._get_dataset(self.data_dir / 'valid', rollout=True)
            self.valid_rollout_dataloader = DataLoader(
                self.valid_rollout_dataset, 
                batch_size=1,
                num_workers=self.num_workers,
                shuffle=False
            )
            
        if stage == 'test':
            # Load datasets from disc into RAM memory
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

    # ================= Visualization Methods =================
    
    def visualize_sample(self, split: str = 'train', idx: int = 0, **kwargs):
        """
        Visualize a sample from the specified dataset split.
        
        Args:
            split (str): Which split to visualize from ('train', 'val', 'test')
            idx (int): Index of the sample to visualize
            **kwargs: Additional arguments passed to the dataset's visualize method
        """
        dataset = self._get_split_dataset(split)
        if dataset is None:
            print(f"No {split} dataset available. Call setup() first.")
            return
            
        if idx >= len(dataset):
            print(f"Index {idx} out of range for {split} dataset (size: {len(dataset)})")
            return
            
        # Call the appropriate visualization method based on dataset type
        if self.dataset_type == 'meshgraph':
            dataset.visualize_graph(idx, **kwargs)
        elif self.dataset_type == 'gnn':
            dataset.visualize_graph(idx, **kwargs)
        elif self.dataset_type == 'poisson':
            dataset.visualize_solution(idx, **kwargs)

    def visualize_batch(self, split: str = 'train', batch_idx: int = 0, samples_per_batch: int = 4):
        """
        Visualize multiple samples from a batch.
        
        Args:
            split (str): Which split to visualize from ('train', 'val', 'test')
            batch_idx (int): Which batch to visualize
            samples_per_batch (int): How many samples to show from the batch
        """
        dataloader = self._get_split_dataloader(split)
        if dataloader is None:
            print(f"No {split} dataloader available. Call setup() first.")
            return
            
        # Get the batch
        for i, batch in enumerate(dataloader):
            if i == batch_idx:
                break
        else:
            print(f"Batch index {batch_idx} out of range for {split} dataloader")
            return
            
        # Determine how many samples to show (don't exceed batch size)
        num_samples = min(samples_per_batch, len(batch))
        
        # Create subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]
            
        for i in range(num_samples):
            ax = axes[i]
            data = batch[i]
            
            if self.dataset_type == 'meshgraph':
                pos = data.pos.numpy()
                features = data.x.numpy()
                if pos.shape[1] >= 3:
                    # 3D plot
                    ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
                    color = np.linalg.norm(features[:, :3], axis=1)
                    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, cmap='viridis', s=10)
                    ax.set_title(f'Sample {i}')
                else:
                    # 2D plot
                    ax.scatter(pos[:, 0], pos[:, 1], c=features[:, 0], cmap='viridis', s=10)
                    ax.set_title(f'Sample {i}')
                    
            elif self.dataset_type == 'gnn':
                if hasattr(data, 'pos'):
                    pos = data.pos.numpy()
                    if pos.shape[1] >= 3:
                        ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
                        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=10)
                    else:
                        ax.scatter(pos[:, 0], pos[:, 1], s=10)
                ax.set_title(f'Sample {i}')
                
            elif self.dataset_type == 'poisson':
                if hasattr(data, 'pos'):
                    pos = data.pos.numpy()
                    u = data.u.numpy()
                    if pos.shape[1] >= 3:
                        ax = fig.add_subplot(1, num_samples, i+1, projection='3d')
                        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=u, cmap='viridis', s=10)
                    else:
                        ax.scatter(pos[:, 0], pos[:, 1], c=u, cmap='viridis', s=10)
                ax.set_title(f'Sample {i}')
                
        plt.tight_layout()
        plt.show()

    def visualize_feature_distribution(self, split: str = 'train', feature_idx: int = 0, num_samples: int = 100):
        """
        Visualize the distribution of a specific feature across the dataset.
        
        Args:
            split (str): Which split to visualize ('train', 'val', 'test')
            feature_idx (int): Index of the feature to visualize
            num_samples (int): Number of samples to include in the distribution
        """
        dataset = self._get_split_dataset(split)
        if dataset is None:
            print(f"No {split} dataset available. Call setup() first.")
            return
            
        if hasattr(dataset, 'visualize_feature_distribution'):
            dataset.visualize_feature_distribution(feature_idx, num_samples)
        else:
            print(f"Feature distribution visualization not implemented for {self.dataset_type} dataset")

    def _get_split_dataset(self, split: str):
        """Helper method to get the dataset for a given split"""
        if split == 'train':
            return getattr(self, 'train_dataset', None)
        elif split == 'val':
            return getattr(self, 'val_dataset', None)
        elif split == 'test':
            return getattr(self, 'test_dataset', None)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")

    def _get_split_dataloader(self, split: str):
        """Helper method to get the dataloader for a given split"""
        if split == 'train':
            return getattr(self, 'train_dataloader', None)()
        elif split == 'val':
            return getattr(self, 'val_dataloader', None)()
        elif split == 'test':
            return getattr(self, 'test_dataloader', None)()
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")