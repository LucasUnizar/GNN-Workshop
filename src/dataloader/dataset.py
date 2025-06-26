import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt


class MPNN_GraphDataset(Dataset):
    def __init__(self, data_dir, rollout=False, ratio=1):
        self.rollout = rollout
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.data_files = self.data_files[:int(ratio * len(self.data_files))]
        self.data = []

        if self.rollout:
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                datalist = torch.load(file_path, weights_only=False)
                for data in datalist:
                    if hasattr(data, 'u'):
                        data.num_nodes = data.u.shape[0]
                        data.n = data.n
                self.data.append(datalist)
        else:
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                datalist = torch.load(file_path, weights_only=False)
                for data in datalist:
                    if hasattr(data, 'u'):
                        data.num_nodes = data.u.shape[0]
                        data.n = data.n
                    self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def plot_first_and_last(self, file_index=0, frame_index=5):
        if not self.rollout:
            print("This method is only available in rollout mode.")
            return

        if file_index >= len(self.data):
            print("Invalid file_index")
            return

        datalist = self.data[file_index]

        first = datalist[0]
        last = datalist[frame_index]

        def plot_data(data, title, cmap='plasma'):
            x = data.X.squeeze().numpy()
            y = data.Y.squeeze().numpy()
            u = data.u.squeeze().numpy()

            edge_index = data.edge_index
            edges = edge_index.t().tolist()  # list of (src, dst) pairs
            edge_set = set(map(tuple, edges))

            # Plot edges with color based on directionality
            for src, dst in edge_set:
                if (dst, src) in edge_set:
                    color = 'black'  # bidirected edge
                else:
                    color = 'red'    # unidirected edge
                plt.plot([x[src], x[dst]], [y[src], y[dst]], color=color, linewidth=0.5, zorder=1)

            # Plot nodes
            plt.scatter(x, y, c=u, cmap=cmap, s=15, zorder=2)
            plt.colorbar(label='u')
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_data(first, f"Initial Frame - File {file_index}", cmap='plasma')
        plt.subplot(1, 2, 2)
        plot_data(last, f"Frame {frame_index} - File {file_index}", cmap='plasma')
        plt.tight_layout()
        plt.show()


class Poisson_GraphDataset(Dataset):
    def __init__(self, data_dir, rollout=False, ratio=1):
        self.rollout = rollout
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.data_files = self.data_files[:int(ratio * len(self.data_files))]
        self.data = []

        if self.rollout:
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                data = torch.load(file_path, weights_only=False)
                data.num_nodes = data.u.shape[0]
                self.data.append([data])
        else:
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                data = torch.load(file_path, weights_only=False)
                data.num_nodes = data.u.shape[0]
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def plot_first_and_last(self, file_index=0):
        if not self.rollout:
            print("This method is only available in rollout mode.")
            return

        if file_index >= len(self.data):
            print("Invalid file_index")
            return

        datalist = self.data[file_index]
        first = datalist[0]
        last = datalist[-1]

        def plot_data(data, title, cmap='plasma', u1_flag=False):
            x = data.X.squeeze().numpy()
            y = data.Y.squeeze().numpy()
            if u1_flag:
                u = data.u1.squeeze().numpy()
            else:
                u = data.u.squeeze().numpy()

            edge_index = data.edge_index
            edges = edge_index.t().tolist()
            edge_set = set(map(tuple, edges))

            for src, dst in edge_set:
                if (dst, src) in edge_set:
                    color = 'black'
                else:
                    color = 'red'
                plt.plot([x[src], x[dst]], [y[src], y[dst]], color=color, linewidth=0.5, zorder=1)

            plt.scatter(x, y, c=u, cmap=cmap, s=15, zorder=2)
            plt.colorbar(label='u')
            plt.title(title)
            plt.xlabel('X')
            plt.ylabel('Y')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plot_data(first, f"Initial state - File {file_index}", cmap='plasma')
        plt.subplot(1, 2, 2)
        plot_data(last, f"Final state - File {file_index}", cmap='plasma', u1_flag=True)
        plt.tight_layout()
        plt.show()
