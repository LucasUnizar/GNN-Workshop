import torch
from torch.utils.data import Dataset
import os


class MGN_GraphDataset(Dataset):
    def __init__(self, data_dir, rollout=False, ratio=1):
        # self.args = args
        self.rollout = rollout
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.data_files = self.data_files[:int(ratio * len(self.data_files))]
        self.data = []

        if self.rollout:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                datalist = torch.load(file_path, weights_only=False)  # Load as DataList
                for data in datalist:
                    data.face = torch.transpose(data.face[:, 1:], 1, 0)
                    data.x = torch.cat([data.x[:, :3], data.x[:, -1:]], dim=-1).float()
                    data.y = torch.cat([data.y[:, :3], data.y[:, -1:]], dim=-1).float()
                    data.pos = data.pos.float()
                    data.mesh_pos = data.mesh_pos.float()

                self.data.append(datalist)  # Append each data to the main list
        else:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                datalist = torch.load(file_path, weights_only=False)  # Load as DataList
                for data in datalist:
                    data.face = torch.transpose(data.face[:, 1:], 1, 0)
                    data.x = torch.cat([data.x[:, :3], data.x[:, -1:]], dim=-1).float()
                    data.y = torch.cat([data.y[:, :3], data.y[:, -1:]], dim=-1).float()
                    data.pos = data.pos.float()
                    data.mesh_pos = data.mesh_pos.float()

                    self.data.append(data)  # Append each data to the main list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class MPNN_GraphDataset(Dataset):
    def __init__(self, data_dir, rollout=False, ratio=1):
        # self.args = args
        self.rollout = rollout
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.data_files = self.data_files[:int(ratio * len(self.data_files))]
        self.data = []

        if self.rollout:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                datalist = torch.load(file_path, weights_only=False)  # Load as DataList
                for data in datalist:
                    if hasattr(data, 'u'):
                        data.num_nodes = data.u.shape[0]
                        data.n = data.n  # Add a batch dimension
                    else:
                        print(f"Warning: Could not infer num_nodes for an item in {file}")
                self.data.append(datalist)  # Append each data to the main list
        else:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                datalist = torch.load(file_path, weights_only=False)  # Load as DataList
                for data in datalist:
                    if hasattr(data, 'u'):
                        data.num_nodes = data.u.shape[0]
                        data.n = data.n  # Add a batch dimension
                    else:
                        print(f"Warning: Could not infer num_nodes for an item in {file}")
                    self.data.append(data)  # Append each data to the main list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Poisson_GraphDataset(Dataset):
    def __init__(self, data_dir, rollout=False, ratio=1):
        # self.args = args
        self.rollout = rollout
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.data_files = self.data_files[:int(ratio * len(self.data_files))]
        if self.rollout:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                data = torch.load(file_path, weights_only=False)  # Load as DataList
                data.num_nodes = data.u.shape[0]
                datalist = [data]
                self.data.append(datalist)  # Append each data to the main list
        else:
            self.data = []
            for file in self.data_files:
                file_path = os.path.join(self.data_dir, file)
                data = torch.load(file_path, weights_only=False)  # Load as DataList
                data.num_nodes = data.u.shape[0]
                self.data.append(data)  # Append each data to the main list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

