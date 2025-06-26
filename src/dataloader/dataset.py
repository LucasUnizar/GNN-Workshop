import torch
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

class MGN_GraphDataset(Dataset):
    # ... (keep your existing __init__, __len__, __getitem__ methods)
    
    def visualize_graph(self, idx, show_edges=True, show_mesh=True, show_prediction=False):
        """Visualize a graph from the dataset.
        
        Args:
            idx (int): Index of the graph to visualize
            show_edges (bool): Whether to show edges
            show_mesh (bool): Whether to show mesh positions
            show_prediction (bool): Whether to show predictions (y) alongside inputs (x)
        """
        data = self[idx]
        fig = plt.figure(figsize=(12, 6))
        
        # Create 3D axis
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot node positions
        pos = data.pos.numpy()
        x = data.x.numpy()
        
        # Color nodes by velocity magnitude (assuming first 3 features are velocity)
        if x.shape[1] >= 3:
            velocity = x[:, :3]
            color = np.linalg.norm(velocity, axis=1)
            sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, cmap='viridis', s=20)
            plt.colorbar(sc, label='Velocity magnitude')
        else:
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=20)
        
        # Plot edges if requested
        if show_edges and hasattr(data, 'edge_index'):
            edge_index = data.edge_index.numpy()
            for i in range(edge_index.shape[1]):
                start = pos[edge_index[0, i]]
                end = pos[edge_index[1, i]]
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                       'gray', alpha=0.5, linewidth=0.5)
        
        # Plot mesh positions if requested
        if show_mesh and hasattr(data, 'mesh_pos'):
            mesh_pos = data.mesh_pos.numpy()
            ax.scatter(mesh_pos[:, 0], mesh_pos[:, 1], mesh_pos[:, 2], 
                      c='red', s=10, alpha=0.5, label='Mesh positions')
        
        # Plot predictions if requested
        if show_prediction and hasattr(data, 'y'):
            y = data.y.numpy()
            pred_pos = pos + y[:, :3]  # Assuming y contains displacement
            ax.scatter(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 
                      c='green', s=20, alpha=0.7, label='Predicted positions')
        
        ax.set_title(f'Graph {idx}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        set_axes_equal(ax)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def visualize_feature_distribution(self, feature_idx=0, num_samples=100):
        """Visualize distribution of a specific feature across the dataset.
        
        Args:
            feature_idx (int): Index of feature to visualize (0-3 based on your x/y tensors)
            num_samples (int): Number of samples to use for histogram
        """
        features = []
        for i in range(min(num_samples, len(self))):
            data = self[i]
            features.extend(data.x[:, feature_idx].numpy())
        
        plt.figure(figsize=(8, 5))
        plt.hist(features, bins=50)
        plt.title(f'Distribution of feature {feature_idx}')
        plt.xlabel('Feature value')
        plt.ylabel('Frequency')
        plt.show()

class MPNN_GraphDataset(Dataset):
    # ... (keep your existing __init__, __len__, __getitem__ methods)
    
    def visualize_graph(self, idx, feature_to_plot='u', show_edges=True):
        """Visualize a graph from the dataset.
        
        Args:
            idx (int): Index of the graph to visualize
            feature_to_plot (str): Which feature to visualize ('u', 'x', etc.)
            show_edges (bool): Whether to show edges
        """
        data = self[idx]
        
        if not hasattr(data, feature_to_plot):
            print(f"Data doesn't have attribute {feature_to_plot}")
            return
            
        features = getattr(data, feature_to_plot)
        if features.dim() > 2:
            print("Can't visualize - features have more than 2 dimensions")
            return
            
        plt.figure(figsize=(10, 6))
        
        if features.shape[1] == 1:
            # Scalar feature
            pos = data.pos.numpy() if hasattr(data, 'pos') else None
            if pos is not None and pos.shape[1] >= 2:
                # 2D or 3D plot
                if pos.shape[1] == 2:
                    sc = plt.scatter(pos[:, 0], pos[:, 1], c=features.numpy(), cmap='viridis')
                    plt.colorbar(sc)
                else:
                    ax = plt.axes(projection='3d')
                    sc = ax.scatter3D(pos[:, 0], pos[:, 1], pos[:, 2], c=features.numpy(), cmap='viridis')
                    plt.colorbar(sc)
                    set_axes_equal(ax)
            else:
                # No position info, just plot feature distribution
                plt.hist(features.numpy(), bins=50)
                plt.title(f'Distribution of {feature_to_plot}')
        elif features.shape[1] == 2:
            # 2D vector feature
            pos = data.pos.numpy() if hasattr(data, 'pos') else None
            if pos is not None and pos.shape[1] >= 2:
                plt.quiver(pos[:, 0], pos[:, 1], 
                          features[:, 0].numpy(), features[:, 1].numpy())
            else:
                plt.scatter(features[:, 0].numpy(), features[:, 1].numpy())
        elif features.shape[1] >= 3:
            # 3D vector feature
            pos = data.pos.numpy() if hasattr(data, 'pos') else None
            if pos is not None and pos.shape[1] >= 3:
                ax = plt.axes(projection='3d')
                # Color by magnitude
                color = np.linalg.norm(features.numpy(), axis=1)
                sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=color, cmap='viridis')
                plt.colorbar(sc, label=f'{feature_to_plot} magnitude')
                set_axes_equal(ax)
            else:
                # Plot first 3 dimensions
                ax = plt.axes(projection='3d')
                ax.scatter(features[:, 0].numpy(), 
                          features[:, 1].numpy(), 
                          features[:, 2].numpy())
                set_axes_equal(ax)
        
        # Plot edges if requested and available
        if show_edges and hasattr(data, 'edge_index'):
            if pos is not None and pos.shape[1] >= 2:
                edge_index = data.edge_index.numpy()
                for i in range(edge_index.shape[1]):
                    start = pos[edge_index[0, i]]
                    end = pos[edge_index[1, i]]
                    if pos.shape[1] == 2:
                        plt.plot([start[0], end[0]], [start[1], end[1]], 
                                'gray', alpha=0.3, linewidth=0.5)
                    else:
                        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                               'gray', alpha=0.3, linewidth=0.5)
        
        plt.title(f'Graph {idx} - Feature: {feature_to_plot}')
        plt.tight_layout()
        plt.show()
    
    def visualize_graph_structure(self, idx):
        """Visualize just the graph structure using networkx."""
        data = self[idx]
        if not hasattr(data, 'edge_index'):
            print("No edge_index attribute found")
            return
            
        G = to_networkx(data)
        plt.figure(figsize=(8, 8))
        nx.draw(G, node_size=20, width=0.5)
        plt.title(f'Graph {idx} structure')
        plt.show()

class Poisson_GraphDataset(Dataset):
    # ... (keep your existing __init__, __len__, __getitem__ methods)
    
    def visualize_solution(self, idx, show_mesh=True):
        """Visualize the Poisson equation solution.
        
        Args:
            idx (int): Index of the solution to visualize
            show_mesh (bool): Whether to show the mesh
        """
        data = self[idx]
        
        if not hasattr(data, 'u'):
            print("No solution (u) found in data")
            return
            
        u = data.u.numpy()
        
        fig = plt.figure(figsize=(12, 6))
        
        if hasattr(data, 'pos') and data.pos.shape[1] >= 2:
            pos = data.pos.numpy()
            if pos.shape[1] == 2:
                # 2D plot
                plt.scatter(pos[:, 0], pos[:, 1], c=u, cmap='viridis')
                plt.colorbar(label='Solution (u)')
                
                if show_mesh and hasattr(data, 'face'):
                    faces = data.face.numpy()
                    for face in faces:
                        triangle = pos[face]
                        plt.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
                                [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]],
                                'r-', alpha=0.3)
                
                plt.title(f'Poisson Solution {idx}')
                plt.xlabel('X')
                plt.ylabel('Y')
                
            else:
                # 3D plot
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=u, cmap='viridis')
                plt.colorbar(sc, label='Solution (u)')
                
                if show_mesh and hasattr(data, 'face'):
                    faces = data.face.numpy()
                    for face in faces:
                        triangle = pos[face]
                        ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
                               [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]],
                               [triangle[0, 2], triangle[1, 2], triangle[2, 2], triangle[0, 2]],
                               'r-', alpha=0.3)
                
                ax.set_title(f'Poisson Solution {idx}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                set_axes_equal(ax)
        else:
            # No position information, just plot the solution values
            plt.plot(u)
            plt.title(f'Poisson Solution Values {idx}')
            plt.xlabel('Node index')
            plt.ylabel('Solution value (u)')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_mesh(self, idx):
        """Visualize just the mesh structure."""
        data = self[idx]
        
        if not hasattr(data, 'face'):
            print("No face information found")
            return
            
        if not hasattr(data, 'pos'):
            print("No position information found")
            return
            
        pos = data.pos.numpy()
        faces = data.face.numpy()
        
        fig = plt.figure(figsize=(8, 6))
        
        if pos.shape[1] == 2:
            # 2D mesh
            for face in faces:
                triangle = pos[face]
                plt.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
                        [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]],
                        'b-', alpha=0.5)
            plt.scatter(pos[:, 0], pos[:, 1], c='r', s=10)
            plt.title(f'Mesh {idx}')
            plt.xlabel('X')
            plt.ylabel('Y')
        else:
            # 3D mesh
            ax = fig.add_subplot(111, projection='3d')
            for face in faces:
                triangle = pos[face]
                ax.plot([triangle[0, 0], triangle[1, 0], triangle[2, 0], triangle[0, 0]],
                        [triangle[0, 1], triangle[1, 1], triangle[2, 1], triangle[0, 1]],
                        [triangle[0, 2], triangle[1, 2], triangle[2, 2], triangle[0, 2]],
                        'b-', alpha=0.5)
            ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', s=10)
            ax.set_title(f'Mesh {idx}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            set_axes_equal(ax)
        
        plt.tight_layout()
        plt.show()