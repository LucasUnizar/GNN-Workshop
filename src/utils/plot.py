from scipy.io import savemat
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import wandb
import numpy as np
import scipy.io as sio
import json
import torch
import matplotlib

def plot_3D_scatter(z_net, z_gt, X, Y, save_dir='outputs/gifs/', name='Liver_actuator', with_wandb=True):
    """
    This function creates a 3D scatter plot comparing Data Driven MeshGraphs predictions with ground truth
    and saves the data as .mat files for later use in MATLAB.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    X (torch.Tensor): X-coordinates.
    Y (torch.Tensor): Y-coordinates.
    save_dir (str): Directory to save the plot.
    name (str): Name for the plot and file.

    Returns:
    None
    """
    T = z_net.shape[0]  # Number of snapshots

    # Ensure save directory exists
    save_path_mats = save_dir + '/mats/'
    os.makedirs(os.path.dirname(save_path_mats), exist_ok=True)

    # Save data for MATLAB
    mat_file_path = os.path.join(save_path_mats, f'{name}_data.mat')
    mat_data = {
        'z_net': z_net,
        'z_gt': z_gt,
        'X': X,
        'Y': Y
    }
    savemat(mat_file_path, mat_data)
    print(f"Data saved for MATLAB at {mat_file_path}")

    # Plot initialization
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # Set titles and labels
    ax1.set_title('GNN Prediction', fontsize=18, fontfamily='serif')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax2.set_title('Ground Truth', fontsize=18, fontfamily='serif')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

    # Calculate ranges
    tensor_max = max(z_gt.max(), z_net.max())
    tensor_min = min(z_gt.min(), z_net.min())
    z_min, z_max = tensor_min.item(), tensor_max.item()
    
    # Get X and Y ranges for proper scaling
    x_range = (X.min().item(), X.max().item())
    y_range = (Y.min().item(), Y.max().item())
    
    # Calculate aspect ratio based on data ranges
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_max - z_min
    
    # Set the plot box aspect ratio to match data proportions
    ax1.set_box_aspect([x_size, y_size, z_size])
    ax2.set_box_aspect([x_size, y_size, z_size])
    
    # Set z-limits
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)
    
    # Set x and y limits
    ax1.set_xlim(x_range[0], x_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax2.set_xlim(x_range[0], x_range[1])
    ax2.set_ylim(y_range[0], y_range[1])

    # Initial snapshot
    var_net0, var_gt0 = z_net[0].flatten().unsqueeze(1), z_gt[0].flatten().unsqueeze(1)
    sc1 = ax1.scatter(X, Y, var_net0, c=var_net0, cmap='plasma', vmax=z_max, vmin=z_min)
    sc2 = ax2.scatter(X, Y, var_gt0, c=var_gt0, cmap='plasma', vmax=z_max, vmin=z_min)

    # Colorbars
    cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.5, pad=0.1)
    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.1)
    cbar1.set_label(r'$u_{i}$', fontsize=14, labelpad=10)
    cbar2.set_label(r'$u_{i}$', fontsize=14, labelpad=10)

    # Animation
    def animate(snap):
        ax1.clear()
        ax2.clear()
        
        # Reapply all settings for each frame
        ax1.set_title('GNN Prediction', fontsize=18, fontfamily='serif')
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth', fontsize=18, fontfamily='serif')
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')
        
        # Reapply aspect ratio and limits
        ax1.set_box_aspect([x_size, y_size, z_size])
        ax2.set_box_aspect([x_size, y_size, z_size])
        ax1.set_zlim(z_min, z_max)
        ax2.set_zlim(z_min, z_max)
        ax1.set_xlim(x_range[0], x_range[1])
        ax1.set_ylim(y_range[0], y_range[1])
        ax2.set_xlim(x_range[0], x_range[1])
        ax2.set_ylim(y_range[0], y_range[1])

        var_net, var_gt = z_net[snap].flatten().unsqueeze(1), z_gt[snap].flatten().unsqueeze(1)
        sc1 = ax1.scatter(X, Y, var_net, c=var_net, cmap='plasma', vmax=z_max, vmin=z_min)
        sc2 = ax2.scatter(X, Y, var_gt, c=var_gt, cmap='plasma', vmax=z_max, vmin=z_min)

        return fig,

    anim = FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=5)

    # Save as gif
    save_path = save_dir + '/gifs/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gif_file_path = os.path.join(save_path, f'{name}.gif')
    anim.save(gif_file_path, writer=writergif)

    if os.path.exists(gif_file_path) and with_wandb:
        wandb.log({f'{name}_3Dscatter': wandb.Image(gif_file_path)})
        print(f"GIF saved at {gif_file_path}")
    elif os.path.exists(save_dir):
        print(f"GIF saved at {save_dir}")
    else:
        print(f"File not found at {save_dir}")
    plt.close()

def plot_3D_scatter_elliptic(z_net, z_gt, save_dir='outputs/gifs/', name='Liver_actuator', with_wandb=True):
    """
    This function creates a 3D scatter plot comparing Data Driven MeshGraphs predictions with ground truth
    and saves the data as .mat files for later use in MATLAB.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    X (torch.Tensor): X-coordinates.
    Y (torch.Tensor): Y-coordinates.
    save_dir (str): Directory to save the plot.
    name (str): Name for the plot and file.

    Returns:
    None
    """
    T = z_net.shape[0]  # Number of snapshots

    # Ensure save directory exists
    # Save as gif
    save_path_mats = save_dir + '/results/mats/'
    os.makedirs(os.path.dirname(save_path_mats), exist_ok=True)

    # Save data for MATLAB
    mat_file_path = os.path.join(save_path_mats, f'{name}_data.mat')
    mat_data = {
        'z_net': z_net,
        'z_gt': z_gt,
    }
    savemat(mat_file_path, mat_data)
    print(f"Data saved for MATLAB at {mat_file_path}")

    # Plot initialization
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.set_title('GNN Prediction', fontsize=18, fontfamily='serif')
    ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
    ax2.set_title('Ground Truth', fontsize=18, fontfamily='serif')
    ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

    # Adjust ranges
    pos_gt, pos_net = z_gt[:, :, :3], z_net[:, :, :3]
    sigma_gt, sigma_net = z_gt[:, :, -1:], z_net[:, :, -1:]
    tensor_max = max(pos_gt.max(), pos_net.max())
    tensor_min = min(pos_gt.min(), pos_net.min())
    z_min, z_max = tensor_min.item(), tensor_max.item()
    sigma_max, sigma_min = np.max(sigma_gt),  np.min(sigma_gt)
    ax1.set_zlim(z_min, z_max)
    ax2.set_zlim(z_min, z_max)

    # Initial snapshot
    var_net0, var_gt0 = sigma_net[0].flatten(), sigma_gt[0].flatten()
    sc1 = ax1.scatter(pos_net[0, :, 0], pos_net[0, :, 1], pos_net[0, :, 2], c=var_net0, cmap='plasma', vmax=sigma_max, vmin=sigma_min)
    sc2 = ax2.scatter(pos_gt[0, :, 0], pos_gt[0, :, 1], pos_gt[0, :, 2], c=var_gt0, cmap='plasma', vmax=sigma_max, vmin=sigma_min)

    # Colorbars
    cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.5, pad=0.1)
    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.5, pad=0.1)
    cbar1.set_label(r'$\sigma_{i}$', fontsize=14, labelpad=10)
    cbar2.set_label(r'$\sigma_{i}$', fontsize=14, labelpad=10)

    # Animation
    def animate(snap):
        ax1.clear()
        ax2.clear()
        ax1.set_zlim(z_min, z_max)
        ax2.set_zlim(z_min, z_max)
        ax1.set_title('GNN Prediction', fontsize=18, fontfamily='serif')
        ax1.set_xlabel('X'), ax1.set_ylabel('Y'), ax1.set_zlabel('Z')
        ax2.set_title('Ground Truth', fontsize=18, fontfamily='serif')
        ax2.set_xlabel('X'), ax2.set_ylabel('Y'), ax2.set_zlabel('Z')

        # var_net, var_gt = z_net[snap].flatten(), z_gt[snap].flatten()
        # sc1 = ax1.scatter(X, Y, var_net, c=var_net, cmap='plasma', vmax=z_max, vmin=z_min)
        # sc2 = ax2.scatter(X, Y, var_gt, c=var_gt, cmap='plasma', vmax=z_max, vmin=z_min)

        var_net_snap, var_gt_snap = sigma_net[snap].flatten(), sigma_gt[snap].flatten()
        sc1 = ax1.scatter(pos_net[snap, :, 0], pos_net[snap, :, 1], pos_net[snap, :, 2], c=var_net_snap, cmap='plasma', vmax=sigma_max, vmin=sigma_min)
        sc2 = ax2.scatter(pos_gt[snap, :, 0], pos_gt[snap, :, 1], pos_gt[snap, :, 2], c=var_gt_snap, cmap='plasma', vmax=sigma_max, vmin=sigma_min)

        return fig,

    anim = FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=5)

    # Save as gif
    save_path = save_dir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    gif_file_path = os.path.join(save_path, f'{name}.gif')
    anim.save(gif_file_path, writer=writergif)

    if os.path.exists(gif_file_path) and with_wandb:
        wandb.log({f'{name}_3Dscatter': wandb.Image(gif_file_path)})
        print(f"GIF saved at {gif_file_path}")
    elif os.path.exists(save_dir):
        print(f"GIF saved at {save_path}")
    else:
        print(f"File not found at {save_dir}")
    plt.close()


def plot_error(z_net, z_gt, X, Y, save_dir='outputs/gifs/', name='Error_plot'):
    """
    This function plots the error between GNN predictions and ground truth.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    X (torch.Tensor): X-coordinates.
    Y (torch.Tensor): Y-coordinates.
    save_dir (str): Directory to save the plot.
    name (str): Name for the plot and file.

    Returns:
    None
    """
    T = z_net.shape[0]  # Number of snapshots

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('Prediction Error', fontsize=18, fontfamily='serif')
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Error')

    # Animation
    def animate(snap):
        ax.clear()
        ax.set_title('Prediction Error', fontsize=18, fontfamily='serif')
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Error')

        error = (z_net[snap] - z_gt[snap])**2
        sc = ax.scatter(X, Y, error, c=error, cmap='coolwarm', vmax=error.max(), vmin=error.min())
        return sc,

    anim = FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=25)

    save_dir = save_dir + name + '_error.gif'
    anim.save(save_dir, writer=writergif)
    wandb.log({f'{name}_error': wandb.Image(save_dir)})
    plt.close()


def plot_combined(z_net, z_gt, X, Y, save_dir='outputs/gifs/', name='Combined_plot', with_wandb=True):
    """
    This function overlays GNN predictions and ground truth in a single plot for comparison,
    with proper axis scaling to maintain true proportions between axes.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    X (torch.Tensor): X-coordinates.
    Y (torch.Tensor): Y-coordinates.
    save_dir (str): Directory to save the plot.
    name (str): Name for the plot and file.

    Returns:
    None
    """
    T = z_net.shape[0]  # Number of snapshots

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('Prediction vs Ground Truth', fontsize=18, fontfamily='serif')
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
    
    # Calculate ranges
    tensor_max = max(z_gt.max(), z_net.max())
    tensor_min = min(z_gt.min(), z_net.min())
    z_min, z_max = tensor_min.item(), tensor_max.item()
    
    # Get X and Y ranges for proper scaling
    x_range = (X.min().item(), X.max().item())
    y_range = (Y.min().item(), Y.max().item())
    
    # Calculate aspect ratio based on data ranges
    x_size = x_range[1] - x_range[0]
    y_size = y_range[1] - y_range[0]
    z_size = z_max - z_min
    
    # Set the plot box aspect ratio to match data proportions
    ax.set_box_aspect([x_size, y_size, z_size])
    
    # Set axis limits
    ax.set_zlim(z_min, z_max)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])

    # Animation
    def animate(snap):
        ax.clear()
        ax.set_title('Prediction vs Ground Truth', fontsize=18, fontfamily='serif')
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        
        # Reapply aspect ratio and limits for each frame
        ax.set_box_aspect([x_size, y_size, z_size])
        ax.set_zlim(z_min, z_max)
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])

        var_net = z_net[snap].flatten().unsqueeze(1)
        var_gt = z_gt[snap].flatten().unsqueeze(1)

        sc_net = ax.scatter(X, Y, var_net, c='blue', label='Prediction', alpha=0.6)
        sc_gt = ax.scatter(X, Y, var_gt, c='red', label='Ground Truth', alpha=0.6)
        ax.legend()

        return sc_net, sc_gt

    anim = FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=5)

    save_path = save_dir + '/gifs/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dir = save_path + name + '.gif'

    anim.save(save_dir, writer=writergif)
    if os.path.exists(save_dir) and with_wandb:
        wandb.log({f'{name}_3Dscatter': wandb.Image(save_dir)})
        print(f"GIF saved at {save_dir}")
    elif os.path.exists(save_dir):
        print(f"GIF saved at {save_dir}")
    else:
        print(f"File not found at {save_dir}")
    plt.close()


def extract_last_components(mssg, filename="outputs/gifs/last_components.mat"):
    """
    This function extracts the last component from each index of the input message list 
    and saves it as a matrix of shape [nodes x 28].

    Parameters:
    mssg (list of np.ndarray): List of NumPy arrays containing variable values.
    filename (str): The name of the file to save the data as a .mat file.

    Returns:
    np.ndarray: A matrix of shape [nodes x 28] where each row is a message with the last component at each time step.
    """
    # Extract the last component from each array in the list
    last_components = [arr[-1] for arr in mssg]

    # Convert the list of last components to a 2D NumPy array
    last_components_matrix = np.stack(last_components)

    # Save the matrix as a .mat file
    sio.savemat(filename, {'last_components': last_components_matrix})

    return last_components_matrix


def plot_mssg(mssg, x, y, save_dir='outputs/gifs/', name='message_passing', idx=0):
    """
    This function creates an animated 3D plot with three subplots, each representing message values at fixed z_gt positions 
    over T frames. Colorbars are added to each subplot.

    Parameters:
    mssg (list of torch.Tensor): List of message tensors containing variable values.
    z_gt (torch.Tensor): Fixed 3D positions (ground truth).
    n (torch.Tensor): Binary mask for distinguishing data subsets.
    save_dir (str): Directory to save the plot.
    name (str): Base name for the saved plot.
    idx (int): Index for naming the saved plot.

    Returns:
    None
    """
    
    # Number of frames in the animation
    T = len(mssg[0])
    mssg_0, mssg_10, mssg_20 = mssg[0], mssg[11], mssg[21]
    mssg0_t, mssg10_t, mssg20_t = [0]*T, [0]*T, [0]*T
    for t in range(T):
        mssg0_t[t] = mssg_0[t] / mssg_0[0]
        mssg10_t[t] = mssg_10[t] / mssg_10[0]
        mssg20_t[t] = mssg_20[t] / mssg_20[0]

    # Set up figure and 3D subplots
    fig = plt.figure(figsize=(18, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    
    # Titles for subplots
    ax1.set_title('Snapshot 0', fontsize=16)
    ax2.set_title('Snapshot 10', fontsize=16)
    ax3.set_title('Snapshot 20', fontsize=16)

    # Common axis labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=20, azim=40)
        ax.grid()
        # Hide ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Determine variable range for color consistency
    X, Y, Z = x, y, np.zeros_like(x)
    z_min0, z_max0 = np.min(np.concatenate(mssg0_t)), np.max(np.concatenate(mssg0_t))
    z_min10, z_max10 = np.min(np.concatenate(mssg10_t)), np.max(np.concatenate(mssg10_t))
    z_min20, z_max20 = np.min(np.concatenate(mssg20_t)), np.max(np.concatenate(mssg20_t))
    z_min = min(z_min0, z_min10, z_min20)
    z_max = max(z_max0, z_max10, z_max20)
    
    # Bounding box settings
    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

    # Initialize scatter plots for first frame and colorbars
    s1 = ax1.scatter(X, Y, Z, c=mssg0_t[0], cmap='jet', vmin=z_min0, vmax=z_max0)
    s2 = ax2.scatter(X, Y, Z, c=mssg10_t[0], cmap='jet', vmin=z_min10, vmax=z_max10)
    s3 = ax3.scatter(X, Y, Z, c=mssg20_t[0], cmap='jet', vmin=z_min20, vmax=z_max20)
    fig.colorbar(s1, ax=ax1, shrink=0.5, aspect=10)
    fig.colorbar(s2, ax=ax2, shrink=0.5, aspect=10)
    fig.colorbar(s3, ax=ax3, shrink=0.5, aspect=10)

    def animate(t):
        ax1.cla()
        ax2.cla()
        ax3.cla()
        
        # Setup titles and labels for each frame
        ax1.set_title('Message Level 0')
        ax2.set_title('Message Level 10')
        ax3.set_title('Message Level 20')
        
        # Scatter plot with values from mssg, but positions from z_gt
        s1 = ax1.scatter(X, Y, Z, c=mssg0_t[t], cmap='jet', vmin=z_min0, vmax=z_max0)
        s2 = ax2.scatter(X, Y, Z, c=mssg10_t[t], cmap='jet', vmin=z_min10, vmax=z_max10)
        s3 = ax3.scatter(X, Y, Z, c=mssg20_t[t], cmap='jet', vmin=z_min20, vmax=z_max20)
        
        # Bounding box
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax1.plot([xb], [yb], [zb], 'w')
            ax2.plot([xb], [yb], [zb], 'w')
            ax3.plot([xb], [yb], [zb], 'w')
        
        return fig,

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=T, repeat=False)
    writergif = animation.PillowWriter(fps=6)

    # Save animation as a GIF
    save_path = f"{save_dir}/{name}_mssg_{idx}.gif"
    anim.save(save_path, writer=writergif)
    plt.close()

def boxplot_error(test_error, save_path='outputs/test_statistics/', name='Liver_actuator', error='RMSE'):
    """
    Create a boxplot for a single error type in the test set,
    with individual data points plotted as dots to the left of the boxplot.

    Parameters:
    test_error (list): List containing testing errors.
    save_path (str): Path to save the boxplot image. Default is 'outputs/test_statistics/'.
    name (str): Name of the plot for saving purposes.

    Returns:
    None
    """
    # Create a new figure
    fig, ax = plt.subplots(figsize=(3, 4))  # Longer and thinner figure
    test_error = np.array(test_error)

    # Define the boxplot position and scatter offset
    position = 1
    scatter_offset = -0.15  # Offset for scatter points

    # Define color for the boxplot and scatter points
    color = 'darkblue'

    # Create the boxplot
    bp = ax.boxplot(test_error, positions=[position], widths=0.15, patch_artist=True,
                    boxprops=dict(facecolor='white', edgecolor=color, linewidth=2), showfliers=False)

    # Add scatter points
    ax.scatter(np.full_like(test_error, position + scatter_offset), test_error, color=color, alpha=0.6, edgecolor='w', zorder=3)

    # Set y-axis to log scale
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=12)  # Smaller y-axis labels

    # Customize font size and family
    ax.set_xlabel(error, fontsize=16, fontfamily='serif')

    # Adjust x-axis ticks and labels
    ax.set_xticks([position])
    ax.set_xticklabels([r'Test Error'], fontsize=16, fontfamily='serif', fontweight='bold', fontstyle='italic')

    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=1.0)

    # Customize the plot background
    ax.set_facecolor('whitesmoke')

    # Save the boxplot image with a transparent background
    save_path = save_path + name + '_boxplot.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', transparent=True)
    plt.close()

def plot_frame(z_net, z_gt, X, Y, frame_idx=0, save_dir='outputs/images/', name='Combined_plot'):
    """
    This function overlays GNN predictions and ground truth for a specific frame.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    X (torch.Tensor): X-coordinates.
    Y (torch.Tensor): Y-coordinates.
    frame_idx (int): Index of the frame to visualize.
    save_dir (str): Directory to save the plot.
    name (str): Name for the plot and file.

    Returns:
    None
    """
    # Validate frame index
    if frame_idx < 0 or frame_idx >= z_net.shape[0]:
        raise ValueError(f"Invalid frame_idx: {frame_idx}. Must be between 0 and {z_net.shape[0] - 1}.")

    # Extract data for the specified frame
    var_net = z_net[frame_idx].flatten()
    var_gt = z_gt[frame_idx].flatten()

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_title('Prediction vs Ground Truth', fontsize=18, fontfamily='serif')
    ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')

    # Scatter plots for predictions and ground truth
    ax.scatter(X, Y, var_net, c='blue', label='Prediction', alpha=0.6)
    ax.scatter(X, Y, var_gt, c='red', label='Ground Truth', alpha=0.6)
    ax.legend()

    # Save the plot
    save_path = f"{save_dir}{name}_frame_{frame_idx}.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

    # Optionally, log to wandb
    try:
        import wandb
        wandb.log({f'{name}_frame_{frame_idx}': wandb.Image(save_path)})
    except ImportError:
        print("wandb not available. Skipping wandb logging.")

    plt.close()

def plot_mse_rollout(mse_list_of_rollouts, name, checkpoint_dir=None, save=False, with_wandb=True, save_path='outputs'):
    """
    This function plots the mean squared error (MSE) for each time step in multiple rollouts,
    and logs the mean of log2(MSE) at powers of 2 indices across all rollouts to Weights & Biases (wandb).

    Parameters:
    mse_list_of_rollouts (list of lists): List of MSE values for each time step in multiple rollouts.
    name (str): Name for the plot.

    Returns:
    None
    """

    # Create a new figure for plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Array to accumulate log2(MSE) values at each power of 2 index
    max_length = max(len(mse) for mse in mse_list_of_rollouts)
    log2_indices = [2**i for i in range(int(np.log2(max_length)) + 1)]
    log2_values = {idx: [] for idx in log2_indices}

    # Plot each rollout in the mse_list_of_rollouts
    for i, mse_list in enumerate(mse_list_of_rollouts):
        steps = range(1, len(mse_list) + 1)  # Steps are the x-axis values from 1 to the length of each list
        ax.plot(steps, mse_list)

        # Compute log2(MSE) values for powers of 2 indices
        for idx in log2_indices:
            if idx <= len(mse_list):
                mse_value = mse_list[idx - 1]
                log2_value = mse_value
                log2_values[idx].append(log2_value)

    # Calculate mean log2(MSE) across all rollouts for each power-of-2 index
    mean_log2_values = {idx: np.mean(values) if values else -np.inf for idx, values in log2_values.items()}

    # Add labels and title to the plot
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel('MSE', fontsize=14)
    ax.set_title(f'Mean Squared Error for {name} - All Rollouts', fontsize=16)

    # Save the plot locally
    save_path = save_path + '/plots'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path = f'{save_path}/{name}_mse_rollouts.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    if with_wandb:
        # Log the plot to wandb
        wandb.log({"mse_rollout_plot": wandb.Image(save_path)})
        # Log the mean log2(MSE) values to wandb
        for idx, mean_log_value in mean_log2_values.items():
            wandb.log({
                f'mean_log2_mse_index_{idx}': mean_log_value
            })

     # Save the mse_list_of_rollouts to a JSON file
    if save:
        mse_save_path = str(checkpoint_dir) + '\\test_mse_rollouts.json'
        # Convert Tensors to lists or scalars
        mse_list_of_rollouts_serializable = [
            t.tolist() if isinstance(t, torch.Tensor) else t
            for t in mse_list_of_rollouts
        ]
        with open(mse_save_path, 'w') as f:
            json.dump(mse_list_of_rollouts_serializable, f, indent=4)
        if with_wandb:
            wandb.save(mse_save_path)
        print(f"MSE rollouts saved to {mse_save_path}")


def plot_mse_rollout_mean_std(mse_list_of_rollouts, name, save=False, output_dir=None, save_path='./outputs'):
    """
    This function plots the mean and standard deviation of the Mean Squared Error (MSE) values 
    for each time step across multiple rollouts.

    Parameters:
    mse_list_of_rollouts (list of lists): List of MSE values for each time step in multiple rollouts.
    name (str): Name for the plot.
    save (bool): Whether to save the plot as a file. Default is False.
    output_dir (str): Directory to save the plot if save is True. Default is None.

    Returns:
    None
    """
    # Convert to NumPy array for easier computation
    mse_array = np.array(mse_list_of_rollouts)

    # Compute mean and standard deviation along the rollouts (axis=0)
    mean_mse = np.mean(mse_array, axis=0).squeeze()
    std_mse = np.std(mse_array, axis=0).squeeze()

    # Create the plot
    time_steps = np.arange(1, len(mean_mse) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, mean_mse, label='Mean MSE', color='blue')
    plt.fill_between(time_steps, 
                     mean_mse - std_mse, 
                     mean_mse + std_mse, 
                     color='blue', alpha=0.3, label='Std Deviation')

    # Add labels and title
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.title(f'Mean and Std of MSE for {name}', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot and data if required
    if save:
        if output_dir is None:
            output_dir = save_path + '/plots'
        os.makedirs(output_dir, exist_ok=True)

        # Save the plot
        save_path = os.path.join(output_dir, f'{name}.png')
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

        # Save the mean, std, and time vectors to a JSON file
        data_save_path = os.path.join(output_dir, f'{name}_mse_data.json')
        data_to_save = {
            "time_steps": time_steps.tolist(),
            "mean_mse": mean_mse.tolist(),
            "std_mse": std_mse.tolist()
        }
        with open(data_save_path, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        print(f"Data saved to {data_save_path}")

    # Show the plot
    plt.close()

def plot_frame_2d(z_net, z_gt, X, Y, frame_idx=0, save_dir='outputs/plots/', name='2d_plot', wandb_flag=True):
    """
    This function creates 2D subplots comparing GNN predictions and ground truth for a specific frame.

    Parameters:
    z_net (torch.Tensor): Predictions from the GNN.
    z_gt (torch.Tensor): Ground truth data.
    X (torch.Tensor): X-coordinates.
    Y (torch.Tensor): Y-coordinates.
    frame_idx (int): Index of the frame to visualize.
    save_dir (str): Directory to save the plot.
    name (str): Name for the plot and file.

    Returns:
    None
    """
    # Validate frame index
    if frame_idx < 0 or frame_idx >= z_net.shape[0]:
        raise ValueError(f"Invalid frame_idx: {frame_idx}. Must be between 0 and {z_net.shape[0] - 1}.")

    # Convert tensors to numpy arrays if they're not already
    X_np = X.cpu().numpy() if hasattr(X, 'cpu') else np.array(X)
    Y_np = Y.cpu().numpy() if hasattr(Y, 'cpu') else np.array(Y)
    var_net = z_net[frame_idx].flatten().cpu().numpy() if hasattr(z_net, 'cpu') else np.array(z_net[frame_idx].flatten())
    var_gt = z_gt[frame_idx].flatten().cpu().numpy() if hasattr(z_gt, 'cpu') else np.array(z_gt[frame_idx].flatten())

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Prediction vs Ground Truth - Frame {frame_idx}', fontsize=16)
    
    # Ground truth plot
    gt_plot = ax1.scatter(X_np, Y_np, c=var_gt, cmap='plasma')
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(gt_plot, ax=ax1, shrink=0.6)
    
    # Prediction plot
    net_plot = ax2.scatter(X_np, Y_np, c=var_net, cmap='plasma')
    ax2.set_title('Prediction')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    fig.colorbar(net_plot, ax=ax2, shrink=0.6)
    
    # Error plot (difference between prediction and ground truth)
    error = var_net - var_gt
    error_plot = ax3.scatter(X_np, Y_np, c=error, cmap='coolwarm')
    ax3.set_title('Prediction Error (Pred - GT)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    fig.colorbar(error_plot, ax=ax3, shrink=0.6)
    
    plt.tight_layout()

    # Save the plot
    save_path = f"{save_dir}{name}_frame_{frame_idx}_2d.png"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {save_path}")

    # Optionally, log to wandb
    if wandb_flag:
        try:
            wandb.log({f'{name}_frame_{frame_idx}_2d': wandb.Image(save_path)})
        except ImportError:
            print("wandb not available. Skipping wandb logging.")

    plt.close()