from torch_geometric.data import Data
import enum
import datetime
import json
from pathlib import Path
import wandb
from scipy.io import savemat
import os
import numpy as np

class NodeTypeDP(enum.IntEnum):
    NORMAL = 0
    WALL_BOUNDARY = 1
    ACTUATOR = 2

def set_constants(args):
    args.ratio = 1.0
    args.shared_mp = False
    args.noise = 0.0001

    # Set the run name based on the model and parameters
    args.plots_flag = True
    args.plot_worst = False
    return args

def transform_data(data, z_net_list, z_gt_list, data_out, edge_index=[], edge_contact=[], faces=[], node_type=[],
                   export_path="mssg_hops_data"):
    """
    Transforms and normalizes simulation data, then exports the results to .mat files.

    Args:
        data (list): A list of simulations organized as 
                     [simulation][frame][node][dimension].
        z_net_list (list): A list of z_net values corresponding to each simulation.
        z_gt_list (list): A list of z_gt values corresponding to each simulation.
        export_path (str): The folder where the .mat files will be saved. 
                           If the folder does not exist, it will be created.

    Returns:
        tuple: 
            - transformed_data (list): Organized as 
              [simulation][frame x nodes x dimension (np tensor)].
            - normalized_data (list): Normalized values organized as 
              [simulation][frame x nodes].
    """
    if len(data) != len(z_net_list) or len(data) != len(z_gt_list):
        raise ValueError("data, z_net_list, and z_gt_list must have the same length.")

    transformed_data = []
    normalized_data = []
    output_data = []

    # Ensure the export directory exists
    if not os.path.exists(export_path):
        os.makedirs(export_path)
        print(f"Created folder: {export_path}")

    for sim_idx, (simulation, z_net, z_gt) in enumerate(zip(data, z_net_list, z_gt_list)):
        sim_transformed = []
        sim_normalized = []

        if not simulation:
            raise ValueError(f"Simulation {sim_idx} data cannot be empty.")

        for frame in simulation:
            # Ensure frame is an np array
            frame0_norm = np.linalg.norm(frame[0], ord=2, axis=1, keepdims=True)
            frame_tensor = np.array(frame)

            # Compute normalized frame (element-wise division by frame0 norm)
            norm_frame = np.linalg.norm(frame_tensor, ord=2, axis=2, keepdims=True) / frame0_norm
            sim_normalized.append(norm_frame.squeeze())  # Remove singleton dimensions
            sim_transformed.append(frame_tensor)

        # Stack frames into a single np array for the simulation
        sim_transformed = np.stack(sim_transformed)
        sim_normalized = np.stack(sim_normalized)

        transformed_data.append(sim_transformed)
        normalized_data.append(sim_normalized)

        # Export to .mat file
        mat_filename = os.path.join(export_path, f"simulation_{sim_idx+1}.mat")
        if len(edge_index) == 0:
            savemat(mat_filename, {
                "transformed_data": sim_transformed,
                "normalized_data": sim_normalized,
                "z_net": z_net,
                "z_gt": z_gt
            })
        else:
            savemat(mat_filename, {
                "transformed_data": sim_transformed,
                "normalized_data": sim_normalized,
                "z_net": z_net,
                "z_gt": z_gt,
                "edge_index": np.array(edge_index[sim_idx]),
                "faces": np.array(faces[sim_idx]),
                "node_type": np.array(node_type[sim_idx])
                #"edge_contact": np.array(edge_contact[sim_idx])
            })
        print(f"Exported Simulation {sim_idx+1} data to {mat_filename}")
    
    # Export output data
    for out_idx, (simulation, z_net, z_gt) in enumerate(zip(data_out, z_net_list, z_gt_list)):
        sim_output = []
        for frame in simulation:
            # Ensure frame is an np array
            frame_tensor = np.array(frame)
            sim_output.append(frame_tensor.squeeze())
        sim_output = np.stack(sim_output)
        output_data.append(sim_output)
        mat_filename = os.path.join(export_path, f"output_simulation_{out_idx+1}.mat")
        savemat(mat_filename, {
            "output_data": sim_output,
            "z_net": z_net,
            "z_gt": z_gt
        })
        print(f"Exported Output Simulation {out_idx+1} data to {mat_filename}")



def set_run_directory(args, safe_mode=True):
    # Generate a unique name for the run directory based on current timestamp and arguments
    name = (f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_MPSteps={args.mp_steps}'
            f'_sharedMP={args.shared_mp}_layers={args.layers}_hidden={args.hidden}_batchsize={args.batch_size}_seed={args.seed}')

    # Create a Path object for the run directory
    chckp_path = Path(f'outputs/runs/{name}')
    # Create the directory if it doesn't exist, including parent directories
    chckp_path.mkdir(exist_ok=True, parents=True)

    if safe_mode:
        # Save the configuration as JSON in the run directory
        with open(chckp_path / 'config.json', 'w') as jsonfile:
            json.dump(vars(args), jsonfile, indent=4)

    # Return the path to the run directory
    return chckp_path, name


def decompose_graph(graph):
    return (graph.x, graph.edge_index)


def copy_geometric_data(graph):
    node_attr, edge_index = decompose_graph(graph)
    return Data(x=node_attr, edge_index=edge_index)


def decompose_meshgraph_graph(graph):
    return (graph.x, graph.edge_attr, graph.edge_index, graph.edge_contact_attr, graph.edge_contact_index)


def copy_meshgraph_graph(graph):
    node_attr, edge_mesh_attr, edge_index, edge_contact_attr, edge_contact_index = decompose_meshgraph_graph(graph)
    return Data(x=node_attr, edge_attr=edge_mesh_attr, edge_index=edge_index, edge_contact_attr=edge_contact_attr, edge_contact_index=edge_contact_index)