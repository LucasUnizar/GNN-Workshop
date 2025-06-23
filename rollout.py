from src.model.model import SimulatorGNN
from src.model.model_meshgraph import SimulatorMeshGraph
from src.dataloader.dataset import MPNN_GraphDataset, MGN_GraphDataset, Poisson_GraphDataset


from pathlib import Path
import pytorch_lightning as pl
import matplotlib
import torch
import argparse


pl.seed_everything(42, workers=True)
matplotlib.use('Agg')
torch.set_float32_matmul_precision('medium')
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MeshGraph Simulation')
    parser.add_argument('--pretrain_path', type=str, default=r"outputs\checkpoints\waves-low\6pass\models\topk1.pth")
    parser.add_argument('--dataset_dir', type=str, default=r'data\Waves_LowRes\dataset', help='Directory containing dataset')
    parser.add_argument('--split', type=str, default="test",  help='Dataset split to load simulation')
    parser.add_argument('--model', type=str, default="gnn", help='Unique identifier for the training run')
    parser.add_argument('--sim', type=int, default=0, help='Simulation index to make rollout')
    parser.add_argument('--mssg_flag', action="store_true", help="Flag to use message passing in the model")
    parser.add_argument('--full_rollout', action="store_false", help="Flag to plot the worst results during validation")

    args = parser.parse_args()

    pretrain_path = Path(args.pretrain_path)  # Convert path string to a Path object
    

    # Load a specific rollout simulation from the dataset
    if 'gnn' in args.model:
        simulator = SimulatorGNN(pretrain=pretrain_path, device='cpu', mssg_flag=args.mssg_flag)
        dataset = MPNN_GraphDataset(Path(args.dataset_dir) / args.split, rollout=True)
    elif 'meshgraph' in args.model:
        simulator = SimulatorMeshGraph(pretrain=pretrain_path, device='cpu', mssg_flag=args.mssg_flag)
        dataset = MGN_GraphDataset(Path(args.dataset_dir) / args.split, rollout=True)
    elif 'poisson' in args.model:
        simulator = SimulatorGNN(pretrain=pretrain_path, device='cpu', mssg_flag=args.mssg_flag)
        dataset = Poisson_GraphDataset(Path(args.dataset_dir) / args.split, rollout=True)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    

    rrmse_error = simulator.test_rollout(dataset, full_rollout=args.full_rollout, save_path=str(pretrain_path.parent.parent))
    print(f"Rollout RMSE error: {rrmse_error}")