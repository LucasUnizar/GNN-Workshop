import torch
import pytorch_lightning as pl
import matplotlib
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src.model.model import SimulatorGNN
from src.model.model_meshgraph import SimulatorMeshGraph
from src.model.callbacks import ModelSaveTopK

import wandb
from src.dataloader.datamodule import GraphDataModule
from src.utils.utils import set_run_directory


import argparse

matplotlib.use('Agg')
torch.set_float32_matmul_precision('medium')
device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MeshGraph Simulation')

    parser.add_argument('--batch_size', type=int, default=64, help='Number of samples per training batch')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--mp_steps', type=int, default=1, help='Number of message-passing steps in the GNN')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers in the model')
    parser.add_argument('--hidden', type=int, default=10, help='Number of hidden units per layer')
    parser.add_argument('--eval_freq', type=int, default=1, help='Frequency (in epochs) of model evaluation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--noise', type=float, default=0.1, help='Standard relative deviation of noise added to the input data')
    parser.add_argument('--seed', type=int, default=1, help='Seed for random number generators')
    parser.add_argument('--ratio', type=float, default=1., help='Standard deviation of noise added to the input data')


    parser.add_argument("--shared_mp", action="store_false", help="Use shared weights for message-passing layers in the GNN")
    parser.add_argument('--dataset_dir', type=str, default='data/Hyperbolic_LowRes/dataset', help='Path to the directory containing the dataset')
    parser.add_argument('--run_name', type=str, default="Tester", help='Unique identifier for the training run')
    parser.add_argument('--model', type=str, default="gnn", help='Unique identifier for the training run')
    parser.add_argument('--plots_flag', action="store_false", help="Enable plotting of the results")
    parser.add_argument('--plot_worst', action="store_true", help="Flag to plot the worst results during validation")
    parser.add_argument('--project', type=str, default="tester", help='Project name for organizing runs')
 
    args = parser.parse_args()

    # Data preparatio
    data_module = GraphDataModule(dataset_dir=args.dataset_dir, batch_size=args.batch_size, ratio=args.ratio, dataset_type=args.model)
    
    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Model instantiation
    if args.model == 'gnn':
        simulator = SimulatorGNN(args, mp_steps=args.mp_steps, input_size=2, output_size=1, hidden=args.hidden,
                                layers=args.layers, shared_mp=args.shared_mp, epochs=args.epochs, lr=args.lr, device=device, noise=args.noise)
        monitor='valid_n_step_rollout_rmse'

    elif args.model == 'poisson':
        simulator = SimulatorGNN(args, mp_steps=args.mp_steps, input_size=2, output_size=1, hidden=args.hidden,
                                layers=args.layers, shared_mp=args.shared_mp, epochs=args.epochs, lr=args.lr, device=device, noise=args.noise)
        monitor='valid_n_step_rollout_rmse'
        
    elif args.model =='meshgraph':
        simulator = SimulatorMeshGraph(args, mp_steps=args.mp_steps, input_size=7, edge_input_size=4, output_size=4, hidden=args.hidden,
                                layers=args.layers, shared_mp=args.shared_mp, epochs=args.epochs, lr=args.lr, device=device, noise=args.noise)
        monitor='valid_n_step_rollout_pos_rmse'

    # Callbacks
    chck_path, name = set_run_directory(args)
    wandb_logger = WandbLogger(name=name, project=args.project)
    
    model_save_custom = ModelSaveTopK(dirpath=str(chck_path / 'models'), monitor=monitor, mode='min', topk=3)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    trainer = pl.Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu",
                            max_epochs=args.epochs,
                            logger=wandb_logger,
                            callbacks=[lr_monitor, model_save_custom],
                            num_sanity_val_steps=0,
                            deterministic=True,
                            check_val_every_n_epoch=args.eval_freq,
                            enable_checkpointing=False,
                            
                         )
    # fit model
    trainer.fit(simulator, datamodule=data_module)
    wandb.save(str(chck_path / 'config.json'))
    trainer.test(simulator, datamodule=data_module)

    # test model
    simulator.load_checkpoint(str(chck_path / 'models' / 'topk1.pth'))
    trainer.test(simulator, datamodule=data_module)