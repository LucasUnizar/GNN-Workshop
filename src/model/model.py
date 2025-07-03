import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.utils.plot import boxplot_error
from src.utils.metrics import rrmse_inf, rmse, mse, mse_roll
from src.model.encoders import EncoderProcessorDecoder

from src.utils.plot import plot_3D_scatter, plot_combined, plot_mse_rollout, plot_mse_rollout_mean_std, plot_frame_2d
from src.utils.utils import transform_data
import wandb
import os
import json
from pathlib import Path
import argparse


class SimulatorGNN(pl.LightningModule):
    def __init__(self, args=None, mp_steps=15, input_size=2, output_size=1, hidden=128,
                 layers=2, model_dir='checkpoint/simulator.pth', pretrain=None,
                 transforms=None, shared_mp=False, epochs=1, lr=1e-4, noise=0.0, device='cpu',
                 optim_flag='cosine_annealing', mssg_flag=False):
        super().__init__()

        self.mssg_flag = mssg_flag

        if pretrain is not None:
            json_path = Path(pretrain).parent.parent / 'config.json'
            # Open and load the JSON file
            with open(json_path, "r") as file:
                config = json.load(file)
            args = argparse.Namespace(**config)

            # reset variables
            mp_steps, layers, hidden, shared_mp = args.mp_steps, args.layers, args.hidden, args.shared_mp
            

        
        self.model = EncoderProcessorDecoder(mp_steps, hidden_size=hidden, layers=layers,
                                             node_input_size=input_size, output_size=output_size,
                                             shared_mp=shared_mp, mssg_flag=self.mssg_flag)

        self._input_normalizer = Normalizer(size=input_size, name='_input_normalizer', device=device)
        self._output_normalizer = Normalizer(size=output_size, name='_output_normalizer', device=device)

        self.args = args
        self.input_size = input_size
        self.transforms = transforms
        self.noise = args.noise
        self.shared_mp = shared_mp
        self.epochs = epochs
        self.lr = lr
        self.model_dir = model_dir
        self.optim_flag = optim_flag
        self.loaded_model = False if pretrain is None else True
        
        # noise
        self.noise = noise

        # Save results
        self.z_net_one_step_list, self.z_gt_one_step_list = [], []
        self.z_net_n_step_list, self.z_gt_n_step_list = [], []
        self.mssg_pass = []

        if pretrain is not None:
            self.load_checkpoint(pretrain)
            
        print('Simulator model initialized')

    def forward(self, graph):

        # extract data from graph
        cur_u = graph.u
        if self.training and self.noise > 0:
            cur_u = self.add_noise(cur_u)
        if self.input_size > 1:
            cur_u = torch.cat((cur_u, graph.n), dim=1)
        else:
            pass
        graph.x = self._input_normalizer(cur_u, self.training)
        # prepare target data
        
        if self.args.model == 'gnn':
            target_du = graph.du
        elif self.args.model == 'poisson':
            target_du = graph.u1
        else:
            raise ValueError(f"Unknown model type: {self.args.model}")

        target_du_norm = self._output_normalizer(target_du, self.training)           
        if self.mssg_flag:
            predicted_du_norm, mssg = self.model(graph)
            return predicted_du_norm, target_du_norm, mssg
        else:
            predicted_du_norm = self.model(graph)
            return predicted_du_norm, target_du_norm

    def training_step(self, batch, batch_idx):

        graph = batch

        predicted_du_norm, target_du_norm = self.forward(graph)

        error = torch.sum((predicted_du_norm - target_du_norm) ** 2, dim=1)
        loss = torch.mean(error)

        self.log('loss_train', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=error.shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):

        graph = batch

        predicted_du_norm, target_du_norm = self.forward(graph)

        error = torch.sum((predicted_du_norm - target_du_norm) ** 2, dim=1)
        loss = torch.mean(error)

        self.log('loss_valid', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=error.shape[0])

    def on_validation_epoch_start(self):

        z_net_one_step_list, z_gt_one_step_list = [], []
        z_net_n_step_list, z_gt_n_step_list = [], []
        one_step_rollout_mse_list = []

        n_step_rollout_mse_list = []
        n_step_rollout_mse_evolution_list = []

        for sim in self.trainer.datamodule.valid_rollout_dataloader:
            z_net, z_gt, mssg_list = self.integrate_sim(sim, full_rollout=False)
            z_net_one_step_list.append(z_net)
            z_gt_one_step_list.append(z_gt)
            one_step_rollout_mse_list.append(mse(z_gt, z_net))

        for sim in self.trainer.datamodule.valid_rollout_dataloader:
            z_net, z_gt, mssg_list = self.integrate_sim(sim, full_rollout=True)
            z_net_n_step_list.append(z_net)
            z_gt_n_step_list.append(z_gt)
            n_step_rollout_mse_list.append(mse(z_gt, z_net))
            n_step_rollout_mse_evolution_list.append(mse_roll(z_gt, z_net))

        rmse_error_one_step_rollout = torch.sqrt(torch.mean(torch.tensor(one_step_rollout_mse_list)))
        rmse_error_n_step_rollout = torch.sqrt(torch.mean(torch.tensor(n_step_rollout_mse_list)))
                                       
        if self.args.plots_flag:
            idx = 0
            if self.args.plot_worst:
                idx = torch.argmax(torch.tensor(one_step_rollout_mse_list)).item()
                print(f'Worst RMSE: {torch.sqrt(one_step_rollout_mse_list[idx]):.4f}')

            print('Starting validation plots...')
            # One-step rollout plots
            if self.args.model == 'gnn':
                plot_combined(z_net_one_step_list[idx], z_gt_one_step_list[idx], self.X, self.Y , name='valid_one_step_rollout_combined')
                plot_3D_scatter(z_net_one_step_list[idx], z_gt_one_step_list[idx], self.X, self.Y, name='valid_one_step_rollout_scatter')
                # N-step rollout plots
                plot_mse_rollout(n_step_rollout_mse_evolution_list, name='valid_n_step_rollout_mse_evolution')
                plot_combined(z_net_n_step_list[idx], z_gt_n_step_list[idx], self.X, self.Y, name='valid_n_step_rollout_combined')
                plot_3D_scatter(z_net_n_step_list[idx], z_gt_n_step_list[idx], self.X, self.Y, name='valid_n_step_rollout_scatter')
            else:
                plot_frame_2d(z_net_n_step_list[idx], z_gt_n_step_list[idx], self.X, self.Y, frame_idx=1)

            print('Plots done')

        self.log('valid_one_step_rollout_rmse', rmse_error_one_step_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_one_step_list))
        self.log('valid_n_step_rollout_rmse', rmse_error_n_step_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_n_step_list))

    def test_step(self, batch, batch_idx):
        # Define the test step (forward pass and loss calculation)
        z_net_one_step_rollout, z_gt_one_step_rollout, mssg_list = self.integrate_sim(batch, full_rollout=False)
        self. z_net_one_step_list.append(z_net_one_step_rollout)
        self.z_gt_one_step_list.append(z_gt_one_step_rollout)
        self.mssg_pass.append(mssg_list)

        # Noised data
        z_net_n_step_rollout, z_gt_n_step_rollout, _ = self.integrate_sim(batch, full_rollout=True)
        self.z_net_n_step_list.append(z_net_n_step_rollout)
        self.z_gt_n_step_list.append(z_gt_n_step_rollout)

    def test_rollout(self, dataset, full_rollout=True, save_path='./'):

        name_steps = 'n_steps' if full_rollout else 'one_step'

        z_net_list, z_gt_list = [], []
        rmse_list, rrmse_list = [], []
        mse_list, mse_list_roll = [], []
        if self.mssg_flag:
            self.mssg_latent = []
            self.mssg_output = []
        
        simulations = dataset.__len__()
        for i in range(simulations):
            sim = dataset[i]
            z_net, z_gt, mssg_list = self.integrate_sim(sim, full_rollout=full_rollout)
            if self.mssg_flag:
                mssg_snaps = [mss[0] for mss in mssg_list]
                out_snaps = [mss[1] for mss in mssg_list]
                self.mssg_latent.append(mssg_snaps)
                mssg_out_snaps = []
                for mssg_snap in out_snaps:
                    mssg_snap_hops_denorm = [self._output_normalizer.inverse(torch.Tensor(mssg_snap_hop)) for mssg_snap_hop in mssg_snap]
                    mssg_out_snaps.append(mssg_snap_hops_denorm)
                self.mssg_output.append(mssg_out_snaps)

            z_net_list.append(z_net)
            z_gt_list.append(z_gt)
            mse_list.append(mse(z_gt, z_net))
            mse_list_roll.append(mse_roll(z_gt, z_net))
            rrmse_list.append(rrmse_inf(z_gt, z_net))
            rmse_list.append(rmse(z_gt, z_net))

        rmse_error = torch.sqrt(torch.mean(torch.tensor(mse_list)))

        # Message passing plots
        if self.mssg_flag:
            transform_data(self.mssg_latent, z_net_list, z_gt_list, self.mssg_output)

        if self.args.plots_flag:
            print(f'Starting validation plots for {name_steps} rollout...')
            
            # Calculate indices for best and worst cases
            mse_tensor = torch.tensor(mse_list)
            worst_idx = torch.argmax(mse_tensor).item()
            best_idx = torch.argmin(mse_tensor).item()

            # Print RMSE for best and worst cases
            print(f'Worst RMSE: {torch.sqrt(mse_tensor[worst_idx]):.4f}')
            print(f'Best RMSE: {torch.sqrt(mse_tensor[best_idx]):.4f}')
            
            # Generate box plots and rollout plots
            boxplot_error(rmse_list, name=f'test_rmse_{name_steps}', error='RMSE', save_path=save_path)
            boxplot_error(rrmse_list, name=f'test_rrmse_{name_steps}', error='RRMSE', save_path=save_path)
            if self.args.model == 'gnn':
                plot_mse_rollout(mse_list_roll, name=f'test_{name_steps}_rollout_mse', with_wandb=False, save_path=save_path)
                plot_mse_rollout_mean_std(mse_list_roll, name=f'test_{name_steps}_rollout_mse_mean_std', save_path=save_path, save=True)
                
                # Plot combined and 3D scatter for worst case
                plot_combined(z_net_list[worst_idx], z_gt_list[worst_idx], self.X, self.Y, name=f'test_{name_steps}_rollout_combined_worst', with_wandb=False, save_dir=save_path)
                plot_3D_scatter(z_net_list[worst_idx], z_gt_list[worst_idx], self.X, self.Y, name=f'test_{name_steps}_rollout_scatter_worst', with_wandb=False, save_dir=save_path)
                
                # Plot combined and 3D scatter for best case
                plot_combined(z_net_list[best_idx], z_gt_list[best_idx], self.X, self.Y, name=f'test_{name_steps}_rollout_combined_best', with_wandb=False, save_dir=save_path)
                plot_3D_scatter(z_net_list[best_idx], z_gt_list[best_idx], self.X, self.Y, name=f'test_{name_steps}_rollout_scatter_best', with_wandb=False, save_dir=save_path)
            else:
                plot_frame_2d(z_net_list[best_idx], z_gt_list[best_idx], self.X, self.Y, frame_idx=1, wandb_flag=False, save_dir=save_path)

            print('Plots done!!')

        return rmse_error

    def on_test_epoch_end(self):
        # Define the test epoch end (calculate the mean loss)
        rmse_one_step_roll_list = []
        rrmse_one_step_roll_list = []
        rmse_n_step_roll_list = []
        rrmse_n_step_roll_list = []

        print('Calculating test loss...')
        for i in range(len(self.z_net_n_step_list)):
            # One-step rollout
            z_net_one_step = self.z_net_one_step_list[i]
            z_gt_one_step = self.z_gt_one_step_list[i]

            rmse_one_step_roll_list.append(rmse(z_gt_one_step, z_net_one_step,))
            rrmse_one_step_roll_list.append(rrmse_inf(z_gt_one_step, z_net_one_step, ))

            # N-step rollout
            z_net_n_step = self.z_net_n_step_list[i]
            z_gt_n_step = self.z_gt_n_step_list[i]

            rmse_n_step_roll_list.append(rmse(z_gt_n_step, z_net_n_step))
            rrmse_n_step_roll_list.append(rrmse_inf(z_gt_n_step, z_net_n_step))

        # One-step rollout
        boxplot_error(rmse_one_step_roll_list, name=f'test_rmse_one_step_rollout_ModelCheckpoint={self.loaded_model}', error='RMSE')
        boxplot_error(rrmse_one_step_roll_list, name=f'test_rrmse_one_step_rollout_ModelCheckpoint={self.loaded_model}', error='RRMSE')

        mean_rmse_one_step = torch.mean(torch.tensor(rmse_one_step_roll_list))
        mean_rrms_one_step = torch.mean(torch.tensor(rrmse_one_step_roll_list))
        self.log(f'test_rmse_one_step_rollout_ModelCheckpoint={self.loaded_model}', mean_rmse_one_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_one_step_list))
        self.log(f'test_rrmse_one_step_rollout_ModelCheckpoint={self.loaded_model}', mean_rrms_one_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_one_step_list))

        # N-step rollout
        boxplot_error(rmse_n_step_roll_list, name=f'test_rmse_n_step_rollout_ModelCheckpooint={self.loaded_model}', error='RMSE')
        boxplot_error(rrmse_n_step_roll_list, name=f'test_rrmse_n_step_rollout_ModelCheckpoint={self.loaded_model}', error='RRMSE')

        mean_rmse_n_step = torch.mean(torch.tensor(rmse_n_step_roll_list))
        mean_rrms_n_step = torch.mean(torch.tensor(rrmse_n_step_roll_list))
        self.log(f'test_rmse_n_step_rollout_ModelCheckpoint={self.loaded_model}', mean_rmse_n_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_n_step_list))
        self.log(f'test_rrmse_n_step_rollout_ModelCheckpoint={self.loaded_model}', mean_rrms_n_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_n_step_list))

        print(f'Test RMSE n-step rollout: {mean_rmse_n_step:.4f}, Test RRMSE n-step rollout: {mean_rrms_n_step:.4f}')
        print('Test loss calculated')

        return mean_rmse_n_step

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Choose the learning rate scheduler based on self.optim_flag
        if self.optim_flag == 'reduce_on_plateau':
            # Set up the ReduceLROnPlateau scheduler
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',        # Use 'min' for minimizing loss; switch to 'max' for maximizing metrics
                    factor=0.5,        # Factor by which the learning rate will be reduced
                    patience=30,       # Number of epochs to wait before reducing LR
                    threshold=1e-4,    # Minimum change in monitored quantity to qualify as improvement
                    min_lr=1e-6        # Minimum learning rate
                ),
                'monitor': 'loss_train',  # Metric to monitor
                'interval': 'epoch',
                'frequency': 1,
            }
        elif self.optim_flag == 'cosine_annealing':
            # Set up the CosineAnnealing scheduler
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.args.epochs, # Maximum number of iterations
                    eta_min=1e-5  # Minimum learning rate
                ),
                'interval': 'epoch',
                'frequency': 1,
            }
        else:
            raise ValueError("Invalid scheduler flag: choose 'reduce_on_plateau' or 'cosine_annealing'")

        return [optimizer], [scheduler]
    
    def integrate_sim(self, data, full_rollout=True, noised_data=False):
        """
        Integrates the simulation for a given dataset and data index.

        Parameters:
        dataset (list): The dataset containing the simulation data.
        data_index (int): The index of the specific data to simulate.
        WorldInfo_flag (bool): Whether to collect WorldInfo data. Default is False.
        full_rollout (bool): Whether to use full rollout mode. Default is True.

        Returns:
        tuple: z_net (tensor), z_gt (tensor), and optionally WorldInfo (list)
        """
        # Extract dt and n flag for BCs
        msg_list = [] if self.mssg_flag else None

        # Extract dx and dys
        self.X = data[0].X.cpu().detach().numpy()
        self.Y = data[0].Y.cpu().detach().numpy()

        # Extract data list and initial conditions
        N_nodes = data[0].X.size(0)

        # Preallocate tensors for network output and ground truth
        u_net = torch.zeros(len(data) + 1, N_nodes, 1)
        u_gt = torch.zeros(len(data) + 1, N_nodes, 1)
        u_net[0, :, :] = data[0].u
        u_gt[0, :, :] = data[0].u
        u_input_next_step = None

        # Rollout loop through each time step
        for t, graph in enumerate(data):
            graph = graph.to(self.device)
            graph.u = u_input_next_step if u_input_next_step is not None else graph.u

            if noised_data:
                graph.u = self.add_relative_noise(graph.u)

            if self.mssg_flag:
                du_prediction_norm, _, mssg = self.forward(graph)
            else:
                du_prediction_norm, _ = self.forward(graph)
            du_prediction = self._output_normalizer.inverse(du_prediction_norm)

            if self.args.model == 'gnn':
                u1_hat = graph.u + du_prediction
            elif self.args.model == 'poisson':
                u1_hat = du_prediction

            # Save the network output and ground truth
            u_net[t + 1] = u1_hat.detach()
            u_gt[t + 1] = graph.u1.detach()
            if self.mssg_flag:
                msg_list.append(mssg)

            # Update the state
            u_input_next_step = u_net[t + 1].to(self.device) if full_rollout else graph.u1.to(self.device)

        if self.mssg_flag:
            return u_net, u_gt, msg_list
        else:
            return u_net, u_gt, msg_list

    def load_checkpoint(self, ckpdir=None):
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir, map_location=torch.device(self.device))
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.' + k)
                if isinstance(value, int) or isinstance(value, str):
                    setattr(object, para, value)
                else:
                    setattr(object, para, value.to(self.device))

        print("Simulator model loaded checkpoint %s" % ckpdir)
        self.loaded_model = True

    def save_checkpoint(self, savedir=None, with_wandb=True):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(savedir), exist_ok=True)

        model = self.state_dict()
        output_normalizer = self._output_normalizer.get_variable()
        input_normalizer = self._input_normalizer.get_variable()

        to_save = {'model': model, '_input_normalizer': input_normalizer, '_output_normalizer': output_normalizer}

        torch.save(to_save, savedir + '.pth')
        print('Simulator model saved at %s' % savedir)

        if with_wandb == True:
            wandb_dir = wandb.run.dir + f'/{savedir.split("/")[-1]}.pth'
            torch.save(to_save, wandb_dir)
            wandb.save(wandb_dir)

    def add_noise(self, variable):
        """
        Adds noise to a variable with higher relative noise in regions of higher values.
        
        Args:
            variable (torch.Tensor): The input tensor to which noise is added.
            noise_scale (float): Scaling factor for the noise. Default is 0.1.
        
        Returns:
            torch.Tensor: Tensor with added noise.
        """
        # Generate random noise with the same shape as the input variable
        noise = torch.randn_like(variable)
        # Scale noise based on the absolute values of the input variable
        #scaled_noise = noise * (torch.abs(variable) * self.noise)
        # Add the scaled noise to the original variable
        #noised_variable = variable + scaled_noise
        noised_variable = variable + noise * self.noise
        return noised_variable


class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10 ** 6, std_epsilon=1e-6, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate:
            # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean().to(batched_data.device)) / self._std_with_epsilon().to(batched_data.device)

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon().to(normalized_batch_data.device) + self._mean().to(normalized_batch_data.device)

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batched_data ** 2, axis=0, keepdims=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(self._acc_count,
                                   torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count,
                                   torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):

        dict = {'_max_accumulations': self._max_accumulations,
                '_std_epsilon': self._std_epsilon,
                '_acc_count': self._acc_count,
                '_num_accumulations': self._num_accumulations,
                '_acc_sum': self._acc_sum,
                '_acc_sum_squared': self._acc_sum_squared,
                'name': self.name
                }

        return dict

    
