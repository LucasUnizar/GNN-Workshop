from src.model.transforms import FaceToEdgeTethra, RadiusGraphMesh, ContactDistance, MeshDistance
from src.model.model import Normalizer
import torch_geometric.transforms as T
import torch.nn.functional as F

import torch
import pytorch_lightning as pl
from src.utils.plot import boxplot_error
from src.utils.metrics import rrmse_inf, rmse, mse, mse_roll
from src.utils.plot import plot_3D_scatter_elliptic
from src.utils.utils import transform_data


from src.model.meshgraph_contact import MeshGraphNetContact

import wandb
import os
import json
from pathlib import Path
import argparse
import numpy as np


class SimulatorMeshGraph(pl.LightningModule):
    def __init__(self, args=None, mp_steps=15, input_size=1, edge_input_size=1, output_size=1, hidden=128, radius=0.05,
                 layers=2, model_dir='checkpoint/simulator.pth', pretrain=None,
                 shared_mp=False, epochs=1, lr=1e-4, device='cpu', optim_flag='cosine_annealing', mssg_flag=False):
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
            input_size, output_size = args.input_size, args.output_size

        
        self.model = MeshGraphNetContact(mp_steps, hidden_size=hidden, layers=layers, shared_mp=shared_mp,
                                        node_input_size=7, edge_input_size=4, output_size=4,)


        self._input_normalizer = Normalizer(size=7, name='_input_normalizer', device=device)
        self._input_edge_mesh_normalizer = Normalizer(size=edge_input_size * 2, name='_input_edge_mesh_normalizer', device=device)
        self._input_edge_contact_normalizer = Normalizer(size=edge_input_size, name='_input_edge_contact_normalizer', device=device)
        self._output_normalizer = Normalizer(size=output_size, name='_output_normalizer', device=device)

        self.args = args
        self.transforms = T.Compose([FaceToEdgeTethra(remove_faces=False), RadiusGraphMesh(r=radius), T.Cartesian(norm=False),
                                  T.Distance(norm=False), MeshDistance(norm=False), ContactDistance(norm=False)])
        self.noise = args.noise
        self.shared_mp = shared_mp
        self.epochs = epochs
        self.lr = lr
        self.model_dir = model_dir
        self.optim_flag = optim_flag
        self.loaded_model = False if pretrain is None else True

        # Save results
        self.z_net_one_step_list, self.z_gt_one_step_list = [], []
        self.z_net_n_step_list, self.z_gt_n_step_list = [], []
        self.mssg_pass = []

        if pretrain is not None:
            self.load_checkpoint(pretrain)

        print('Simulator model initialized')
    
    def __feature_engineering(self, graph):
        # Compute the Edge attributes for mesh and world
        graph = self.transforms(graph.cpu()).to(self.device)

        # extract data from graph
        node_type = graph.n
        cur_position = graph.pos
        target_position = graph.y[:, :3]
        target_stress = graph.y[:, -1:]

        # prepare target data
        target_disp = target_position - cur_position
        target = torch.cat((target_disp, target_stress), dim=-1)
        target_norm = self._output_normalizer(target, self.training)


        # compute displacement with target for actuator
        graph.x = self.__process_node_attr(node_type, target_position, cur_position)
        # Process the edges attribute data, normalize
        graph.edge_attr, graph.edge_contact_attr = self.__process_edge_attr(graph.edge_attr, graph.edge_contact_attr)

        return graph, target_norm
                              

    def forward(self, graph):
        # Feature engineering and target construction
        graph, target_norm = self.__feature_engineering(graph)
        # Forward pass through GNN
        predicted_norm = self.model(graph)

        return predicted_norm, target_norm
    
    def training_step(self, batch, batch_idx):

        graph = batch
        mask = torch.where(graph.n != 1, True, False).squeeze()

        predicted_norm, target_norm = self.forward(graph)

        error = torch.sum((predicted_norm - target_norm) ** 2, dim=1)
        loss = torch.mean(error[mask])

        self.log('loss_train', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=error[mask].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):

        graph = batch
        mask = torch.where(graph.n != 1, True, False).squeeze()

        predicted_norm, target_norm = self.forward(graph)

        error = torch.sum((predicted_norm - target_norm) ** 2, dim=1)
        loss = torch.mean(error[mask])

        self.log('loss_valid', loss.detach().item(), prog_bar=True, on_epoch=True, on_step=False,
                 batch_size=error[mask].shape[0])
        
    def test_step(self, batch, batch_idx):
        # One-step predictions (forward pass and loss calculation)
        z_net_one_step_rollout, z_gt_one_step_rollout, mssg_list = self.integrate_sim(batch, full_rollout=False)
        self. z_net_one_step_list.append(z_net_one_step_rollout)
        self.z_gt_one_step_list.append(z_gt_one_step_rollout)
        self.mssg_pass.append(mssg_list)

        # N-step predictions
        z_net_n_step_rollout, z_gt_n_step_rollout, _ = self.integrate_sim(batch, full_rollout=True)
        self.z_net_n_step_list.append(z_net_n_step_rollout)
        self.z_gt_n_step_list.append(z_gt_n_step_rollout)

    def configure_optimizers(self):
        # Set up the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Set up the CosineAnnealing scheduler
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs, eta_min=1e-5),
            'interval': 'epoch',
            'frequency': 1,
        }

        return [optimizer], [scheduler]
    
    @torch.no_grad()
    def integrate_sim(self, loader, full_rollout=True):
        
        mask_bc, predicted_position = None, None
        predictions, targets = [], []
        msg_list = []
        edges_index, edges_contact = [], []

        for graph in loader:
          
            if (predicted_position is not None) and (full_rollout):
                graph.x[:, :3] = predicted_position
                graph.pos = predicted_position
            
            current_position = graph.x[:, :3].to(self.device)
            target_position_stress = torch.clone(graph.y.to(self.device))

            if mask_bc is None:
                node_type = graph.n
                mask_bc = torch.where(node_type == 3, True, False).squeeze().to(self.device)
                mask_actuator = torch.where(node_type == 1, True, False).squeeze().to(self.device)
                predictions.append(graph.x.detach().cpu().numpy())
                targets.append(graph.x.detach().cpu().numpy())

            prediction_norm, _ = self.forward(graph)  # pred_norm, target_norm
            prediction = self._output_normalizer.inverse(prediction_norm)
            
            predicted_position_stress = torch.cat([prediction[:, :3] + current_position, prediction[:, -1:]], dim=-1)
            # Only infer positions fixed points, stress is calculated
            predicted_position_stress[mask_bc, :3] = target_position_stress[mask_bc, :3]
            # Only both, positions and stress for actuator
            predicted_position_stress[mask_actuator, :] = target_position_stress[mask_actuator, :]
            # Set the predicted positions for actuator, bc an normal nodes
            predicted_position = predicted_position_stress[:, :3]

            predictions.append(predicted_position_stress.detach().cpu().numpy())
            targets.append(target_position_stress.detach().cpu().numpy())
            
            if self.mssg_flag:
                mssg, output_mssg = [], []
                # feature engineering
                graph, _ = self.__feature_engineering(graph)
                # Disclousure forward
                graph = self.model.encoder(graph)
                # decode encoded space
                mssg.append(graph.x.cpu().detach().numpy())  # Save the latent representation for visualization.
                output_mssg.append(self.model.decoder(graph).cpu().detach().numpy())
                # decode after each mp iteration
                for model in self.model.processor_list:
                    graph = model(graph)
                    mssg.append(graph.x.cpu().detach().numpy())  # Save the latent representation for visualization.
                    output_mssg.append(self.model.decoder(graph).cpu().detach().numpy())
                msg_list.append((mssg, output_mssg))
                edges_index.append(graph.edge_index.cpu().detach().numpy())
                edges_contact.append(graph.edge_contact_index.cpu().detach().numpy())

        result = [np.stack(predictions), np.stack(targets)]
        # n = graph.n.squeeze().cpu().detach().numpy()
        # face = graph.face_mesh.cpu().numpy() if 'face_mesh' in graph.keys() else graph.face.cpu().numpy()

        return result[0], result[1], (msg_list, (edges_index, edges_contact))

    def on_validation_epoch_start(self):

        z_net_one_step_list, z_gt_one_step_list = [], []
        z_net_n_step_list, z_gt_n_step_list = [], []
        one_step_rollout_mse_list = []
        one_step_rollout_pos_mse_list, one_step_rollout_stress_mse_list = [], []

        n_step_rollout_mse_list = []
        n_step_rollout_mse_evolution_list = []
        n_step_rollout_pos_mse_list, n_step_rollout_stress_mse_list = [], []

        for sim in self.trainer.datamodule.valid_rollout_dataloader:
            z_net, z_gt, _ = self.integrate_sim(sim, full_rollout=False)
            z_net_one_step_list.append(z_net)
            z_gt_one_step_list.append(z_gt)
            
            one_step_rollout_mse_list.append(mse(z_gt, z_net))
            one_step_rollout_pos_mse_list.append(mse(z_gt[:, :, :3], z_net[:, :, :3]))
            one_step_rollout_stress_mse_list.append(mse(z_gt[:, :, -1:], z_net[:, :, -1:]))

        for sim in self.trainer.datamodule.valid_rollout_dataloader:
            z_net, z_gt, _ = self.integrate_sim(sim, full_rollout=True)
            z_net_n_step_list.append(z_net)
            z_gt_n_step_list.append(z_gt)

            n_step_rollout_mse_list.append(mse(z_gt, z_net))
            n_step_rollout_pos_mse_list.append(mse(z_gt[:, :, :3], z_net[:, :, :3]))
            n_step_rollout_stress_mse_list.append(mse(z_gt[:, :, -1:], z_net[:, :, -1:]))
            n_step_rollout_mse_evolution_list.append(mse_roll(z_gt, z_net))

        rmse_error_one_step_rollout = torch.sqrt(torch.mean(torch.tensor(one_step_rollout_mse_list)))
        rmse_error_one_step_pos_rollout = torch.sqrt(torch.mean(torch.tensor(one_step_rollout_pos_mse_list)))
        rmse_error_one_step_stress_rollout = torch.sqrt(torch.mean(torch.tensor(one_step_rollout_stress_mse_list)))
        
        rmse_error_n_step_rollout = torch.sqrt(torch.mean(torch.tensor(n_step_rollout_mse_list)))
        rmse_error_n_step_pos_rollout = torch.sqrt(torch.mean(torch.tensor(n_step_rollout_pos_mse_list)))
        rmse_error_n_step_stress_rollout = torch.sqrt(torch.mean(torch.tensor(n_step_rollout_stress_mse_list)))
                                       
        if self.args.plots_flag:
            idx = 0
            if self.args.plot_worst:
                idx = torch.argmax(torch.tensor(one_step_rollout_mse_list)).item()
                print(f'Worst RMSE: {torch.sqrt(one_step_rollout_mse_list[idx]):.4f}')

            print('Starting validation plots...')
            # One-step rollout plots
            # plot_combined(z_net_one_step_list[idx], z_gt_one_step_list[idx], self.X, self.Y , name='valid_one_step_rollout_combined')
            # plot_3D_scatter(z_net_one_step_list[idx], z_gt_one_step_list[idx], self.X, self.Y, name='valid_one_step_rollout_scatter')
            plot_3D_scatter_elliptic(z_net_one_step_list[idx], z_gt_one_step_list[idx], name='valid_one_step_rollout_scatter')
            # N-step rollout plots
            # plot_mse_rollout(n_step_rollout_mse_evolution_list, name='valid_n_step_rollout_mse_evolution')
            # plot_combined(z_net_n_step_list[idx], z_gt_n_step_list[idx], self.X, self.Y, name='valid_n_step_rollout_combined')
            # plot_3D_scatter(z_net_n_step_list[idx], z_gt_n_step_list[idx], self.X, self.Y, name='valid_n_step_rollout_scatter')
            plot_3D_scatter_elliptic(z_net_n_step_list[idx], z_gt_n_step_list[idx], name='valid_n_step_rollout_scatter')

            print('Plots done')

        self.log('valid_one_step_rollout_rmse', rmse_error_one_step_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_one_step_list))
        self.log('valid_one_step_rollout_pos_rmse', rmse_error_one_step_pos_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_n_step_list))
        self.log('valid_one_step_rollout_stress_rmse', rmse_error_one_step_stress_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_n_step_list))
        
        
        self.log('valid_n_step_rollout_rmse', rmse_error_n_step_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_n_step_list))
        self.log('valid_n_step_rollout_pos_rmse', rmse_error_n_step_pos_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_n_step_list))
        self.log('valid_n_step_rollout_stress_rmse', rmse_error_n_step_stress_rollout, on_epoch=True, on_step=False, batch_size=len(z_net_n_step_list))

    def on_test_epoch_end(self):
        # Define the test epoch end (calculate the mean loss)
        rmse_one_step_pos_list = []
        rrmse_one_step_pos_list = []
        rmse_one_step_stress_list = []
        rrmse_one_step_stress_list = []

        rmse_n_step_pos_list = []
        rrmse_n_step_pos_list = []
        rmse_n_step_stress_list = []
        rrmse_n_step_stress_list = []

        print('Calculating test loss...')
        for i in range(len(self.z_net_n_step_list)):
            # One-step rollout
            z_net_one_step = self.z_net_one_step_list[i]
            z_gt_one_step = self.z_gt_one_step_list[i]

            rmse_one_step_pos_list.append(rmse(z_gt_one_step[:, :, :3], z_net_one_step[:, :, :3],))
            rrmse_one_step_pos_list.append(rrmse_inf(z_gt_one_step[:, :, :3], z_net_one_step[:, :, :3], ))
            rmse_one_step_stress_list.append(rmse(z_gt_one_step[:, :, -1:], z_net_one_step[:, :, -1:],))
            rrmse_one_step_stress_list.append(rrmse_inf(z_gt_one_step[:, :, -1:], z_net_one_step[:, :, -1:], ))

            # N-step rollout
            z_net_n_step = self.z_net_n_step_list[i]
            z_gt_n_step = self.z_gt_n_step_list[i]

            rmse_n_step_pos_list.append(rmse(z_gt_n_step[:, :, :3], z_net_n_step[:, :, :3]))
            rrmse_n_step_pos_list.append(rrmse_inf(z_gt_n_step[:, :, :3], z_net_n_step[:, :, :3]))
            rmse_n_step_stress_list.append(rmse(z_gt_n_step[:, :, -1:], z_net_n_step[:, :, -1:]))
            rrmse_n_step_stress_list.append(rrmse_inf(z_gt_n_step[:, :, -1:], z_net_n_step[:, :, -1:]))

        # One-step rollout
        boxplot_error(rmse_one_step_pos_list, name=f'test_rmse_one_step_pos_rollout_ModelCheckpoint={self.loaded_model}', error='RMSE')
        boxplot_error(rrmse_one_step_pos_list, name=f'test_rrmse_one_step_pos_rollout_ModelCheckpoint={self.loaded_model}', error='RRMSE')
        boxplot_error(rmse_one_step_stress_list, name=f'test_rmse_one_step_stress_rollout_ModelCheckpoint={self.loaded_model}', error='RMSE')
        boxplot_error(rrmse_one_step_stress_list, name=f'test_rrmse_one_step_stress_rollout_ModelCheckpoint={self.loaded_model}', error='RRMSE')

        mean_rmse_one_pos_step = torch.mean(torch.tensor(rmse_one_step_pos_list))
        mean_rrms_one_pos_step = torch.mean(torch.tensor(rrmse_one_step_pos_list))
        mean_rmse_one_stress_step = torch.mean(torch.tensor(rmse_one_step_stress_list))
        mean_rrms_one_stress_step = torch.mean(torch.tensor(rrmse_one_step_stress_list))
        self.log(f'test_rmse_one_step_pos_rollout_ModelCheckpoint={self.loaded_model}', mean_rmse_one_pos_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_one_step_list))
        self.log(f'test_rrmse_one_step_pos_rollout_ModelCheckpoint={self.loaded_model}', mean_rrms_one_pos_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_one_step_list))
        self.log(f'test_rmse_one_step_rollout_stress_ModelCheckpoint={self.loaded_model}', mean_rmse_one_stress_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_one_step_list))
        self.log(f'test_rrmse_one_step_rollout_stress_ModelCheckpoint={self.loaded_model}', mean_rrms_one_stress_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_one_step_list))

        # N-step rollout
        boxplot_error(rmse_n_step_pos_list, name=f'test_rmse_n_step_rollout_ModelCheckpooint={self.loaded_model}', error='RMSE')
        boxplot_error(rrmse_n_step_pos_list, name=f'test_rrmse_n_step_rollout_ModelCheckpoint={self.loaded_model}', error='RRMSE')
        boxplot_error(rmse_n_step_stress_list, name=f'test_rmse_n_step_rollout_ModelCheckpooint={self.loaded_model}', error='RMSE')
        boxplot_error(rrmse_n_step_stress_list, name=f'test_rrmse_n_step_rollout_ModelCheckpoint={self.loaded_model}', error='RRMSE')

        mean_rmse_n_pos_step = torch.mean(torch.tensor(rmse_n_step_pos_list))
        mean_rrms_n_pos_step = torch.mean(torch.tensor(rrmse_n_step_pos_list))
        mean_rmse_n_stress_step = torch.mean(torch.tensor(rmse_n_step_stress_list))
        mean_rrms_n_stress_step = torch.mean(torch.tensor(rrmse_n_step_stress_list))
        self.log(f'test_rmse_n_step_pos_rollout_ModelCheckpoint={self.loaded_model}', mean_rmse_n_pos_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_n_step_list))
        self.log(f'test_rrmse_n_step_pos_rollout_ModelCheckpoint={self.loaded_model}', mean_rrms_n_pos_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_n_step_list))
        self.log(f'test_rmse_n_step_stress_rollout_ModelCheckpoint={self.loaded_model}', mean_rmse_n_stress_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_n_step_list))
        self.log(f'test_rrmse_n_step_stress_rollout_ModelCheckpoint={self.loaded_model}', mean_rrms_n_stress_step, on_epoch=True, on_step=False, batch_size=len(self.z_net_n_step_list))

        print(f'Test RMSE n-step pos rollout: {mean_rmse_n_pos_step:.4f}, Test RRMSE n-step pos rollout: {mean_rrms_n_pos_step:.4f}')
        print(f'Test RMSE n-step stress rollout: {mean_rmse_n_stress_step:.4f}, Test RRMSE n-step stress rollout: {mean_rrms_n_stress_step:.4f}')
        print('Test loss calculated')

    def test_rollout(self, dataset, full_rollout=True, save_path='./'):

        name_steps = 'n_steps' if full_rollout else 'one_step'

        z_net_list, z_gt_list = [], []
        edge_list, edge_contact_list = [], []
        face_list, node_list = [], []
        rmse_list, rrmse_list = [], []
        mse_list, mse_list_roll = [], []
        if self.mssg_flag:
            self.mssg_latent = []
            self.mssg_output = []

        for sim in dataset:
            z_net, z_gt, (mssg_list, edges) = self.integrate_sim(sim, full_rollout=full_rollout)
            edge_index, edge_contact = edges
            if self.mssg_flag:
                mssg_snaps = [mss[0] for mss in mssg_list]
                out_snaps = [mss[1] for mss in mssg_list]
                self.mssg_latent.append(mssg_snaps)
                mssg_out_snaps = []
                for mssg_snap in out_snaps:
                    mssg_snap_hops_denorm = [self._output_normalizer.inverse(torch.Tensor(mssg_snap_hop)) for
                                             mssg_snap_hop in mssg_snap]
                    mssg_out_snaps.append(mssg_snap_hops_denorm)
                self.mssg_output.append(mssg_out_snaps)

            z_net_list.append(z_net)
            z_gt_list.append(z_gt)
            edge_list.append(edge_index)
            edge_contact_list.append(edge_contact)
            face_list.append(np.array([s.face.cpu().numpy() for s in sim]))
            node_list.append(np.array([s.n.cpu().numpy() for s in sim]))
            mse_list.append(mse(z_gt, z_net))
            mse_list_roll.append(mse_roll(z_gt, z_net))
            rrmse_list.append(rrmse_inf(z_gt, z_net))
            rmse_list.append(rmse(z_gt, z_net))

            plot_3D_scatter_elliptic(z_net, z_gt, name='test_n_step_rollout_scatter', save_dir=save_path, with_wandb=False)

        rmse_error = torch.sqrt(torch.mean(torch.tensor(mse_list)))

        # Message passing plots
        if self.mssg_flag:
            transform_data(self.mssg_latent, z_net_list, z_gt_list, self.mssg_output,
                           export_path=save_path + '/results/message_hops',
                           edge_index=edge_list, edge_contact=edge_contact_list,
                           faces=face_list, node_type=node_list)

            print('Plots done!!')

        return rmse_error

    def __process_node_attr(self, types, target_pos, curr_pos):

        node_feature = []
        # build one-hot for node type
        node_type = torch.squeeze(types.long())
        one_hot = F.one_hot(node_type, 4)
        # compute velocity based on future position xt+1
        disp = target_pos - curr_pos
        # set to zero non-actuator displacements
        mask_no_actuator = (node_type != 1)  # This creates a boolean mask
        disp[mask_no_actuator] = 0.
        # append and concatenate node attributes
        node_feature.append(disp)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1).float()
        node_feats = self._input_normalizer(node_feats, self.training)

        return node_feats

    def __process_edge_attr(self, edge_mesh_attr, edge_contact_attr):

        edge_mesh_attr_norm = self._input_edge_mesh_normalizer(edge_mesh_attr, self.training)
        edge_contact_attr_norm = self._input_edge_contact_normalizer(edge_contact_attr, self.training)

        return edge_mesh_attr_norm, edge_contact_attr_norm
    
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
        _input_normalizer = self._input_normalizer.get_variable()
        _input_edge_mesh_normalizer = self._input_edge_mesh_normalizer.get_variable()
        _input_edge_contact_normalizer = self._input_edge_contact_normalizer.get_variable()
        _output_normalizer = self._output_normalizer.get_variable()

        to_save = {'model': model, '_input_normalizer': _input_normalizer, '_input_edge_mesh_normalizer': _input_edge_mesh_normalizer, 
                   '_input_edge_contact_normalizer': _input_edge_contact_normalizer, '_output_normalizer': _output_normalizer, 
                   }

        torch.save(to_save, savedir + '.pth')
        print('Simulator model saved at %s' % savedir)
        
        if with_wandb == True:
            wandb_dir = wandb.run.dir + f'/{savedir.split("/")[-1]}.pth'
            torch.save(to_save, wandb_dir)
            wandb.save(wandb_dir)