import os
import pytorch_lightning as pl
import wandb


import numpy as np
import shutil


class ModelSave(pl.Callback):
    def __init__(self, dirpath=None, save_freq=None):
        super().__init__()
        self.dirpath = dirpath
        self.save_freq = save_freq

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.save_freq is not None:
            pl_module.save_checkpoint(savedir=self.dirpath+f'/epoch{trainer.current_epoch}')
        else:
            print('Not saving model checkpoint...')
    
    def on_train_end(self, trainer, pl_module):
        pl_module.save_checkpoint(savedir=self.dirpath+f'/final_model')




class ModelSaveTopK(pl.Callback):
    def __init__(self, dirpath=None, filename='best', monitor='valid_loss', mode='min', topk=3, wandb_on=True):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.top_k = [{'epoch': -1, 'value': np.inf if mode == 'min' else -np.inf} for _ in range(topk)]
        self.compare = (lambda a, b: a < b) if mode == 'min' else (lambda a, b: a > b)
        self.wandb_on = wandb_on

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Get the monitored metric value
        metrics = trainer.callback_metrics
        new_value = metrics.get(self.monitor)

        if new_value is None:
            return  # Skip if the metric is not available

        worst_epoch = self.top_k[-1]

        if self.compare(new_value, worst_epoch['value']):
            # Replace the worst entry with the new value

            self.top_k.append({'epoch': trainer.current_epoch, 'value': new_value})
            self.top_k = sorted(self.top_k, key=lambda x: x['value'], reverse=(self.mode != 'min'))

            # Save the new model checkpoint
            pl_module.save_checkpoint(savedir=self.dirpath+f'/epoch{trainer.current_epoch}', with_wandb=True)

            # remove not top k models
            to_be_removed = self.top_k[-1]
            del self.top_k[-1]

            for file in os.listdir(self.dirpath):
                if f"epoch{to_be_removed['epoch']}"in file:
                    file_path = os.path.join(self.dirpath, file)
                    # Remove the file
                    os.remove(file_path)

            # rename top k models
            for filename in os.listdir(self.dirpath):
                # Check if the filename contains the old pattern
                for k, element_k in enumerate(self.top_k):
                    if f"epoch{element_k['epoch']}" in filename:
                        # Get full file paths
                        old_filepath = os.path.join(self.dirpath, filename)
                        new_filepath = os.path.join(self.dirpath, f'topk{k+1}_epoch{element_k["epoch"]}.pth')
                        # Rename the file
                        os.rename(old_filepath, new_filepath)
                        if self.wandb_on:
                            # Copy the file to the destination folder
                            top_k_name = os.path.join(self.dirpath, f'topk{k + 1}.pth')
                            shutil.copy(new_filepath, top_k_name)
                            wandb.save(top_k_name)







