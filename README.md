# Material for MeshGraph Simulation Workshop

This project provides a training and evaluation pipeline for a Graph Neural Network (GNN) tailored for mesh-based simulations using PyTorch Lightning. Configuration is handled via command-line arguments for flexibility and reproducibility.

<div align="center">
<img src="/material/gnn.png" width="250">
</div>

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.9+ and the following Python packages installed:

```bash
# Clone the repository
git clone https://github.com/LucasUnizar/GNN-Workshop
cd GNN-Workshop

# Create a dedicated conda environment (highly recommended for dependency management)
conda create -n gnn python=3.11
conda activate gnn

# Install the required Python packages
pip install -r requirements.txt
```

---

## 📦 Running the Training Script

You can start the Poisson problem training by running:

```bash
python train.py --dataset_dir data/Jaca-SummerSchool25_Elliptic_HighRes/dataset --model poisson 
```
You can start also the Hyperbolic problem training by running:

```bash
python train.py --dataset_dir data/Jaca-SummerSchool25_waves/dataset --model gnn
```

---

## ⚙️ Argument Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | `int` | `64` | Number of samples per training batch |
| `--epochs` | `int` | `500` | Total number of training epochs |
| `--mp_steps` | `int` | `6` | Number of message-passing steps in the GNN |
| `--layers` | `int` | `2` | Number of GNN layers |
| `--hidden` | `int` | `16` | Number of hidden units per GNN layer |
| `--eval_freq` | `int` | `25` | Frequency (in epochs) for model evaluation |
| `--lr` | `float` | `1e-3` | Learning rate for the optimizer |
| `--seed` | `int` | `1` | Random seed for reproducibility |
| `--dataset_dir` | `str` | `data/Jaca-SummerSchool25_waves/dataset` | Path to dataset directory |
| `--run_name` | `str` | `"test"` | Unique name for the training run |
| `--model` | `str` | `"gnn"` | Model type (`gnn` or `poisson`) |
| `--project` | `str` | `"Jaca-SummerSchool25-GNNs"` | Project name for wandb logging |

---

## 🧪 Model Evaluation

The script will:
1. Train the model
2. Save the top 3 checkpoints
3. Load the best model (`topk1.pth`) after training
4. Evaluate it on the test set

---

## 📊 Experiment Tracking with Weights & Biases

Make sure you are logged into wandb before running the script:

```bash
wandb login
```

Logs, metrics, and model summaries will be uploaded to your wandb dashboard under the specified `--project` name.

---

## 📁 Project Structure

```
src/
├── dataloader/
│   └── datamodule.py
├── model/
│   ├── model.py
│   └── callbacks.py
└── utils/
    └── utils.py
```
---

## 🙋‍♂️ Need Help?

If you have any issues or questions during the workshop, feel free to open an issue or ask one of the instructors.

Happy simulating! 🎉
