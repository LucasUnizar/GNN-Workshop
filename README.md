# 🧠 MeshGNN-PDE: Solving PDEs with Message Passing Neural Networks and MeshGraphNets

This repository integrates **Message Passing Neural Networks (MPNNs)** and **MeshGraphNets** to solve a variety of Partial Differential Equations (PDEs), ranging from toy elliptic problems to realistic dynamics like plate collisions.

---

## 🚀 Project Overview

This repo explores and benchmarks **graph-based deep learning architectures** for solving PDE systems:

| Model        | Type of PDE Solved     | Description |
|--------------|------------------------|-------------|
| `gnn`        | Hyperbolic / Parabolic | Uses a GNN architecture (MPNN) to simulate time-dependent systems |
| `poisson`    | Elliptic               | Solves stationary toy PDE problems like the Poisson equation |
| `meshgraph`  | Dynamic (Collision)    | Uses MeshGraphNets to simulate elliptic plate collision systems |

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/LucasUnizar/MPNN-PDE
cd MPNN-PDE

# Create environment
conda create -n MPNN python=3.11
conda activate MPNN

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Project Structure

```bash
.
├── src/
│   ├── model/
│   │   ├── model.py              # MPNN-based model
│   │   ├── model_meshgraph.py    # MeshGraphNet model
│   ├── dataloader/datamodule.py  # Data handling
│   ├── callbacks.py              # Custom callbacks
│   └── utils/utils.py            # Utility functions
├── data/                         # Datasets (default path)
├── main.py                       # Entry point
└── README.md
```

---

## 🧪 Running Experiments

### 🔹 MPNN for Hyperbolic/Parabolic PDEs

```bash
python main.py \
  --model gnn \
  --dataset_dir data/Hyperbolic_LowRes/dataset \
  --run_name hyperbolic_run
```

### 🔹 Poisson Solver (Elliptic PDE)

```bash
python main.py \
  --model poisson \
  --dataset_dir data/Poisson/dataset \
  --run_name poisson_test
```

### 🔹 MeshGraphNet for Plate Collision

```bash
python main.py \
  --model meshgraph \
  --dataset_dir data/PlateCollision \
  --run_name plate_sim
```

---

## ⚙️ Argument Reference (`argparse`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch_size` | `int` | `64` | Number of samples per training batch |
| `--epochs` | `int` | `1` | Number of training epochs |
| `--mp_steps` | `int` | `1` | Message-passing steps in GNN |
| `--layers` | `int` | `2` | Number of model layers |
| `--hidden` | `int` | `10` | Hidden units per layer |
| `--eval_freq` | `int` | `1` | Evaluation frequency (in epochs) |
| `--lr` | `float` | `1e-3` | Learning rate |
| `--noise` | `float` | `0.1` | Relative noise std added to inputs |
| `--seed` | `int` | `1` | Random seed |
| `--ratio` | `float` | `1.0` | Data subsampling or additional control param |
| `--shared_mp` | `flag` | Enabled | Use shared weights for message-passing layers |
| `--dataset_dir` | `str` | `'data/Hyperbolic_LowRes/dataset'` | Path to dataset |
| `--run_name` | `str` | `"Tester"` | Unique run identifier |
| `--model` | `str` | `"gnn"` | Choose from: `gnn`, `poisson`, `meshgraph` |
| `--plots_flag` | `flag` | Enabled | Enable plotting of results |
| `--plot_worst` | `flag` | Disabled | Plot worst validation results |
| `--project` | `str` | `"tester"` | W&B project name |

---

## 🧠 Models


### MeshGraphNet
Designed for **realistic dynamic simulations**, like elastic plate collisions. Uses spatial message passing on mesh graphs.

---

## 📊 Logging & Visualization

We use **Weights & Biases (W&B)** for experiment tracking.

Log in or sign up:

```bash
wandb login
```

All metrics, checkpoints, and plots are logged under the given `--project` and `--run_name`.

---

## 📈 Example Results



