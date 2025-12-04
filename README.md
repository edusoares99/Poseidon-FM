# Phoenix: Modular Deep Learning Framework for Physical Simulation

Phoenix is a modular, PyTorch-based framework designed for training, fine-tuning, and visualizing models that learn from **partial differential equation (PDE)-driven data** and **3D volumetric simulations**.  
It provides an extensible research codebase for developing **physics-informed architectures**, **long-horizon prediction systems**, and **volumetric neural representations**, supporting backbones such as **Mamba-SSM** and **Transformers**.

📄 **Preprint**  
If you use Phoenix in academic or industrial research, please cite our associated preprint:  
**"Toward a Foundation Model for Partial Differential Equations Across Physics Domains"**, arXiv 2025.  
https://arxiv.org/pdf/2511.21861


## Features
- Modular, research-friendly structure (`phoenix` package)
- End-to-end workflows for training, fine-tuning, evaluation, and visualization
- Native support for volumetric / VTK data and PDE-based simulation datasets
- Physics-informed and long-horizon training utilities
- 3D visualization powered by `pyvista`
- Reproducible environment via `requirements.txt`
- Flexible backbone integration, including Mamba-SSM and Transformer models


## Repository Structure
```
├── phoenix/
│   ├── data/
│   │   └── vtk_dataset.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   └── spectral.py
│   ├── model/
│   │   ├── backbone.py
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   ├── phoenix.py
│   │   └── tokens.py
│   ├── tools/
│   │   └── visualize_vtk_preds.py
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── finetune.py
│   ├── finetune_long_horizon.py
│   ├── finetune_vtk.py
│   ├── losses.py
│   ├── main.py
│   ├── train.py
│   └── utils.py
├── finetune_drivaerml.sh
├── README.md
├── requirements.txt
├── run_finetune.sh
├── run_finetune_long_horizon.sh
└── run_phoenix.sh
```


## Setup

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Usage

Run any of the supported entry-point scripts:

```bash
python phoenix/finetune_vtk.py
```

```bash
python phoenix/finetune_long_horizon.py
```

```bash
python phoenix/main.py
```

```bash
python phoenix/finetune.py
```

```bash
python phoenix/tools/visualize_vtk_preds.py
```


## Citation

If you use **Phoenix** in your work, please cite the preprint:

```bibtex
@article{soares2025towards,
  title={Towards a Foundation Model for Partial Differential Equations Across Physics Domains},
  author={Soares, Eduardo and Brazil, Emilio Vital and Shirasuna, Victor and de Carvalho, Breno WSR and Malossi, Cristiano},
  journal={arXiv preprint arXiv:2511.21861},
  year={2025}
}
```
