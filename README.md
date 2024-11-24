# FedGA
This is the source code of the paper "FedGA: Federated Learning with Gradient Alignment for Error Asymmetry Mitigation".
Please refer to the [pdf file](./GA_Convergence_Analysis.pdf) for the **convergence analysis** and proof.

## System Overview
All the experiments were conducted on a Linux machine with the following specifications:
- OS: Red Hat Enterprise Linux 8.5 (Ootpa)
- CPU: Intel(R) Xeon(R) Platinum 8360Y CPU @ 2.40GHz


## Requirements
The code was developed using the following versions of the software:
- Python: 3.11.9

The following packages are required to run the code:
- datasets 2.21.0
- python-box 7.0
- scipy 1.12.0
- torchvision 0.15.2
- torchmetrics 0.10.1
- wandb 0.17.7

## Usage
Always `cd` to the root directory of this project before running the experiments.

### Prepare the environment
Sync the dependencies with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

Activate the virtual environment before running the experiments:
```bash
source .venv/bin/activate
```

### Run single experiment
To run a short example, use the following command:
```bash
python fedlearn.py
```

### Run specific experiment
To run a specific experiment on dataset **SVHN** with Dirichlet parameter **0.1** based on **GA** algorithm, you can supply the configuration as command line argument:
```bash
python fedlearn.py DATA.NAME SVHN DATA.IB.ALPHA 0.1 FL.ALG FedGA
```
or, make a copy of the default config file [config_default](GA/utils/config_default.yaml) and modify it and run the following command:
```bash
python fedlearn.py --cfg configs/your_config.yaml
```

### Run multiple experiments
To reproduce the results in the paper, run the following command:
```bash
python fedlearnParallel.py --cfg configs/config_MNIST.yaml
python fedlearnParallel.py --cfg configs/config_SVHN.yaml
python fedlearnParallel.py --cfg configs/config_CIFAR10.yaml
```
