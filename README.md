# MARPO (Based on MAPPO)
In this repository, we extend MAPPO by introducing a Reflection Mechanism combined with an Adaptive Clipping algorithm to improve policy optimization stability and performance in multi-agent environments.

The Reflection Mechanism enhances the policy update by considering future state-action pairs, enabling more stable and reflective learning dynamics.

The Adaptive Clipping dynamically adjusts the PPO clipping range based on the estimated KL divergence, allowing more flexible yet safe policy updates compared to the fixed symmetric clipping in vanilla PPO.

These improvements lead to more robust training and better empirical results across various benchmarks.


## Environments supported:

- [StarCraftII (SMAC)](https://github.com/oxwhirl/smac)
- [StarCraftII (SMAC-HARD)](https://github.com/devindeng94/smac-hard) is an extension environment of [SMAC](https://github.com/oxwhirl/smac) for research in the field of cooperative multi-agent reinforcement learning (MARL) based on [Blizzard](http://blizzard.com)'s [StarCraft II](https://en.wikipedia.org/wiki/StarCraft_II:_Wings_of_Liberty) RTS game. SMAC-HARD makes use of Blizzard's [StarCraft II Machine Learning API](https://github.com/Blizzard/s2client-proto) and [DeepMind](https://deepmind.com)'s [PySC2](https://github.com/deepmind/pysc2) to provide a convenient interface for autonomous agents to interact with StarCraft II, getting observations and performing actions. In the self-play implementation, we align the centralized training with decentralized execution interfaces for opponents to agents. In the opponent script editing mode, users can use pysc2 script based on raw observation to write decision-tree scripts and add them to the opponent strategy pools.

# Quick Start

## Installing SMAC-HARD

You can install SMAC-HARD by using the following command:

```shell
pip install git+https://github.com/devindeng94/smac-hard.git
```

Alternatively, you can clone the SMAC-HARD repository and then install `smac_hard` with its dependencies:

```shell
git clone https://github.com/devindeng94/smac-hard.git
pip install -e smac-hard/
```



## 1. Usage
**WARNING: by default all experiments assume a shared policy by all agents i.e. there is one neural network shared by all agents**

All core code is located within the onpolicy folder. The algorithms/ subfolder contains algorithm-specific code
for MAPPO. 

* The envs/ subfolder contains environment wrapper implementations for the MPEs, SMAC, and Hanabi. 

* Code to perform training rollouts and policy updates are contained within the runner/ folder - there is a runner for 
each environment. 

* Executable scripts for training with default hyperparameters can be found in the scripts/ folder. The files are named
in the following manner: train_algo_environment.sh. Within each file, the map name (in the case of SMAC and the MPEs) can be altered. 
* Python training scripts for each environment can be found in the scripts/train/ folder. 

* The config.py file contains relevant hyperparameter and env settings. Most hyperparameters are defaulted to the ones
used in the paper; however, please refer to the appendix for a full list of hyperparameters used. 


## 2. Installation

 Here we give an example installation on CUDA == 10.1. For non-GPU & other CUDA version installation, please refer to the [PyTorch website](https://pytorch.org/get-started/locally/). We remark that this repo. does not depend on a specific CUDA version, feel free to use any CUDA version suitable on your own computer.

``` Bash
# create conda environment
conda create -n marl python==3.6.1
conda activate marl
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# install on-policy package
cd on-policy
pip install -e .
```

Even though we provide requirement.txt, it may have redundancy. We recommend that the user try to install other required packages by running the code and finding which required package hasn't installed yet.

### 2.1 StarCraftII [4.10](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip)

   

``` Bash
unzip SC2.4.10.zip
# password is iagreetotheeula
echo "export SC2PATH=~/StarCraftII/" >> ~/.bashrc
```

* download SMAC Maps, and move it to `~/StarCraftII/Maps/`.

* To use a stableid, copy `stableid.json` from https://github.com/Blizzard/s2client-proto.git to `~/StarCraftII/`.

For SMAC v2, please refer to https://github.com/oxwhirl/smacv2.git. Make sure you have the `32x32_flat.SC2Map` map file in your `SMAC_Maps` folder.



## 3.Train
Here we use train_mpe.sh as an example:
```
cd onpolicy/scripts
chmod +x ./train_mpe.sh
./train_mpe.sh
```
Local results are stored in subfold scripts/results. Note that we use Weights & Bias as the default visualization platform; to use Weights & Bias, please register and login to the platform first. More instructions for using Weights&Bias can be found in the official [documentation](https://docs.wandb.ai/). Adding `--no_wandb` or `--use_tensorboard` in command line or in the .sh file will use Tensorboard instead of Weights & Biases. 

### 3.1 Train SMAC `2c_vs_64zg` with MARPO/RMAPPO

The script `onpolicy/scripts/train_smac_mappo_simple_scripts/train_smac_2c_vs_64zg.sh` runs MARPO/RMAPPO training on the SMAC `2c_vs_64zg` map.

Because the script calls `python ../train/train_smac.py` with a relative path, run it from the script directory:

``` Bash
cd onpolicy/scripts/train_smac_mappo_simple_scripts
chmod +x train_smac_2c_vs_64zg.sh
./train_smac_2c_vs_64zg.sh
```

The script automatically adds the project root to `PYTHONPATH` based on its own location.

The current training script uses the following settings:

* Environment: `StarCraft2`
* Map: `2c_vs_64zg`
* Algorithm: `rmappo`
* Experiment name: `check`
* Seeds: `4`, `5`, `6`
* GPU: `CUDA_VISIBLE_DEVICES=0`
* Training threads: `1`
* Rollout threads: `8`
* Mini-batches: `1`
* Episode length: `400`
* Environment steps: `10050000`
* PPO epochs: `5`
* Evaluation: enabled with `--use_eval --eval_episodes 32`

The three seeds are executed serially. Although `nohup` is used, the script does not put the Python command in the background, so the next seed starts only after the previous seed finishes.

Each seed writes a log file in the script directory:

``` text
2c_vs_64zg_seed4.log
2c_vs_64zg_seed5.log
2c_vs_64zg_seed6.log
```

Training results are written by `train_smac.py` under:

``` text
onpolicy/scripts/results/StarCraft2/2c_vs_64zg/rmappo/check/
```

Common changes:

* Change the seed list in `for seed in 4 5 6; do` to run different random seeds.
* Change `CUDA_VISIBLE_DEVICES=0` to select another GPU.
* Change `num_env_steps`, `episode_length`, `n_rollout_threads`, or `eval_episodes` to adjust training and evaluation workload.
* Add `--no_wandb` or `--use_tensorboard` if you want to disable Weights & Biases and write TensorBoard logs instead.

