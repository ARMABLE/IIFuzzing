# InSkipFuzz

This repository contains data and code for the paper **Accelerating Fuzzing Testing with In-process Skip for Sequential Decision-making Model**.



## Overview

We propose an in-process skipping fuzzing framework InSkipFuzz to effectively and efficiently test sequential decision-making models in blackbox settings.

We make experimental evaluation of effectiveness and efficiency of InSkipFuzz  with **four models** involving **three simulation environments** with promising performance, outperforming the SOTA baseline.



## Environment Setup

#### RL CARLA

The RL model we evaluate is borrowed from this awesome repository: https://github.com/valeoai/LearningByCheating.  

Part of the `PythonAPI` code is borrowed from the official [CARLA](https://github.com/carla-simulator/carla) repo, which is under MIT license.

##### Setting up environment:

Setup CARLA:

```shell
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
mkdir carla_RL_IAs
tar -xvzf CARLA_0.9.6.tar.gz -C carla_RL_IAs
cd carla_RL_IAs
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin
mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg
```

Download models:

```shell
wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_only_town01_train_weather.zip
unzip model_RL_IAs_only_town01_train_weather.zip
wget https://github.com/marintoro/LearningByCheating/releases/download/v1.0/model_RL_IAs_CARLA_Challenge.zip
unzip model_RL_IAs_CARLA_Challenge.zip
```

Main requirements:

```shell
carla==0.9.6
numpy==1.18.5
pygame==2.0.1
Cython==0.29.33
hdbscan==0.8.26
python-weka-wrapper3==0.1.6
torch==1.5.1
scikit-learn==0.22.2
scipy==1.4.1
```



#### IL CARLA

The RL model we evaluate is borrowed from this awesome repository: https://github.com/dotchen/LearningByCheating, which is under MIT license.  

Part of the `PythonAPI` code is borrowed from the official [CARLA](https://github.com/carla-simulator/carla) repo, which is under MIT license.

##### Setting up environment:

Setup CARLA:

```shell
wget http://carla-assets-internal.s3.amazonaws.com/Releases/Linux/CARLA_0.9.6.tar.gz
mkdir carla_RL_IAs
tar -xvzf CARLA_0.9.6.tar.gz -C carla_RL_IAs
cd carla_RL_IAs
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town01.bin
wget http://www.cs.utexas.edu/~dchen/lbc_release/navmesh/Town02.bin
mv Town*.bin CarlaUE4/Content/Carla/Maps/Nav/
cd PythonAPI/carla/dist
rm carla-0.9.6-py3.5-linux-x86_64.egg
wget http://www.cs.utexas.edu/~dchen/lbc_release/egg/carla-0.9.6-py3.5-linux-x86_64.egg
```

Download models:

```shell
mkdir -p ckpts/image
cd ckpts/image
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/image/model-10.th
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/image/config.json
cd ../..
mkdir -p ckpts/priveleged
cd ckpts/priveleged
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/privileged/model-128.th
wget http://www.cs.utexas.edu/~dchen/lbc_release/ckpts/privileged/config.json
```

Main requirements:

```shell
carla==0.9.6
numpy==1.18.5
pygame==2.0.1
Cython==0.29.33
hdbscan==0.8.26
python-weka-wrapper3==0.1.6
torch==1.5.1
scikit-learn==0.22.2
scipy==1.4.1
```



#### MARL Coop Navi

The MARL model we evaluate is borrowed from this awesome repository: https://github.com/openai/maddpg, which is under MIT license.  

The Coop Navi environment is installed according to this repository: https://github.com/openai/multiagent-particle-envs, which is under MIT license.

##### Setting up environment:

Main requirements:

```shell
h5py==3.1.0
keras-applications==1.0.8
keras-preprocessing==1.1.2
tensorboard==1.15.0
tensorflow-estimator==1.15.1
tensorflow-gpu==1.15.0
hdbscan==0.8.26
python-weka-wrapper3==0.2.12
torch==1.5.1
scikit-learn==0.24.2
scipy==1.4.1
```

In addition, you need to execute the following command to install maddpg and multiagent:

```shell
cd ./maddpg
pip install -e .
cd ../multiagent-particle-envs
pip install -e .
```



#### RL BipedalWalker

The RL algorithm is in the folder `./RL_BipedalWalker/rl-baselines3-zoo`.  

The RL model we evaluate is borrowed from these awesome repositories: https://github.com/DLR-RM/rl-baselines3-zoo, https://github.com/DLR-RM/rl-trained-agents, which are under MIT license.

##### Setting up environment:

Download models:

```shell
cd ./rl-baselines3-zoo
git clone https://github.com/DLR-RM/rl-trained-agents
```

Main requirements:

```shell
numpy==1.19.3
gym-minigrid==1.0.3
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
python-weka-wrapper3==0.2.12
torch==1.10.2
scikit-learn==0.24.2
scipy==1.5.4
```

In addition, you need to execute the following command to install gym and stable_baselines3:

```shell
cp ./gym/setup.py ./
pip install -e .
cp ./stable_baselines3/setup.py ./
pip install -e .
```



## Running

#### Training  Sequence Representation Model:

```shell
# RL_CARLA for example
cd UnsupervisedScalableRepresentationLearningTimeSeries
python uea.py --dataset RL_CARLA --path ./data/TestData --save_path ./models --hyper default_hyperparameters.json [--cuda --gpu 0]
```

#### Training Cluster Model:

```shell
# RL_CARLA for example
cd UnsupervisedScalableRepresentationLearningTimeSeries/cuz_utils
python cluster.py
```

**Note**: 

(1) We have provided the above models

(2) Data used for training sequence representation model and cluster model can be downloaded on the following location: [Google Drive](https://drive.google.com/drive/folders/1To9UupwZ-57z5pyqfzOSxbyuXOPc7D2i?usp=share_link ). After the download is complete, you can place it in the following directory: ``\UnsupervisedScalableRepresentationLearningTimeSeries\data\TestData``.



#### InSkipFuzz Test:

**Note**: For RL_CARLA and IL_CARLA, first run `./carla_RL_IAs/CarlaUE4.sh -fps=10 -benchmark -carla-port=3000` to start the CARLA environment.

```python
# RL CARLA
cd /RL_CARLA
python benchmark_agent.py --suite=town2 --max-run 100 --path-folder-model model_RL_IAs_only_town01_train_weather/ --crop-sky --emguide --port=3000

# IL CARLA
cd /IL_CARLA
python benchmark_agent.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide

# MARL Coop Navi
cd /MARL_CoopNavi/maddpg/experiments
python testing.py

# RL BipedalWalker
cd /RL_BipedalWalker/rl-baselines3-zoo
python enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --em --no-render
```



## Trend Graph Results

The trend of Crash and Test number of InSkipFuzz and baseline(MDPFuzz):

![image](https://github.com/ARMABLE/InSkipFuzz/blob/main/images/trend.png?raw=true)



## Reference

- https://github.com/dotchen/LearningByCheating
- https://github.com/DLR-RM/rl-baselines3-zoo
- https://github.com/DLR-RM/rl-trained-agents
- https://github.com/openai/maddpg
- https://github.com/openai/multiagent-particle-envs
- https://github.com/Qi-Pang/MDPFuzz
- https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries
