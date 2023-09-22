# MACIM
MACIM: Multi-Agent Collaborative Implicit Mapping Through Distributed Optimizationn

 ## Abstract
Collaborative mapping aids agents in achieving an efficient and comprehensive understanding of their environment. Recently, there has been growing interest in using neural networks as maps to represent functions that implicitly define the geometric features of a scene.
However, existing implicit mapping algorithms are constrained to single-agent scenarios, thus restricting mapping efficiency. In this paper, we present MACIM, a multi-agent collaborative implicit mapping algorithm to construct an implicit Euclidean Signed Distance Field (ESDF), formulated as a distributed optimization task.  
In our formulation, each agent independently maintains its own local data and neural network. At each iteration, agents train networks using local data and network parameters from their peers. Subsequently, they transmit the latest version of network parameters to their neighbors, thus keeping the local network parameters of all agents continuously consistent. 
When optimizing the network model, the agents use not raw but in-grid fused sensor data to prevent training data conflicts. In addition, we constrain the signed distance values of unobserved regions by Small Batch Euclidean Distance Transform (SBEDT) to mitigate reconstruction artifacts. 
 
<img src="https://github.com/BIT-DYN/MOCIM/blob/main/figs/poster.png">

## Install
```bash
git clone https://github.com/BIT-DYN/MACIM
conda env create -f environment.yml
conda activate macim
```

## Download Data

```bash
bash data/download_data.sh
```

## Run

### ReplicaCad
```bash
cd mocim/train/
python train.py --config configs/replicaCAD.json
```
### Scannet
```bash
cd mocim/train/
python train.py --config configs/scannet.json
```

## Result
<img src="https://github.com/BIT-DYN/MOCIM/blob/main/figs/result.png"  width="100%">
