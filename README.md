# LGSDF
MOCIM: Multi-Robot Online Collaborative Implicit Mapping Through Distributed Optimization

 ## Abstract
Constructing high-quality dense maps in real-time serves as a crucial cornerstone for enabling robots to effectively perform downstream tasks. Recently, there has been growing interest in using neural networks as maps to represent functions that implicitly define the geometric or optical features of a scene. However, existing implicit mapping algorithms are constrained to single-robot scenarios, restricting both perceptual range and exploration efficiency. In this paper, we present MOCIM, a multi-robot online collaborative implicit mapping algorithm to construct an implicit Euclidean Signed Distance Field (ESDF), formulated as a distributed optimization task.   In our formulation, Each robot independently maintains its own local data and neural network. At each iteration, robots train networks using local data and network parameters from their peers. Subsequently, they transmit the latest version of network parameters to their neighbors, thus keeping the local network parameters of all robots continuously consistent.  When optimizing the network model, the robots use not raw but in-grid fused sensor data to prevent training data conflicts. In addition, we constrain the signed distance values of concealed regions by Small Batch Euclidean Distance Transform (SBEDT) to eliminate reconstruction artifacts.
 
<img src="https://github.com/BIT-DYN/MOCIM/blob/main/figs/poster.png" width="50%">

## Install
```bash
git clone https://github.com/BIT-DYN/MOCIM
conda env create -f environment.yml
conda activate mocim
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