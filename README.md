
# p2v-slam


**Full SLAM codes are updated. I'll update this readme file soon.**


This repository contains the open-source code for the my *IEEE Transactions on Automation Science and Engineering (T-ASE)* paper: **"Implicit Point-to-Voxel LiDAR-IMU SLAM"**.  
*(The paper is accepted, and I'll upload it soon after published)*

<div align="center">
  <img src="media/algorithm.png" alt="Algorithm Workflow" width="80%">
  <p align="center"><b>The proposed point-to-voxel SLAM algorithm overview.</b></p>
</div>


Unlike traditional "point-to-plane" observation models, we propose an **implicit point-to-voxel** observation model to achieve reliable and robust localization in unstructured environments.

<table table-layout="fixed" width="100%">
  <tr>
    <td align="center" width="50%">
      <img src="media/demo-1.gif" alt="Demo 1" style="max-width:100%;">
    </td>
    <td align="center" width="50%">
      <img src="media/demo-2.gif" alt="Demo 2" style="max-width:100%;">
    </td>
  </tr>
  <tr>
    <td align="center" colspan="2">
      point-to-voxel matches ($\color{green}{\textbf{green}}$) and point-to-plane matches ($\color{red}{\textbf{red}}$). Our p2v-matches are more consistent in tree leaves regions.
    </td>
  </tr>
</table>

## Installation

### Dependencies
This code is developed based on [PV-LIO](https://github.com/HViktorTsoi/PV-LIO).


1. livox_msg_ros (maybe not the full livox driver is okay. Download the messages from [this repo](https://github.com/LarryDong/livox_msg_ros2))
2. 

* **Core Dependency:** [ONNX Runtime](https://onnxruntime.ai/) (for neural network inference).
* Please refer to the original PV-LIO repository for other basic requirements (e.g., ROS, Eigen, PCL).

## Usage
1. Run the launch file. For example, for Botanic Garden Dataset, run: 
```bash
# In one terminal:
roslaunch p2v-slam run_BotanicGarden.launch
# In a new terminal
rosbag play xxx.bag
```
You can download a demo Botanic Garden sequence (`1018_13`) from here (GoogleDrive)[TODO:] or full dataset from: [BotanicGarden Repo](https://github.com/robot-pesg/BotanicGarden).

2. 




## Models
`p2v-slam` requires two neural networks:
1. **VE-Net**: Extracts implicit features from voxels.
2. **IR-Net**: Predicts observation residuals and uncertainty.

A pre-trained model is available at: `/pretrained-model/xxx.onnx`

## Train your own model
**TODO:**
1. Release of the full model architecture and training pipeline.
2. Release of our official training datasets.
3. Tutorials on how to train the model with your own data to enhance performance in specific scenarios.

## Acknowledgements
We build upon the excellent work of [PV-LIO](https://github.com/HViktorTsoi/PV-LIO). We thank the authors for their contribution to the community.

## Roadmap / TODO List
- [ ] Upload the complete SLAM source code.
- [ ] Upload the full training code.
- [ ] Provide data processing scripts for generating training data from your own datasets.






# TODO For My Own
[] Remove voxel map original codes? `voxel_map_util.cpp` and `hpp`?
[] 