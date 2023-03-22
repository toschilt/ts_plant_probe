# Semantic feature detector for cornfields

This repository contains a ROS package that implements a semantic feature detector for cornfields. It uses data gathered with TerraSentia, a autonomous robot developed by the University of Illinois (Urbana-Champaign). Research supported by grant #2022/13040-0, São Paulo Research Foundation (FAPESP).

## Setup

### Requirements

- Ubuntu 20
- Python 3.8.10
- ROS Noetic
- ZED SDK ([reference](https://www.stereolabs.com/docs/installation/))
- zed-ros-wrapper package (installation steps below)
- Python libraries
  - ```cv2```
  - ```matplotlib```
  - ```numpy```
  - ```PIL```
  - ```plotly```
  - ```pycocotools```
  - ```scipy```
  - ```sklearn```
  - ```torch``` (1.13.1)
  - ```torchvision``` (0.14.1)
  - ```json```

### Installation

To install this package, open a bash terminal, clone the **zed-ros-wrapper** repository and this repository. After that, update the dependencies and build the packages.

```
cd ~/catkin_ws/src
git clone --recursive https://github.com/stereolabs/zed-ros-wrapper.git
git clone git@github.com:toschilt/ts_semantic_feature_detector.git
cd ..
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source ./devel/setup.sh
```

We also need to change some ZED camera parameters. To do that, copy the files into this repository */params* folder to the *zed-ros-wrapper/zed_wrapper/params* folder.

## Running

This package is still in development thus some configurations are still done manually in the source files. The needed modifications are shown below.

### Using the segmentation model
This detector uses a Mask-RCNN model implemented with the PyTorch framework. The dataset used to train the network can be downloaded [here](https://drive.google.com/file/d/14yGDzg8SLhFUf-vlLLKnwUDPvkTYqAI-/view?usp=share_link). The final weights can be downloaded [here](https://drive.google.com/drive/folders/1XSaAKSRYraLnB9FV0a6x_diLWJ_trV11?usp=share_link).

The *scripts/_demo.py* script has basic model training and inference syntax to help you to get started. Please see code documentation for more details.

### Extracting data for detection
This project uses data storaged in a rosbag and in SVO files. The rosbag contains the robot's EKF poses and IMU information. The SVO contains the RGB and depth images.

The algorithm expects a standard folder structure to work properly. The folder */data* contains the basic structure. Copy the rosbag file to the */data/rosbag* folder and the SVO file to the */data/svo* folder. After that, create the specific type data folders for better organization:

```
cd ~/catkin_ws/src/ts_semantic_feature_detector/data/rosbag
mkdir ekf
mkdir imu
cd ~/catkin_ws/src/ts_semantic_feature_detector/data/svo
mkdir rgb
mkdir depth
```

You need to specify the absolute path and the .bag and .svo filenames for the extractor. To do that, edit the *scripts/extract_data.py* file inside this package. Your DataExtractor class constructor must have something similar to:

```
DataExtractor(
    'absolute_path_to_data_folder',
    'rosbag_filename.bag',
    'svo_filename.svo'
)
```

You can customize the data path folder if the subfolders are organized in the same structure. After that, run the **roscore** and in another terminal run the node using:

```
rosrun ts_semantic_feature_detector extract_data.py
```

### Entire pipeline

To run this project entire pipeline, you need to do some path changes in the *scripts/ts_perception.py* file. Check the TerraSentiaPerception class constructor and change the paths for your system accordingly.

After that, run the **roscore** and in another terminal run the node using

```
rosrun ts_semantic_feature_detector ts_perception.py
```

## To-do
- Improve this README with more details
- Automatically create folder when extracting data (problem with permissions)
- Review code documentation
- Generate documentation page