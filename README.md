# Semantic and Instance segmentation

This repository contains the deliverable code for Task 3.2 (D3.3) for ROS1 Noetic and ROS2 Foxy, tested on Ubuntu 20.04.

The code is based on the work developed for the paper *Robust Double-Encoder Network for RGB-D Panoptic Segmentation*, submitted to the IEEE International Conference on Robotics and Automation (ICRA) 2023 ([link](https://arxiv.org/abs/2210.02834)):

```
@article{sodano2022robust,
  title={Robust Double-Encoder Network for RGB-D Panoptic Segmentation},
  author={Sodano, Matteo and Magistri, Federico and Guadagnino, Tiziano and Behley, Jens and Stachniss, Cyrill},
  journal={arXiv preprint arXiv:2210.02834},
  year={2022}
}
```

For further information about the code, how to re-train the network from scratch, and all libraries needed to do that, please refer to [this repository](https://github.com/PRBonn/PS-res-excite).

## How to use
Prerequisites:
- PyTorch 1.10+
- Torchvision
- Numpy
- OpenCV

### ROS1
Package the repo to allow inner imports, then run `python main.py --weights_filename <path_to_weights> --image_topic <topic_to_listen_to>`.

### ROS2
Execute `colcon build`, `. install/setup.bash`, then move your launch file into the `share` directory, and finally `ros2 launch <package> <launchfile>.
