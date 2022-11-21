#!/usr/bin/env python
# license removed for brevity
import rclpy
from rclpy.node import Node


from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision

import click
import numpy as np
import os
import torch.nn as nn
import sys
sys.path.append(os.path.abspath('/home/nickybones/Code/harmony_segmentation/ros2_ws/src/harmony_segmentation/harmony_segmentation/'))


#from ESANet.src.models.model_one_modality import ESANetOneModality
from ESANet.src.datasets.sunrgbd.sunrgbd import SUNRBDBase
from ESANet.hapt_model import HaptModel
# from ESANet.src.models.model import Upsample
# from ESANet.src.models.model_one_modality import ESANetOneModality



file_dir = os.path.dirname(os.path.realpath(__file__))
cmap = SUNRBDBase.CLASS_COLORS


class SemanticSegmentor(Node):
    def __init__(self):
        super().__init__('SemanticSegmentor')
        self.ok = True
        # self.model = ESANetOneModality(
        #     encoder="resnet34",
        #     num_classes=16,
        #     channels_decoder=[512, 256, 128],
        #     pretrained_on_imagenet=False,
        #     context_module="ppm",
        #     nr_decoder_blocks=[3, 3, 3],
        #     encoder_block="NonBottleneck1D",
        #     upsampling="learned-3x3-zeropad",
        #     weighting_in_encoder="SE-add",
        # )
        # self.model.decoder.decoder_module_1.side_output = nn.Conv2d(512, 16, 3, 1)
        # self.model.decoder.decoder_module_2.side_output = nn.Conv2d(256, 16, 3, 1)
        # self.model.decoder.decoder_module_3.side_output = nn.Conv2d(128, 16, 3, 1)
        # self.model.decoder.conv_out = nn.Conv2d(128, 16, 1, 1)
        # self.model.decoder.upsample1 = Upsample(mode='learned-3x3-zeropad', channels=16)
        # self.model.decoder.upsample2 = Upsample(mode='learned-3x3-zeropad', channels=16)

        self.declare_parameter('path_to_weights')
        path_to_weights = self.get_parameter('path_to_weights').value
        self.get_logger().info("path_to_weights: %s" % (str(path_to_weights),))

        self.declare_parameter('image_topic')
        image_topic = self.get_parameter('image_topic').value
        self.get_logger().info("image_topic: %s" % (str(image_topic),))

        self.declare_parameter('compressed')
        self.compressed =  self.get_parameter('compressed').value
        self.get_logger().info("compressed: %s" % (str(self.compressed),))
       
    	
        self.model = HaptModel(n_classes=1)
        self.model.decoder_semseg.output_conv = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

        self.model.load_state_dict(torch.load(path_to_weights)["state_dict"])
        self.model.eval()
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "segmenter", 10)

        if self.compressed:
            self.subscriber = self.create_subscription(CompressedImage, image_topic, self.image_callback, 10)
        else:      
            self.subscriber = self.create_subscription(Image, image_topic, self.image_callback, 10)
        

        self.torchify = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((480, 640)),
            ]
        )

    def image_callback(self, img):

        self.get_logger().info("Got image!")

        if self.compressed:
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(img, desired_encoding="rgb8")
            except CvBridgeError as e:
                print(e)
        else:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(img, "rgb8")
            except CvBridgeError as e:
                print(e)
        x = self.torchify(cv_image)
        sem_segmentation = torch.argmax(
            torch.softmax(self.model(x.unsqueeze(0)).squeeze(), dim=0), dim=0
        )
        sem_numpy = sem_segmentation.cpu().numpy()
        unique = np.unique(sem_numpy)
        Y = np.zeros((sem_numpy.shape[0], sem_numpy.shape[1], 3), dtype=np.uint8)
        for idx in unique:
            r, c = np.where(sem_numpy == idx)
            Y[r, c, :] = np.array(cmap[idx], dtype=np.uint8)
        labels = self.bridge.cv2_to_imgmsg(Y, "rgb8")
        self.pub.publish(labels)

def main(args=None):
  
    rclpy.init(args=args)

    semantic_segmentor = SemanticSegmentor()

    if semantic_segmentor.ok:
        rclpy.spin(semantic_segmentor)

        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        semantic_segmentor.destroy_node()
        rclpy.shutdown()
    else:
        print("There was a problem with initialization and the node could not start.")


if __name__ == "__main__":
    main()
