#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torchvision
from ESANet.src.models.model_one_modality import ESANetOneModality
from ESANet.src.datasets.sunrgbd.sunrgbd import SUNRBDBase
from ESANet.src.models.model import Upsample
import click
import numpy as np
import os
import torch.nn as nn
from ESANet.hapt_model import HaptModel

file_dir = os.path.dirname(os.path.realpath(__file__))
cmap = SUNRBDBase.CLASS_COLORS


class SemanticSegmentor:
    def __init__(self, path_to_weights: str, image_topic: str):
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
    	
        self.model = HaptModel(n_classes=1)
        self.model.decoder_semseg.output_conv = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)

        self.model.load_state_dict(torch.load(path_to_weights)["state_dict"])
        self.model.eval()
        self.bridge = CvBridge()

        self.pub = rospy.Publisher("segmenter", Image, queue_size=10)
        self.subscriber = rospy.Subscriber(image_topic, Image, self.image_callback)

        self.torchify = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((480, 640)),
            ]
        )

    def image_callback(self, img):
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


@click.command()
@click.option(
    "--weights_filename",
    default="weights.pth",
    help="Path to esanet pretrained weights",
)
@click.option(
    "--image_topic",
    default="/image_raw",
    help="Image topic for publishing semantic segmentation",
)
def main(weights_filename, image_topic):
    weights = os.path.join(file_dir, weights_filename)
    semantic_segmentor = SemanticSegmentor(
        path_to_weights=weights, image_topic=image_topic
    )
    rospy.init_node("semantic_segmentation_node")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Cacati nel petto")


if __name__ == "__main__":
    main()
