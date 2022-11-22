import os
import click
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import InterpolationMode as im
import matplotlib.pyplot as plt
from src.models.model_one_modality import ESANetOneModality
from src.models.model import Upsample
from hapt_model import HaptModel


@click.command()
@click.option('--num_custom_classes', default=4, help='Number of classes used for finetuning the model')
@click.option('--image', default=None, help='Path of a specific image')

def main(num_custom_classes, image):
    # create model
    model = ESANetOneModality(encoder='resnet34', num_classes=37, channels_decoder=[512, 256, 128], pretrained_on_imagenet=False,
                            context_module='ppm', nr_decoder_blocks=[3, 3, 3], encoder_block='NonBottleneck1D', 
                            upsampling='learned-3x3-zeropad', weighting_in_encoder='SE-add').cuda()
    # model = HaptModel(n_classes=1).cuda()

    # change modules for finetuning
    model.decoder.decoder_module_1.side_output = nn.Conv2d(512, num_custom_classes, 3, 1).cuda()
    model.decoder.decoder_module_2.side_output = nn.Conv2d(256, num_custom_classes, 3, 1).cuda()
    model.decoder.decoder_module_3.side_output = nn.Conv2d(128, num_custom_classes, 3, 1).cuda()
    model.decoder.conv_out = nn.Conv2d(128, num_custom_classes, 1, 1).cuda()
    model.decoder.upsample1 = Upsample(mode='learned-3x3-zeropad', channels=num_custom_classes).cuda()
    model.decoder.upsample2 = Upsample(mode='learned-3x3-zeropad', channels=num_custom_classes).cuda()
    # model.decoder_semseg.output_conv = nn.ConvTranspose2d(
    #         in_channels=16, out_channels=num_custom_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True).cuda()

    model.load_state_dict(torch.load('/home/matteo/Code/harmony/deployment_esanet_ros/ESANet/model_best_miou_new.pth')['state_dict'])
    print('Weights loaded.')
    print('\n-------------------------------------------------------------\n')
    model.eval()

    resize_image = torchvision.transforms.Resize((480, 640), interpolation=im.BILINEAR)
    
    if image is not None:
        image = torchvision.io.read_image(image).unsqueeze(0).cuda()
        image = resize_image(image).float() / 255.
        out = model(image)
        sem_seg = torch.argmax(torch.softmax(out, axis=1), axis=1).squeeze()
        plt.imshow(image[0].permute(1, 2, 0).detach().cpu())
        plt.imshow(sem_seg.detach().cpu(), alpha=0.5)
        plt.show()
    else:
        path = '/home/matteo/Downloads/box_videos/other'
        images = os.listdir(path)
        for image in images:
            image = torchvision.io.read_image(os.path.join(path, image)).unsqueeze(0).cuda()
            image = resize_image(image).float() / 255.
            out = model(image)
            sem_seg = torch.argmax(torch.softmax(out, axis=1), axis=1).squeeze()
            sem_seg[sem_seg > 2] = 0
            plt.imshow(image[0].permute(1, 2, 0).detach().cpu())
            plt.imshow(sem_seg.detach().cpu(), alpha=0.5)
            plt.show()


if __name__ == '__main__':
    main()