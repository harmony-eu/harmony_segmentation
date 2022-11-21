import os
import click
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchmetrics import IoU
from torch.utils.data import DataLoader
# from ESANet.src.models.model_one_modality import ESANetOneModality
# from ESANet.src.models.model import Upsample
from src.models.model_one_modality import ESANetOneModality
from src.models.model import Upsample
from dataset_lab import CustomDataset
from hapt_model import HaptModel

# details for finetuning
# atm:  *** classes for lab data, 7 classes for boxes
#       path_lab = /home/matteo/segments/matsod2_indoor_navigation/v1/
#       path_box = /home/matteo/segments/matsod_boxes/v1.0/
@click.command()
@click.option('--num_custom_classes', default=7, help='Number of classes needed for finetuning the model')
@click.option('--epochs', default=50, help='Epochs for finetuning')
@click.option('--path', help='Path to the dataset')
@click.option('--pretrained', default=True, help='Boolean for using a pretrained model or not')


def main(num_custom_classes, epochs, path, pretrained):
    # create model
    # model = ESANetOneModality(encoder='resnet34', num_classes=37, channels_decoder=[512, 256, 128], pretrained_on_imagenet=False,
    #                         context_module='ppm', nr_decoder_blocks=[3, 3, 3], encoder_block='NonBottleneck1D', 
    #                         upsampling='learned-3x3-zeropad', weighting_in_encoder='SE-add').cuda()
    model = HaptModel(n_classes=1).cuda()
    print(model)

    # load pre-trained weights
    # model.load_state_dict(torch.load('../weights.pth')['state_dict'])
    if pretrained:
        weights = torch.load('/home/matteo/Code/hapt_models/bosch/ours.ckpt')
        # weights = torch.load('/home/matteo/Code/deployment_esanet_ros/best_210922.pth')
        for name, param in model.named_parameters():
            param = weights['state_dict'][name]
        print('Weights loaded.')
    print('\n-------------------------------------------------------------\n')

    # for param in model.parameters():
    #     param.requires_grad = False

    # change modules for finetuning
    model.decoder_semseg.output_conv = nn.ConvTranspose2d(
            in_channels=16, out_channels=num_custom_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True).cuda()
    # model.decoder.decoder_module_1.side_output = nn.Conv2d(512, num_custom_classes, 3, 1).cuda()
    # model.decoder.decoder_module_2.side_output = nn.Conv2d(256, num_custom_classes, 3, 1).cuda()
    # model.decoder.decoder_module_3.side_output = nn.Conv2d(128, num_custom_classes, 3, 1).cuda()
    # model.decoder.conv_out = nn.Conv2d(128, num_custom_classes, 1, 1).cuda()
    # model.decoder.upsample1 = Upsample(mode='learned-3x3-zeropad', channels=num_custom_classes).cuda()
    # model.decoder.upsample2 = Upsample(mode='learned-3x3-zeropad', channels=num_custom_classes).cuda()

    # dataloader for finetuning data
    path_to_train_images = os.path.join(path, 'images')
    path_to_train_masks = os.path.join(path, 'annos')
    path_to_valid_images = os.path.join(path, 'val_images')
    path_to_valid_masks = os.path.join(path, 'val_masks')
    train_data = CustomDataset(path_to_train_images, path_to_train_masks, transform=False)
    valid_data = CustomDataset(path_to_valid_images, path_to_valid_masks, transform=False)
    train_loader = DataLoader(train_data, batch_size=8, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=1, drop_last=False)

    # stuff for training
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.00)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    loss_weights = torch.tensor([1., 3., 1., 1.]).cuda()
    loss = nn.CrossEntropyLoss(weight=loss_weights)
    for param in model.parameters():
        param.requires_grad = True

    # evaluation
    miou = IoU(num_classes=num_custom_classes, reduction='none').cuda()
    best_miou = 0.0

    # training loop
    for epoch in range(epochs):
        model.train()
        train_batch = tqdm(enumerate(train_loader), total=len(train_loader))
        train_batch.set_description(f'Epoch {epoch}')
        losses = []
        for i, sample in train_batch:
            # import ipdb; ipdb.set_trace()
            image, sem_mask = sample
            image, sem_mask = image.cuda(), sem_mask.cuda()
            pred_scales = model(image/255)
            sem_mask = sem_mask.squeeze().long()
            l = loss(pred_scales, sem_mask)
            l.backward()
            optimizer.step()
            losses.append(l.cpu().detach().numpy())
        print('Loss:', np.mean(np.array(losses)))
        print('Finished epoch. Starting validation...')

        # validation loop
        ious = np.zeros(num_custom_classes)
        count = np.zeros(num_custom_classes)
        for i, sample in enumerate(valid_loader):
            model.eval()
            with torch.no_grad():
                image, sem_mask = sample
                image, sem_mask = image.cuda(), sem_mask.cuda()
                pred_scales = model(image/255)
                sem_mask = sem_mask.squeeze().long()
                prediction = torch.argmax(torch.softmax(pred_scales, dim=1), dim=1)
                iou = miou(prediction, sem_mask)
                ious += iou.cpu().numpy()
                classes = torch.unique(sem_mask).cpu().numpy()
                count[classes] += 1
        ious /= count
        current_miou = np.mean(np.array(ious))
        print('miou in validation:', ious) # current_miou)

        # save models
        if ious[1] > best_miou:
            best_miou = ious[1]
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            ckpt_model_filename = "model_best_miou_w3noaug.pth"
            torch.save(state, ckpt_model_filename)
            print('Saved best model.')
        
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        ckpt_model_filename = "model_latest_w3noaug.pth"
        torch.save(state, ckpt_model_filename)
        print('Saved last model.\n')

        scheduler.step()

    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
