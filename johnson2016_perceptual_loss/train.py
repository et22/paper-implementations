import os
import torch
import wandb
import torchvision.utils as vutils
import torch.nn as nn

from torchvision import transforms
from torch.utils.data import DataLoader

from PIL import Image
from model import ImageTNet
from loss import PerceptualLoss
from dataset import StyleDataset
from multiprocessing import freeze_support
from argparse import ArgumentParser

class TrainOptions:
    """
    TrainOptions defines arguments for train.py. 
    """
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--data_path', type=str, default='./data/', help='Path to data directory with subdirectories training and validation')
        self.parser.add_argument('--checkpoint_optim_path', type=str, default=None, help='Path to optimizer checkpoint if resuming training')
        self.parser.add_argument('--checkpoint_net_path', type=str, default=None, help='Path to model checkpoint if resuming training')
        self.parser.add_argument('--resume_training', dest='resume_training', action='store_true', help='Whether to resume training from checkpoint')
        self.parser.add_argument('--no-resume_training', dest='resume_training', action='store_false', help='Whether to resume training from checkpoint')
        self.parser.add_argument('--style_image_path', type=str, default="./style/picasso_the_poet.jpeg", help='Path to style image')
        self.parser.set_defaults(resume_training=False)
        self.parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        self.parser.add_argument('--num_epochs', type=int, default=25, help='Epochs')
        self.parser.add_argument('--log_iters', type=int, default=25, help='How often to log loss and images')
        self.parser.add_argument('--checkpoint_dir', type=str, default="./", help='Where to log checkpoints')
        self.parser.add_argument('--dl_workers', type=int, default=2, help='Number of dataloader workers')
        self.parser.add_argument('--save_epochs', type=int, default=2, help='How often to save checkpoints')
        self.parser.add_argument('--style_weight', type=int, default=15, help='Weight of style in loss function')
    def parse(self):
        opts = self.parser.parse_args()
        return opts

def train(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #### parameters ####
    lr = .001
    data_dir = args.data_path
    image_size = 256
    workers = args.dl_workers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    style_img = args.style_image_path
    style_label = ""
    style_weight = args.style_weight
    content_weight = 1
    tv_weight = .5
    display_iters = args.log_iters
    save_epochs = args.save_epochs
    checkpoint_dir = args.checkpoint_dir
    save_dir = "./temp/"
    logging = 1
    ####################
    if logging:
        wandb.init(project='papertest', entity='et22', name="test1", settings=wandb.Settings(start_method="fork"))

    transform = transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])

    dataset = StyleDataset(path=data_dir, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    net = ImageTNet().to(device)
    criterion = PerceptualLoss(style_weight, content_weight, tv_weight).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    style_target = transform(Image.open(style_img)).to(device)[None, :, :, :]
    style_target = style_target.repeat(batch_size, 1, 1, 1)
    target_log = wandb.Image(vutils.make_grid(style_target.add(1).mul(0.5), padding=2, normalize=True))
    if logging:
        wandb.log({"style target": target_log})

    # if resume from checkpoint, load model and optimizer state dicts from checkpoint
    if args.resume_training:
        net.load_state_dict(torch.load(args.checkpoint_net_path))
        optimizer.load_state_dict(torch.load(args.checkpoint_optim_path))

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        # Save checkpoint every save_epochs epochs
        if (epoch % save_epochs == 0):
            with torch.no_grad():
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # save optimizers
                torch.save(net.state_dict(),os.path.join(checkpoint_dir, f"net_{style_label}_{epoch}.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"opt_{style_label}_{epoch}.pt"))

        for i, data in enumerate(dataloader):
            net.zero_grad()
            output = net(data.to(device))
            loss, style_loss, content_loss, tv_loss = criterion(output=output, style_target=style_target, content_target=data.to(device))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Display training stats and log images every display_iters iterations
                if i % display_iters == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f\tStyle Loss: %.4f\tContent Loss: %.4f\tTV Loss: %.4f' % (epoch, num_epochs, i, len(dataloader), loss.item(), style_loss.item(), \
                        content_loss.item(), tv_loss.item()))

                    output = output.detach().cpu()

                    generated_images = wandb.Image(vutils.make_grid(output.add(1).mul(0.5), padding=2, normalize=True))
                    input_images = wandb.Image(vutils.make_grid(data.add(1).mul(0.5), padding=2, normalize=True))
                    if logging:
                        wandb.log({"generated images": generated_images})
                        wandb.log({"input images": input_images})
                    else:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        vutils.save_image(output.add(1).mul(0.5),  save_dir + 'out_%d.jpg'%i, nrow=4)
                        vutils.save_image(data.add(1).mul(0.5), save_dir +'dat_%d.jpg'%i, nrow=4)



if __name__ == '__main__':
    freeze_support()
    args = TrainOptions().parse()
    train(args)