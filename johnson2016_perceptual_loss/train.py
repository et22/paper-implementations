import os
import torch
import wandb
import torchvision.utils as vutils

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
        self.parser.set_defaults(resume_training=False)
    def parse(self):
        opts = self.parser.parse_args()
        return opts

def train(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    #### parameters ####
    lr = .001
    data_dir = args.data_path
    image_size = 256
    workers = 8
    batch_size = 16
    num_epochs = 5
    style_img = "./style/picasso_studio_with_plaster_head.jpeg"
    style_label = ""
    style_weight = 20
    content_weight = 10
    tv_weight = 1
    display_iters = 5
    save_epochs = 1
    checkpoint_dir = "./models"
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    net = ImageTNet().to(device)
    criterion = PerceptualLoss(style_weight, content_weight, tv_weight).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    style_target = transform(Image.open(style_img)).to(device)[None, :, :, :]

    # if resume from checkpoint, load model and optimizer state dicts from checkpoint
    if args.resume_training:
        net.load_state_dict(torch.load(args.checkpoint_net_path))
        optimizer.load_state_dict(torch.load(args.checkpoint_optim_path))

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            net.zero_grad()
            output = net(data.to(device))
            target = style_target.repeat(output.shape[0], 1, 1, 1)
            loss = criterion(output=output, style_target=target, content_target=data.to(device))
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                # Display training stats and log images every display_iters iterations
                if i % display_iters == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f' % (epoch, num_epochs, i, len(dataloader), loss.item()))

                    output = output.detach().cpu()

                    generated_images = wandb.Image(vutils.make_grid(output.add(1).mul(0.5), padding=2, normalize=True))
                    product_images = wandb.Image(vutils.make_grid(data.add(1).mul(0.5), padding=2, normalize=True))
                    if logging:
                        wandb.log({"generated images": generated_images})
                        wandb.log({"product images": product_images})
                    else:
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        vutils.save_image(output.add(1).mul(0.5),  save_dir + 'out_%d.jpg'%i, nrow=4)
                        vutils.save_image(data.add(1).mul(0.5).mul(0.5), save_dir +'dat_%d.jpg'%i, nrow=4)

            # Save checkpoint every save_epochs epochs
            if (epoch % save_epochs == 0):
                with torch.no_grad():
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)

                    # save optimizers
                    torch.save(net.state_dict(),os.path.join(checkpoint_dir, f"net_{style_label}_{epoch}.pt"))
                    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f"opt_{style_label}_{epoch}.pt"))


if __name__ == '__main__':
    freeze_support()
    args = TrainOptions().parse()
    train(args)