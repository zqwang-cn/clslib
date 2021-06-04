import os
import random
import argparse
import torch
import torchvision
from tools import get_model, get_dataset, load_checkpoint, train, val


def parse():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--arch', type=str, default='resnet18', help='model arch')
    # data
    parser.add_argument('--data_root', type=str, default='data/car_type', help='dataset root')
    parser.add_argument('--image_size', nargs=2, type=int, default=[256, 256], help='image input size')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=16, help='worker number')
    # train
    parser.add_argument('--n_epochs', type=int, default=100, help='iteration number')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume training from last checkpoint')
    parser.add_argument('--val', action='store_true', help='is validation')
    # checkpoint
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='checkpoint dir')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='checkpoint saving interval')
    # other
    parser.add_argument('--seed', type=int, default=123456, help='random seed')
    parser.add_argument('--gpus', type=str, default='0', help='used gpu ids')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse()
    print(opt)

    random.seed(opt.seed)
    if not os.path.exists(opt.ckpt_dir):
        os.mkdir(opt.ckpt_dir)

    # Prepare dataset
    split = 'val' if opt.val else 'train'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ColorJitter(0.5),
        torchvision.transforms.Resize(opt.image_size),
        torchvision.transforms.ToTensor()
    ])
    dataset = get_dataset(opt.data_root, split, transform)
    dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size, True, num_workers=opt.num_workers)

    # Prepare model
    n_classes = len(dataset.classes)
    model = get_model(opt.arch, n_classes)
    if opt.gpus:
        gpus = list(map(int, opt.gpus.split(',')))
        model = torch.nn.DataParallel(model.cuda(), gpus)

    if opt.val:
        # Load last checkpoint and validate
        epoch = load_checkpoint(opt.ckpt_dir, -1, model)
        val(model, dataloader)
    else:
        # Prepare criterion and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        # Load checkpoint if resume
        if opt.resume:
            begin_epoch = load_checkpoint(opt.ckpt_dir, -1, model, optimizer) + 1
        else:
            begin_epoch = 1
        # Start training
        train(model, dataloader, criterion, optimizer, begin_epoch, opt.n_epochs, opt.ckpt_dir, opt.ckpt_interval)
