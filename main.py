import os
import random
import torch
import torchvision
from tools import parse_args, get_model, get_dataset, load_checkpoint, train, val


if __name__ == '__main__':
    opt = parse_args()
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
