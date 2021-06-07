import os
import argparse
import tqdm
import torch
import torchvision


def parse_args():
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
    # export
    parser.add_argument('--load', type=int, default=-1, help='epoch of checkpoint to load')
    parser.add_argument('--export-pt', type=str, default='', help='pt filename to export')
    parser.add_argument('--export-onnx', type=str, default='', help='onnx filename to export')
    # other
    parser.add_argument('--seed', type=int, default=123456, help='random seed')
    parser.add_argument('--gpus', type=str, default='0', help='used gpu ids')

    opt = parser.parse_args()
    return opt


def get_model(arch, n_classes):
    """Create model according to arch name.

    Args:
        arch (str): model arch name
        n_classes (int): total class number

    Raises:
        RuntimeError: raise error if arch name not supported

    Returns:
        torch.nn.Module: created model
    """
    func = getattr(torchvision.models, arch)
    if func:
        return func(num_classes=n_classes)
    else:
        raise RuntimeError("No such arch name: %s" % arch)


def get_dataset(data_root, split, transform):
    """Create dataset.

    Args:
        data_root (str): dataset root dir
        split (str): dataset split
        transform (torch.nn.Module): transformation for input images

    Returns:
        torch.utils.data.Dataset: created dataset
    """
    dataset = torchvision.datasets.ImageFolder(os.path.join(data_root, split), transform)
    return dataset


def save_checkpoint(ckpt_dir, epoch, model, optimizer):
    """Save model and optimizer state dict to checkpoint.

    Args:
        ckpt_dir (str): checkpoint dir
        epoch (int): epoch index to save
        model (torch.nn.Module): model to save
        optimizer (torch.optim.Optimizer): optimizer to save
    """
    if hasattr(model, 'module'):
        model = model.module
    # Save checkpoint
    data = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    ckpt_filename = '%d.pth' % epoch
    torch.save(data, os.path.join(ckpt_dir, ckpt_filename))

    # Add link to last checkpoint
    last_ckpt_link = os.path.join(ckpt_dir, 'last.pth')
    if os.path.exists(last_ckpt_link):
        os.remove(last_ckpt_link)
    os.symlink(ckpt_filename, last_ckpt_link)


def load_checkpoint(ckpt_dir, epoch, model, optimizer=None):
    """Load model and optimizer state dict from checkpoint.

    Args:
        ckpt_dir (str): checkpoint dir
        epoch (int): epoch index to load. -1 for last epoch
        model (torch.nn.Module): model to load
        optimizer (torch.optim.Optimizer, optional): optimizer to load. Defaults to None.
    Returns:
        int: epoch index of loaded checkpoint
    """
    if hasattr(model, 'module'):
        model = model.module
    ckpt_filename = '%d.pth' % epoch if epoch != -1 else 'last.pth'
    data = torch.load(os.path.join(ckpt_dir, ckpt_filename))
    model.load_state_dict(data['model'])
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer'])
    return data['epoch']


def train(model, dataloader, criterion, optimizer, begin_epoch, n_epochs, ckpt_dir, ckpt_interval):
    """Train a model.

    Args:
        model (torch.nn.Module): model to train
        dataloader (torch.utils.data.Dataloader): dataloader
        criterion (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        begin_epoch (int): begin epoch index
        n_epochs (int): total epoch number
        ckpt_dir (str): checkpoint dir
        ckpt_interval (int): checkpoint saving interval
    """
    model.train()
    for epoch in range(begin_epoch, n_epochs+1):
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm.tqdm(dataloader, ascii=True)
        pbar.set_description('epoch %d,loss:-,acc:-' % epoch)
        for input, label in pbar:
            input = input.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss
            correct += (torch.argmax(output, 1) == label).sum()
            total += input.size(0)

            pbar.set_description('epoch %d,loss:%f,acc:%f' % (epoch, running_loss/total, float(correct)/total))
        if epoch % ckpt_interval == 0:
            save_checkpoint(ckpt_dir, epoch, model, optimizer)


def val(model, dataloader):
    """Validate a model.

    Args:
        model (torch.nn.Module): model to validate
        dataloader (torch.utils.data.Dataloader): dataloader
    """
    correct = 0
    total = 0
    pbar = tqdm.tqdm(dataloader, ascii=True)
    pbar.set_description('val acc:-')
    model.eval()
    with torch.no_grad():
        for input, label in pbar:
            input = input.cuda()
            label = label.cuda()
            output = model(input)
            correct += (torch.argmax(output, 1) == label).sum()
            total += input.size(0)
            pbar.set_description('val acc:%f' % (float(correct)/total))
