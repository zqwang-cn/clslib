import sys
import torch
from tools import parse_args, get_model, get_dataset, load_checkpoint


if __name__ == '__main__':
    opt = parse_args()
    print(opt)

    if not opt.export_pt and not opt.export_onnx:
        print('Please set export args.')
        sys.exit()

    # Get dataset class number
    dataset = get_dataset(opt.data_root, 'train', None)
    n_classes = len(dataset.classes)

    # Create model
    model = get_model(opt.arch, n_classes)

    # Load checkpoint
    load_checkpoint(opt.ckpt_dir, opt.load, model)

    # Create dummy input
    image_size = opt.image_size
    dummy_input = torch.rand(1, 3, image_size[0], image_size[1])

    # Export
    if opt.export_pt:
        traced_script_module = torch.jit.trace(model, dummy_input)
        traced_script_module.save(opt.export_pt)
    if opt.export_onnx:
        torch.onnx.export(model, dummy_input, opt.export_onnx, verbose=False)
