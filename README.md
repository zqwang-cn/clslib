# A simple classification lib based on PyTorch

## Dependencies

```bash
pip install -r requirements.txt
```

## Dataset folder structure

```
data_root
├── train
│   ├── class_1
│   │   ├── a.jpg
│   │   └── b.jpg
│   └── class_2
│       ├── c.jpg
│       └── d.jpg
└── val
    ├── class_1
    └── class_2

```

### Usage

Train

```bash
python main.py --arch resnet18 --data_root data/data_root --gpus 0,1,2,3
```

Validate

```bash
python main.py --arch resnet18 --data_root data/data_root --val
```
