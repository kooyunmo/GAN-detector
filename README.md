# GAN-detector
Fake image detection model that can also classify which GAN was used to generate the fake images

## Directory Structure
```
GAN-dectector
├── checkpoints
│   ├── gan-detection-resnet101.h5
│   └── gan-detection-xception.h5
├── datasets
│   ├── test
│   │   ├── msgstylegan
│   │   ├── pggan
│   │   ├── stylegan
│   │   └── vgan
│   └── train
│       ├── msgstylegan
│       ├── pggan
│       ├── stylegan
│       └── vgan
├── GANs
│   ├── msgstylegan
│   ├── pggan
│   ├── stylegan
│   └── vgan
├── main.py
├── models
│   ├── models.py
│   └── Xception
│       ├── xception-b5690688.pth
│       └── xception.py
├── notebooks
│   └── ResNet_Fake_Face.ipynb
├── README.md
├── requirements.txt
├── run.sh
├── test.py
└── utils
    ├── plot.py
    └── preprocess.py
```

## Overview
This is manipulated image detection models which can be globally applied to multiple GAN images.

## How To Use
### Prerequisite
```
# install dependent python packages
$ pip install -r requirements.txt

# download pretrained checkpoint for Xception
$ wget http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth -P [path to GAN-detector/models/Xception]
```

### Train
```
// run with a shell script
$ sh train.sh

// manually run
$ python main.py \
      --phase train \
      --data-dir ./datasets \
      --model-name {resnet101, xception, ...} \
      --model-path ./checkpoints/gan-detection-{resnet101, xception, ...}.h5 \
      --num-epochs {the number of training epochs} \
      --batch-size {batch size} \
      --save-dir ./checkpoints \
      --gpu {GPU PCI ID}
```

### Test
```
TODO: Fill this
```

### Result Analysis with Grad-CAM
```
// run with a shell script
$ sh grad_cam.sh

// manually run
$ python test_cam.py demo1 -a {model name} -t {layer name} -i {image path}
```