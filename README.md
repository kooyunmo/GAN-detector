# GAN-detector
Fake image detection model that can also classify which GAN was used to generate the fake images

## Directory Structure
```
GAN-dectector
├── cam_results
│   ├── demo1
│   └── demo2
├── checkpoints
│   ├── gan-detection-resnet101.h5
│   ├── gan-detection-resnet50.h5
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
├── grad_cam.sh
├── layer_grad_cam.sh
├── main.py
├── models
│   ├── models.py
│   └── Xception
│       ├── __init__.py
│       ├── xception-b5690688.pth
│       └── xception.py
├── README.md
├── requirements.txt
├── resnet101_training_log.png
├── resnet50_training_log.png
├── test_cam.py
├── test.py
├── train.sh
├── utils
│   ├── grad_cam.py
│   ├── plot.py
│   └── preprocess.py
└── xception_training_log.png
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
**1) Grad-CAM visualization for a target layer**
```
// run with a shell script
$ sh grad_cam.sh

// manually run
$ python test_cam.py demo1 \
      -a {model name}
      -t {layer name}
      -i {image path}
```

**2) Grad-CAM map for a specific class at different layers**
```
// run with a shell script
$ sh layer_grad_cam.sh

$ python test_cam.py demo2 \
      -a xception \
      -i datasets/test/pggan/00876.png \
      -c pggan
```