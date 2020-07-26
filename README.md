# GAN-detector
Fake image detection model that can also classify which GAN was used to generate the fake images

## Directory Structure
```
GAN-dectector
    ├- datasets
    |    ├- train
    |    |     ├- msgstylegan
    |    |     ├- pggan
    |    |     ├- stylegan
    |    |     ├- vgan 
    |    |     └- real
    |    └- test 
    |          ├- msgstylegan
    |          ├- pggan
    |          ├- stylegan
    |          ├- vgan 
    |          └- real 
    ├- models
    |    ├- models.py
    |    ├- Xception
    |    └- ResNet
    ├- utils
    |     ├- args.py (argument parsing) 
    |     └- preprocessing.py
    ├- train.py
    └- test.py
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
TODO: Fill this
```

### Test
```
TODO: Fill this
```