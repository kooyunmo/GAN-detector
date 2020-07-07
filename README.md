# GAN-detector
Fake image detection model that can also classify which GAN was used to generate the fake images

## Directory Structure
```
GAN-dectector
    ├ models
    |    ├--- Xception
    |    └--- ResNet
    └ utils
         ├ args.py (argument parsing) 
         ├ preprocessing.py
         ├ train.py
         └ test.py
```

## Overview
This is manipulated image detection models which can be globally applied to multiple GAN images. 