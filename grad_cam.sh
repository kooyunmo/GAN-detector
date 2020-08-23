# resnet: model.layer4
# xception: model.conv4

python test_cam.py demo1 \
    -a xception \
    -t model.conv4 \
    -i datasets/train/msgstylegan/09899.png
