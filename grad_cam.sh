# resnet: model.layer4
# xception: model.conv4

python test_cam.py demo1 \
    -a xception \
    -t model.block12 \
    -i datasets/test/pggan/00876.png
