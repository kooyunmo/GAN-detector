python main.py \
    --phase train \
    --data-dir ./datasets \
    --model-name xception \
    --model-path ./checkpoints/gan-detection-xception.h5 \
    --num-epochs 3 \
    --batch-size 16 \
    --save-dir ./checkpoints \
    --gpu 1