python main.py \
    --phase train \
    --data-dir ./datasets \
    --model-name xception \
    --model-path ./checkpoints/gan-detection-xception.h5 \
    --num-epochs 500 \
    --batch-size 32 \
    --save-dir ./checkpoints \
    --gpu 1