# pretrained_celebahq.py

### A bit fine-tuning of pretrained_example.py

* Use CelebA-HQ pretrained model
* Set the number of generated images

![example](./stylegan_celebahq_examples/ensemble.png)

### Environment

* Windows 10
* NVIDIA GeForce GTX 970
* Python 3.6.10
* conda 4.8.3
* tensorflow-gpu 1.10.0
* numpy 1.18.5
* cuda 9.0
* Newest pillow, image, requests, ...



### How to Use

1. [Download](https://drive.google.com/file/d/1MT9USX2Q8rKxDtVBs1fHn6pw7PRXhqvF/view?usp=sharing) pretrained model (.pkl) and move it to the home directory (the same directory with README)
2. Changing the parameter in "rnd = np.random.RandomState()" may provide different results.
3. Run the following script
```
python pretrained_celebahq.py \
    --img-num-start [START_IMG_NUM] \
    --num-imgs [NUM_IMAGES] \
    --result-dir [RESULT_DIR] \
    --gpu-num [GPU_ID]
```