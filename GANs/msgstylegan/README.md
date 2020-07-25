# MSG-StyleGAN: celebahq_generate_samples.py

* A slight adjustment of [generate_samples.py](https://github.com/akanimax/msg-stylegan-tf/blob/master/generate_samples.py)
* Generate fake face images using MSG-StyleGAN pretrained by CelebA HQ dataset



![](./celebahq_examples/ensemble.png)



### Environment

The code was built and tested for:

* Conda 4.8.3 (Windows 10)
* 64-bit Python 3.6.7
* TensorFlow 1.13.1 with GPU support
* NVIDIA GeForce GTX 970
* CUDA toolkit 10.0.130
* cuDNN 7.6.5



### How to Use

1. Download the pretrained model: [celebahq_6.37.pkl](https://drive.google.com/file/d/1IP7J-a3HT7EcuHrb0Qz9fsA2CVTeDz08/view) (Do **NOT** change the name of the file)

2. Run `celebahq_generate_samples.py` script. Please see the example below:

```
(your_virtual_env) $ python celebahq_generate_samples.py \
--num_samples [NUM_SAMPLES, default=100] \
--random_random_state \
--random_state [RANDOM_STATE, default=33] \
--output_path [OUTPUT_PATH, default="celebahq_samples"]
```



#### Arguments

* `num_samples`: Integer, Number of samples to be generated. You need to adjust the code if bigger than `99999`.
* `random_random_state`: (optional) Use this when you want to select random_state (seed) randomly between 1 and 1000.
* `random_state`: Integer, Random state (seed) for the script to run. Meaningless if `random_random_state` is used.
* `output_path`: String, (Absolute / Relative) Path to directory for saving the files.



