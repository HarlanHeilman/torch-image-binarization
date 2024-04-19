# torch-image-binarization

![test input image](test_image.png)
![test output image](test_image-binarized.png)

Local image binarization algorithms implemented in Pytorch. Includes the [Otsu thresholding algorithm](https://en.wikipedia.org/wiki/Otsu's_Method) and the algorithm by [Su et al.](https://doi.org/10.1145/1815330.1815351) (which is an extension of the Otsu algorithm). The algorithms are implemented to optimize performance. With `torch.compile` it is approximately 4125x faster than the [CPU-based version](https://github.com/nopperl/vectorized-image-binarization).

This was written to test the benefits of `torch.compile`. For a log of performance improvements, refer to [`optimizations.ipynb`](optimizations.ipynb). The notebook expects `triton==3.0.0` and `torch==2.3.0`.

## Install

Install using `pip`:

    pip install https://github.com/nopperl/torch-image-binarization

The package requires `torch>=2.2.0` and optionally `triton>=2.21`, which need to be installed seperately, e.g. using `pip`:

    pip install torch torchvision triton

## Usage

Read an image:
```
from torchvision.io import ImageReadMode, read_image 
img = read_image("test_image.png", mode=ImageReadMode.GRAY)
```

Binarize the image:
```
from torck_image_binarization.thresholding import su
su(img)
```

For more information, refer to `torch_image_binariztion/main.py` 

## Benchmark

To show the performance gains of the CUDA-based PyTorch implementation over the CPU-based NumPy implementation and the benefits of `torch.compile`, the runtime is measured across different input image sizes. For more information, refer to [`optimizations.ipynb`](optimizations.ipynb).

Results:

```
[------------------------------------ su -------------------------------------]
                                                                    |   runtime
1 threads: --------------------------------------------------------------------
      numpy                                                         | 3548992.0
      su(img)                                                       |   10426.5
      torch.compile(su)(img)                                        |    1333.6
      torch.compile(su, mode='reduce-overhead')(img)                |     858.8
      torch.compile(su, mode='max-autotune')(img)                   |     859.6
      torch.compile(su, dynamic=True)(img)                          |     859.7
      torch.compile(su, dynamic=True, mode='reduce-overhead')(img)  |     860.0
      torch.compile(su, dynamic=True, mode='max-autotune')(img)     |     860.0

Times are in microseconds (us).
```
