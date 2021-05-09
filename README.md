# mnist-from-scratch

This is an implementation of a simple linear model to classify handwritten digits from the MNIST dataset using `C` and `Cuda`. It doesn't use any libraries such as `PyTorch` or `Tensorflow`, and it was not designed to be extensible. This was really just practice for me to learn how to use `CUDA`, and learn about backpropagation & gradient descent at a low level. This implementation is also not memory optimized. It uses pitched memory allocations for the CUDA kernels as this is supposedly faster than just using non-strided arrays, but other than that there are no speed optimizations.

To compile, simply run `make install`. Note, to compile and run this program, you need to have CUDA, NVidia drivers, and an NVidia GPU.
