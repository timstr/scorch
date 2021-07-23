# Scorch

A lightweight, cross-platform, header-only library written standard C++ for tensor arithmetic with automatic differentiation, designed to closely mimick the famous PyTorch library in usage and appearance but with improved compile-time safety checks.

Here's an example that trains a two-layer neural network to learn the identity function. This code can be run in `demo.cpp`. Most of this should look and feel immediately familiar to a PyTorch user.
```C++
// layer sizes
constexpr std::size_t InputDim = 4;
constexpr std::size_t HiddenDim = 16;
constexpr std::size_t OutputDim = 4;

// learnable network parameters
auto W0 = scorch::rand<float, HiddenDim, InputDim>();
auto b0 = scorch::rand<float, HiddenDim>();
auto W1 = scorch::rand<float, HiddenDim, HiddenDim>();
auto b1 = scorch::rand<float, HiddenDim>();
auto W2 = scorch::rand<float, OutputDim, HiddenDim>();
auto b2 = scorch::rand<float, OutputDim>();

// optimizer
// learning rate, momentum ratio, parameters...
auto opt = scorch::optim::SGD(0.1f, 0.8f, W0, b0, W1, b1);

// batch size
constexpr std::size_t BatchDim = 16;

for (auto i = 0; i < 100; ++i) {
    // random input
    auto x = scorch::rand<float, BatchDim, InputDim>();

    // identity function: output is equal to input
    auto y = copy(x);

    // compute the network output
    // Yes, it's actually this simple
    auto y_hat = sigmoid(sigmoid(x % W0 + b0) % W1 + b1) % W2 + b2;

    // compute the loss
    auto l = mean((y_hat - y) ^ 2.0f);

    // don't forget to zero the gradients before back-propagation
    opt.zero_grad();

    // compute the gradients of all parameters w.r.t. the loss
    l.backward();

    // take a training step
    opt.step();
}
```

Notable features:
 - Support for vector, matrix, and tensor variables with arbitrarily many dimensions
 - Support for scalar variables
 - Element-wise functions, broadcasting semantics*, matrix-vector mulitplication, and more.
 - The usual overloaded operators, plus `%` for matrix-vector multiplication and `^` for exponentiation.
 - Extremely ergonomic syntax for writing expressions (see the example)
 - Compile-time checking of tensor shape compatibility (!!!)
 - Automatic differentiation using reverse-mode gradient computation
 - Dynamic computational graphs
 - Optimizers (only SGD for now)
 - Tested with MSVC, GCC, and Clang

\* Broadcasting semantics are only supported for pairs of tensors whose shapes are identical except that one may have additional higher dimensions. For example, a size 3x5x7 tensor is broadcastable with a size 5x7 tensor and a size 7 tensor, but a size 3x5x7 tensor is **not** broadcastable with a size 1x1x7 tensor, or a size 1x1x1 tensor.

Features that are not supported but are probably coming soon:
 - Tensor views, clever indexing, and differentiation through tensor scalar element access
 - Convolutions
 - Matrix-matrix multiplication
 - Some remaining basic mathematical functions (e.g. `cbrt`, `atan`, etc...)
 - Smarter optimizers (e.g. Adam, RMSProp, if I can understand them)
 - Higher-order derivatives (maybe)

Features that not supported and probably never will be:
 - GPU acceleration
 - Dynamically-sized tensors

This code was written by Tim Straubinger and is made available for free use under the MIT license.
