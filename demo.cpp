#include <iomanip>
#include <iostream>

#include <scorch.hpp> 

template<typename T>
void td();

int main() {
    // Train a simple neural network with 4 inputs, 4 outputs,
    // and two hidden layers with 16 neurons each to learn
    // the identity function

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
    // learning rate, momentum ration, parameters...
    auto opt = scorch::SGD(0.1f, 0.8f, W0, b0, W1, b1);

    // batch size
    constexpr std::size_t BatchDim = 16;


    constexpr auto num_iterations = 100;
    for (auto i = 0; i < num_iterations; ++i) {

        // random input
        auto x = scorch::rand<float, BatchDim, InputDim>();

        // identity function: output is equal to input
        auto y = copy(x);

        // compute the network output
        // I love how operator overloading and  ADL work here
        auto y_hat = sigmoid(sigmoid(x % W0 + b0) % W1 + b1) % W2 + b2;

        // compute the loss
        auto l = mean((y_hat - y) ^ 2.0f);

        // don't forget to zero the gradients before back-propagation
        opt.zero_grad();

        // compute the gradients of all parameters w.r.t. the loss
        l.backward();

        // take a training step
        opt.step();

        std::cout << "l = " << l << std::endl;

        if (i == 0 || (i + 1) == num_iterations) {
            std::cout << "  x = " << x << std::endl;
            std::cout << "  y = " << y << std::endl;
        }
    }


    return 0;
}