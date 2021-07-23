#include <iomanip>
#include <iostream>

#include <scorch.hpp> 

template<typename T>
void td();

int main() {
    constexpr std::size_t InputDim = 4;
    constexpr std::size_t HiddenDim = 16;
    constexpr std::size_t OutputDim = 4;

    auto W0 = scorch::rand<float, HiddenDim, InputDim>();
    auto b0 = scorch::rand<float, HiddenDim>();
    auto W1 = scorch::rand<float, HiddenDim, HiddenDim>();
    auto b1 = scorch::rand<float, HiddenDim>();
    auto W2 = scorch::rand<float, OutputDim, HiddenDim>();
    auto b2 = scorch::rand<float, OutputDim>();

    auto opt = scorch::SGD(0.1f, 0.8f, W0, b0, W1, b1);

    constexpr std::size_t BatchDim = 16;

    constexpr auto num_iterations = 100;

    for (auto i = 0; i < num_iterations; ++i) {
        auto x = scorch::rand<float, BatchDim, InputDim>();
        auto y = copy(x);

        auto y_hat = sigmoid(sigmoid(x % W0 + b0) % W1 + b1) % W2 + b2;

        auto l = mean((y_hat - y) ^ 2.0f);

        static_assert(l.Scalar);

        opt.zero_grad();

        l.backward();

        opt.step();

        std::cout << "l = " << l << std::endl;

        if (i == 0 || (i + 1) == num_iterations) {
            std::cout << "  x = " << x << std::endl;
            std::cout << "  y = " << y << std::endl;
        }
    }


    return 0;
}