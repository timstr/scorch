#include <iomanip>
#include <iostream>

#include <scorch.hpp> 



int main() {

    // constexpr std::size_t InputDim = 1;
    // constexpr std::size_t HiddenDim = 16;
    // constexpr std::size_t OutputDim = 1;

    // auto W0 = scorch::rand<float, HiddenDim, InputDim>();
    // auto b0 = scorch::rand<float, HiddenDim>();
    // auto W1 = scorch::rand<float, OutputDim, HiddenDim>();
    // auto b1 = scorch::rand<float, OutputDim>();

    constexpr std::size_t D = 8;
    auto A = scorch::rand<float, D, D>();

    auto opt = scorch::SGD(0.1f, 0.8f, A);

    for (auto i = 0; i < 1000; ++i) {
        constexpr std::size_t B = 1024;

        // auto x = scorch::rand<float, InputDim>();
        auto x = scorch::rand<float, B, D>();
        auto y = copy(x);

        // auto y_hat = sigmoid(x % W0 + b0) % W1 + b1;
        auto y_hat = x % A;

        auto l = mean((y_hat - y) ^ 2.0f);

        static_assert(l.Scalar);

        opt.zero_grad();

        l.backward();

        opt.step();

        std::cout << "l = " << l << std::endl;
        std::cout << "A = " << A << std::endl;
        // std::cout << "  x = " << x << std::endl;
        // std::cout << "  y = " << y << std::endl;
    }


    // auto t = scorch::TensorStorage<float, 1, 2, 3, 4>{};

    // for (const auto& d : t.Dims) {
    //     std::cout << d << ' ';
    // }
    // std::cout << "\n\n";

    // t(0, 0, 0, 0) = 0;
    // t(0, 0, 0, 1) = 1;
    // t(0, 0, 0, 2) = 2;
    // t(0, 0, 0, 3) = 3;
    // t(0, 0, 1, 0) = 4;
    // t(0, 0, 1, 1) = 5;
    // t(0, 0, 1, 2) = 6;
    // t(0, 0, 1, 3) = 7;
    // t(0, 0, 2, 0) = 8;
    // t(0, 0, 2, 1) = 9;
    // t(0, 0, 2, 2) = 10;
    // t(0, 0, 2, 3) = 11;
    // t(0, 1, 0, 0) = 12;
    // t(0, 1, 0, 1) = 13;
    // t(0, 1, 0, 2) = 14;
    // t(0, 1, 0, 3) = 15;
    // t(0, 1, 1, 0) = 16;
    // t(0, 1, 1, 1) = 17;
    // t(0, 1, 1, 2) = 18;
    // t(0, 1, 1, 3) = 19;
    // t(0, 1, 2, 0) = 20;
    // t(0, 1, 2, 1) = 21;
    // t(0, 1, 2, 2) = 22;
    // t(0, 1, 2, 3) = 23;

    // std::cout << t;

    // auto s = scorch::sin(t);

    // std::cout << s << std::endl;

    // std::cout << (s + t) << std::endl;

    return 0;
}