#include <iomanip>
#include <iostream>

#include <scorch.hpp> 



int main() {

    auto A = scorch::Tensor<float, 2, 2>{};
    A(0, 0) = 1.0f;
    A(0, 1) = 0.0f;
    A(1, 0) = 0.0f;
    A(1, 1) = 1.0f;

    auto x = scorch::Tensor<float, 2>{};
    x(0) = 3.0f;
    x(1) = 5.0f;

    auto opt = scorch::SGD(0.01f, x);

    for (auto i = 0; i < 1000; ++i) {
        auto y = sum(square(matvecmul(x, A)));

        static_assert(y.Scalar);

        opt.zero_grad();

        y.backward();

        opt.step();

        std::cout << "x = " << x;
        std::cout << "  y = " << y << std::endl;
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