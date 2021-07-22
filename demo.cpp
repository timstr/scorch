#include <iomanip>
#include <iostream>

#include <scorch.hpp> 



int main() {
    auto a = scorch::Tensor<float, 1>{0.0f};
    auto b = scorch::Tensor<float, 1>{3.0f};

    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;

    auto y = exp((sin(a * a) + b) / 4.0f) + 9.5f;

    std::cout << "y = " << y << std::endl;

    y.backward();

    std::cout << "dy/da = " << a.grad() << std::endl;
    std::cout << "dy/db = " << b.grad() << std::endl;


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