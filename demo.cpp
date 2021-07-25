#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <scorch.hpp> 

constexpr float domain_min = -1.0f;
constexpr float domain_max = 1.0f;
constexpr std::size_t graphWidth = 70;
constexpr std::size_t graphHeight = 30;

void print(std::function<float(float)> fn){
    std::array<float, graphWidth> values;
    for (std::size_t gx = 0; gx < graphWidth; ++gx){
        const float t = (static_cast<float>(gx) / static_cast<float>(graphWidth - 1));
        const float x = t * (domain_max - domain_min) + domain_min;
        values[gx] = fn(x);
    }

    for (std::size_t gy = 0; gy < graphHeight; ++gy){
        const auto y = static_cast<float>(gy);
        for (std::size_t gx = 0; gx < graphWidth; ++gx){
            const auto h = (0.5f * (1.0f - values[gx])) * static_cast<float>(graphHeight);
            if (h > y + 1.0f){
                std::cout << ' ';
            } else if (h > y + (2.0f / 3.0f)){
                std::cout << '_';
            } else if (h > y + (1.0f / 3.0f)){
                std::cout << '=';
            } else {
                std::cout << '#';
            }
        }
        std::cout << '\n';
    }
}

template<typename F, typename T, std::size_t... Dimensions>
bool finite_difference_test(const scorch::Tensor<T, Dimensions...>& input_tensor, const F& scalar_function, T step_size = static_cast<T>(1e-3), T tolerance = static_cast<T>(1e-3)) {
    using TensorT = scorch::Tensor<T, Dimensions...>;
    static_assert(std::is_invocable_v<F, const TensorT&>);
    static_assert(scorch::detail::IsTensor<std::invoke_result_t<F, const TensorT&>>);
    static_assert(std::invoke_result_t<F, const TensorT&>::Scalar);

    auto fd_grad = scorch::TensorStorage<T, Dimensions...>{};
    auto fd_input = scorch::copy(input_tensor);
    for (auto i = 0; i < fd_input.NElements; ++i) {
        const auto x = input_tensor.get_flat(i);
        fd_input.value_mut().get_flat(i) = x - step_size;
        const auto v_0 = scalar_function(fd_input).item();
        fd_input.value_mut().get_flat(i) = x + step_size;
        const auto v_1 = scalar_function(fd_input).item();
        fd_grad.get_flat(i) = (v_1 - v_0) / (T{2} * step_size);
        fd_input.value_mut().get_flat(i) = x;
    }

    auto v = scalar_function(fd_input);
    v.backward();

    const auto& bp_grad = fd_input.grad();

    for (auto i = 0; i < fd_grad.NElements; ++i) {
        if (std::abs(fd_grad.get_flat(i) - bp_grad.get_flat(i)) > tolerance) {
            return false;
        }
    }
    return true;
}

template<typename T, std::size_t... Dimensions, typename F>
void test_with_random_inputs(const F& f, const std::string& description) {
    const auto num_trials = 100;
    auto num_success = 0;
    for (auto i = 0; i < 100; ++i) {
        auto x = scorch::rand<T, Dimensions...>(T{-1}, T{1});
        const auto correct = finite_difference_test(x, f);
        if (correct) {
            ++num_success;
        }
    }
    if (num_success < num_trials) {
        std::cout << "**Failed**: " << description << std::endl;
        std::cout << "  " << (
            100.0 * static_cast<double>(num_success) / static_cast<double>(num_trials)
        ) << "% success rate" << std::endl;
    } else {
        std::cout << "Passed: " << description << std::endl;
    }
}

template<typename T>
void test(){
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x; },
        "Identity"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return -x; },
        "Negation"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x + x; },
        "x + x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x - x; },
        "x - x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x * x; },
        "x * x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x / x; },
        "x / x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x + T{1}; },
        "x + 1"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x - T{1}; },
        "x - 1"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x * T{2}; },
        "x * 2"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x / T{2}; },
        "x / 2"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return x ^ T{2}; },
        "x ^ 2"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return T{1} + x; },
        "1 + x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return T{1} - x; },
        "1 - x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return T{2} * x; },
        "2 * x"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return scorch::sin(x); },
        "sin(x)"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return scorch::cos(x); },
        "cos(x)"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return scorch::exp(x); },
        "exp(x)"
    );
    test_with_random_inputs<T>(
        [](const scorch::Tensor<T>& x){ return scorch::sigmoid(x); },
        "sigmoid(x)"
    );
    test_with_random_inputs<T, 1>(
        [](const scorch::Tensor<T, 1>& x){ return scorch::sum(x); },
        "sum(x) 1D 1"
    );
    test_with_random_inputs<T, 2>(
        [](const scorch::Tensor<T, 2>& x){ return scorch::sum(x); },
        "sum(x) 1D 2"
    );
    test_with_random_inputs<T, 4>(
        [](const scorch::Tensor<T, 4>& x){ return scorch::sum(x); },
        "sum(x) 1D 4"
    );
    test_with_random_inputs<T, 8>(
        [](const scorch::Tensor<T, 8>& x){ return scorch::sum(x); },
        "sum(x) 1D 8"
    );
    test_with_random_inputs<T, 16>(
        [](const scorch::Tensor<T, 16>& x){ return scorch::sum(x); },
        "sum(x) 1D 16"
    );
    test_with_random_inputs<T, 32>(
        [](const scorch::Tensor<T, 32>& x){ return scorch::sum(x); },
        "sum(x) 1D 32"
    );
    test_with_random_inputs<T, 64>(
        [](const scorch::Tensor<T, 64>& x){ return scorch::sum(x); },
        "sum(x) 1D 64"
    );
    test_with_random_inputs<T, 128>(
        [](const scorch::Tensor<T, 128>& x){ return scorch::sum(x); },
        "sum(x) 1D 128"
    );
    test_with_random_inputs<T, 256>(
        [](const scorch::Tensor<T, 256>& x){ return scorch::sum(x); },
        "sum(x) 1D 256"
    );
    test_with_random_inputs<T, 8, 8>(
        [](const scorch::Tensor<T, 8, 8>& x){ return scorch::sum(x); },
        "sum(x) 2D"
    );
    test_with_random_inputs<T, 4, 4, 4>(
        [](const scorch::Tensor<T, 4, 4, 4>& x){ return scorch::sum(x); },
        "sum(x) 3D"
    );
    test_with_random_inputs<T, 64>(
        [](const scorch::Tensor<T, 64>& x){ return scorch::mean(x); },
        "mean(x) 1D"
    );
    test_with_random_inputs<T, 8, 8>(
        [](const scorch::Tensor<T, 8, 8>& x){ return scorch::mean(x); },
        "mean(x) 2D"
    );
    test_with_random_inputs<T, 4, 4, 4>(
        [](const scorch::Tensor<T, 4, 4, 4>& x){ return scorch::mean(x); },
        "mean(x) 3D"
    );
    auto A0 = scorch::random_matrix<T, 1, 16>();
    test_with_random_inputs<T, 16>(
        [&A0](const scorch::Tensor<T, 16>& x){ return scorch::sum(scorch::matvecmul(x, A0)); },
        "sum(matvecmul(x, A)) 1"
    );
    auto A1 = scorch::random_matrix<T, 16, 16>();
    test_with_random_inputs<T, 16>(
        [&A1](const scorch::Tensor<T, 16>& x){ return scorch::sum(scorch::matvecmul(x, A1)); },
        "sum(matvecmul(x, A)) 16"
    );
    auto X0 = scorch::rand<T, 1, 1>();
    test_with_random_inputs<T, 1>(
        [&X0](const scorch::Tensor<T, 1>& x){ return scorch::sum(x + X0); },
        "sum(x + X) wrt x 1x1"
    );
    auto X1 = scorch::rand<T, 2, 2>();
    test_with_random_inputs<T, 2>(
        [&X1](const scorch::Tensor<T, 2>& x){ return scorch::sum(x + X1); },
        "sum(x + X) wrt x 2x2"
    );
    auto X2 = scorch::rand<T, 4, 4>();
    test_with_random_inputs<T, 4>(
        [&X2](const scorch::Tensor<T, 4>& x){ return scorch::sum(x + X2); },
        "sum(x + X) wrt x 4x4"
    );
    auto X3 = scorch::rand<T, 8, 8>();
    test_with_random_inputs<T, 8>(
        [&X3](const scorch::Tensor<T, 8>& x){ return scorch::sum(x + X3); },
        "sum(x + X) wrt x 8x8"
    );
    auto X4 = scorch::rand<T, 16, 16>();
    test_with_random_inputs<T, 16>(
        [&X4](const scorch::Tensor<T, 16>& x){ return scorch::sum(x + X4); },
        "sum(x + X) wrt x 16x16"
    );
    auto x0 = scorch::rand<T, 16, 16>();
    test_with_random_inputs<T, 16, 16>(
        [&x0](const scorch::Tensor<T, 16, 16>& X){ return scorch::sum(X + x0); },
        "sum(x + X) wrt X"
    );
}

int main(int argc, char** argv) {
    if (argc == 2 && argv[1] == std::string{"test"}) {
        std::cout << "Performing finite-difference gradient tests in double and single precision" << std::endl;
        std::cout << "--- double ---" << std::endl;
        test<double>();
        std::cout << "--- float ---" << std::endl;
        test<float>();
        std::cout << "All done." << std::endl;
        return 0;
    }
    if (argc > 1) {
        std::cout << "Usage:\n  " << argv[0] << " [test]" << std::endl;
        return -1;
    }

    // Train a simple neural network with 4 inputs, 4 outputs,
    // and two hidden layers with 16 neurons each to learn
    // the identity function

    // layer sizes
    constexpr std::size_t InputDim = 1;
    constexpr std::size_t HiddenDim = 16;
    constexpr std::size_t OutputDim = 1;

    // learnable network parameters
    auto W0 = scorch::random_matrix<double, HiddenDim, InputDim>();
    auto b0 = scorch::zeros<double, HiddenDim>();
    auto W1 = scorch::random_matrix<double, HiddenDim, HiddenDim>();
    auto b1 = scorch::zeros<double, HiddenDim>();
    auto W2 = scorch::random_matrix<double, OutputDim, HiddenDim>();
    auto b2 = scorch::zeros<double, OutputDim>();
    // auto a = scorch::ones<double>();
    // auto b = scorch::ones<double>();
    // auto c = scorch::ones<double>();

    const auto network = [&](const auto& x) {
        return sin(sin(x % W0 + b0) % W1 + b1) % W2 + b2;
        // return a * sin(b * x) + c;
    };

    // optimizer
    // learning rate, momentum ratio, parameters...
    auto opt = scorch::optim::SGD(0.01, 0.5, W0, b0, W1, b1, W2, b2);
    // auto opt = scorch::optim::SGD(0.01, 0.0, a, b, c);

    // batch size
    constexpr std::size_t BatchDim = 32;

    constexpr auto num_iterations = 10000;
    constexpr auto plot_interval = 100;
    for (auto i = 0; i < num_iterations; ++i) {

        // random input
        auto x = scorch::rand<double, BatchDim, InputDim>(domain_min, domain_max);

        // target function
        // auto y = copy(x) * copy(x);
        // auto y = 0.5 - 0.5 * copy(x);
        auto y = sin(1.5 * 3.141592654 * copy(x));

        // compute the network output
        auto y_hat = network(x);

        static_assert(y.NDims == y_hat.NDims);

        // compute the loss
        auto l = mean((y_hat - y) ^ 2.0);

        // don't forget to zero the gradients before back-propagation
        opt.zero_grad();

        // compute the gradients of all parameters w.r.t. the loss
        l.backward();

        // take a training step
        opt.step();

        const auto time_to_print = (i == 0) || ((i + 1) == num_iterations) || ((i + 1) % plot_interval == 0);

        // print a line whose width is proportional to the log of the loss
        // const auto lmin = std::log(0.001);
        // const auto lmax = std::log(10.0);
        // assert(lmin < lmax);
        // const auto ll = std::clamp(std::log(l.item()), lmin, lmax);
        // const auto w = static_cast<std::size_t>(
        //     std::round(80.0f * (ll - lmin) / (lmax - lmin))
        // );
        // for (auto j = std::size_t{0}; j < w; ++j) {
        //     std::cout << '=';
        // }
        // std::cout << std::endl;

        if (time_to_print) {
            std::cout << "l = " << l << std::endl;
            // std::cout << "  x = " << x << std::endl;
            // std::cout << "  y = " << y << std::endl;

            print([&](float x) -> float {
                auto input = scorch::Tensor<double, 1>{};
                input.get_mut(0) = x;
                return static_cast<float>(network(input)(0));
                // return (0.5 + 0.5 * sin(3.0 * 3.141592654 * input))(0);
            });

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    }

    return 0;
}