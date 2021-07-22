#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <ios>
#include <memory>
#include <type_traits>
#include <vector>

#include <gsl/span>


namespace scorch {

    namespace detail {

        template<auto Dummy, typename T>
        struct MapNonTypeToTypeImpl {
            using Type = T;
        };

        template<auto Dummy, typename T>
        using MapNonTypeToType = typename MapNonTypeToTypeImpl<Dummy, T>::Type;

        template<std::size_t... Dimensions>
        struct IsScalarImpl {
            static constexpr bool Value = false;
        };

        template<>
        struct IsScalarImpl<1> {
            static constexpr bool Value = true;
        };

        template<std::size_t... Dimensions>
        constexpr bool IsScalar = IsScalarImpl<Dimensions...>::Value;

    } // namespace detail

    template<typename T, std::size_t... Dimensions>
    class TensorStorage {
    public:
        TensorStorage() noexcept
            : m_value{} {

        };

        T operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) const noexcept {
            return m_value[flatten_index({indices...})];
        }
        T& operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) noexcept {
            return m_value[flatten_index({indices...})];
        }

        T get_flat(std::size_t flat_index) const noexcept {
            assert(flat_index < NElements);
            return m_value[flat_index];
        }
        T& get_flat(std::size_t flat_index) noexcept {
            assert(flat_index < NElements);
            return m_value[flat_index];
        }

        static constexpr std::size_t NDims = sizeof...(Dimensions);
        static constexpr std::array<std::size_t, NDims> Dims = {{Dimensions...}};
        static constexpr std::size_t NElements = (Dimensions * ...);

        static constexpr std::size_t flatten_index(std::array<std::size_t, NDims> indices) noexcept {
            auto flat_index = std::size_t{0};
            for (auto iDim = std::size_t{0}; iDim < NDims; ++iDim) {
                assert(indices[iDim] < Dims[iDim]);
                flat_index = flat_index * Dims[iDim] + indices[iDim];
            }
            return flat_index;
        }

        static constexpr std::array<std::size_t, NDims> unflatten_index(std::size_t flat_index) noexcept {
            auto indices = std::array<std::size_t, NDims>{};
            for (auto i = 0; i < NDims; ++i) {
                const auto ir = NDims - i - 1;
                const auto d = Dims[ir];
                indices[ir] = flat_index % d;
                flat_index /= d;
            }
            return indices;
        }
    private:

        std::array<T, NElements> m_value;
    };

    template<typename T, std::size_t... Dimensions>
    class Tensor {
    public:
        using Storage = TensorStorage<T, Dimensions...>;
        using GradFunction = std::function<void(const Tensor<T, Dimensions...>&)>;

        Tensor(std::shared_ptr<Storage> storage, std::shared_ptr<GradFunction> grad_fn) noexcept
            : m_storage(std::move(storage))
            , m_grad(std::make_shared<Storage>())
            , m_grad_fn(std::move(grad_fn)) {
        }

        Tensor(std::enable_if_t<detail::IsScalar<Dimensions...>, T> scalar_value) noexcept
            : m_storage(std::make_shared<Storage>())
            , m_grad(std::make_shared<Storage>())
            , m_grad_fn(nullptr) {

            (*m_storage)(0) = scalar_value;
        }

        // TODO:
        // - only allow if Dimensions... == 1
        // - assert that m_grad is empty
        // - call backward_impl
        std::enable_if_t<detail::IsScalar<Dimensions...>, void>
        backward() {
            (*m_grad)(0) = T{1};
            backward_impl();
        }

        T operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) const noexcept {
            assert(m_storage);
            return (*m_storage)(indices...);
        }

        T get_flat(std::size_t flat_index) const noexcept {
            assert(m_storage);
            return m_storage->get_flat(flat_index);
        }

        const Storage& grad() const noexcept {
            assert(m_grad);
            return *m_grad;
        }

        bool requires_grad() const noexcept {
            return static_cast<bool>(m_grad_fn);
        }

        static constexpr std::size_t NDims = sizeof...(Dimensions);
        static constexpr std::array<std::size_t, NDims> Dims = {{Dimensions...}};
        static constexpr std::size_t NElements = (Dimensions * ...);

    // private:

        void backward_impl() {
            if (m_grad_fn == nullptr) {
                return;
            }
            const auto& fn = *m_grad_fn;
            assert(fn != nullptr);
            fn(*this);
        }

        std::shared_ptr<Storage> m_storage;
        std::shared_ptr<Storage> m_grad;
        std::shared_ptr<GradFunction> m_grad_fn;
    };

    template<typename T, std::size_t... Dimensions>
    std::ostream& operator<<(std::ostream& o, const TensorStorage<T, Dimensions...>& t) noexcept {
        auto fmt = std::ios{nullptr};
        fmt.copyfmt(o);
        auto flags = std::ios_base::fmtflags{o.flags()};
        o << std::setprecision(4) << std::fixed;
        std::cout << std::setprecision(4) << std::fixed;
        const auto numColumns = t.Dims.back();
        const auto numRows = t.NElements / numColumns;
        auto prev_indices = t.unflatten_index(0);
        std::fill(
            prev_indices.begin(),
            prev_indices.end(),
            static_cast<std::size_t>(-1)
        );
        if (t.NDims > 1) {
            o << "[\n";
        }
        for (auto row = std::size_t{0}; row < numRows; ++row) {
            const auto indices = t.unflatten_index(row * numColumns);
            if (t.NDims > 1) {
                o << ' ';
            }
            for (auto i = std::size_t{0}; i + 2 < t.NDims; ++i) {
                if (indices[i] != prev_indices[i]) {
                    o << "[\n";
                    if (t.NDims > 1) {
                        o << ' ';
                    }
                    for (auto j = std::size_t{0}; j < i + 1; ++j) {
                        o << ' ';
                    }
                } else {
                    o << ' ';
                }
            }

            o << '[';

            for (auto col = std::size_t{0}; col < numColumns; ++col) {
                const auto& v = t.get_flat(row * numColumns + col);
                if (v >= T{0}) {
                    o << ' ';
                }
                o << v;
                if (col + 1 < numColumns) {
                    o << ", ";
                }
            }

            o << ']';

            const auto next_indices = t.unflatten_index((row + 1) * numColumns);

            if (t.NDims >= 2) {
                if (indices[t.NDims - 2] < next_indices[t.NDims - 2]) {
                    o << ',';
                }
            }

            for (auto i = std::size_t{0}; i + 2 < t.NDims; ++i) {
                const auto ir = t.NDims - i - 3;
                if (indices[ir] != next_indices[ir]) {
                    o << '\n';
                    for (auto j = std::size_t{0}; j < ir + 1; ++j) {
                        o << ' ';
                    }
                    o << ']';
                    if (indices[ir] + 1 != t.Dims[ir]) {
                        o << ',';
                    }
                }
            }

            o << '\n';

            prev_indices = indices;
        }
        if (t.NDims > 1) {
            o << ']';
        }
        o.flags(flags);
        o.copyfmt(fmt);
        return o;
    }

    template<typename T, std::size_t... Dimensions>
    std::ostream& operator<<(std::ostream& o, const Tensor<T, Dimensions...>& t) noexcept {
        assert(t.m_storage);
        o << (*t.m_storage);
        return o;
    }

    //------------------------------------------

    namespace detail {

        template<typename F, typename G, typename T, std::size_t... Dimensions>
        Tensor<T, Dimensions...> elementwise_unary_function(F&& f, G&& g, const Tensor<T, Dimensions...>& x) {
            // F : T => T
            static_assert(std::is_invocable_v<F, T>);
            static_assert(std::is_same_v<std::invoke_result_t<F, T>, T>);

            // G : T => T
            static_assert(std::is_invocable_v<G, T>);
            static_assert(std::is_same_v<std::invoke_result_t<G, T>, T>);

            using TensorT = Tensor<T, Dimensions...>;
            using Storage = TensorStorage<T, Dimensions...>;
            using GradFn = std::function<void(const TensorT&)>;

            auto ptr_output = std::make_shared<Storage>();
            for (auto i = std::size_t{0}; i < ptr_output->NElements; ++i) {
                ptr_output->get_flat(i) = f(x.get_flat(i));
            }

            auto grad_fn = [
                c_x = x,
                c_g = std::forward<G>(g)
            ](const TensorT& t) mutable {
                assert(t.m_grad);
                assert(c_x.m_grad);

                for (auto i = std::size_t{0}; i < t.NElements; ++i) {
                    const auto& x_i = c_x.get_flat(i);
                    const auto& g_i = t.m_grad->get_flat(i);
                    c_x.m_grad->get_flat(i) += c_g(x_i) * g_i;
                }

                c_x.backward_impl();
            };

            auto ptr_grad_fn = std::make_shared<GradFn>(std::move(grad_fn));

            return Tensor{std::move(ptr_output), std::move(ptr_grad_fn)};
        };

        template<typename F, typename G0, typename G1, typename T, std::size_t... Dimensions>
        Tensor<T, Dimensions...> elementwise_binary_function(F&& f, G0&& g0, G1&& g1, const Tensor<T, Dimensions...>& x0, const Tensor<T, Dimensions...>& x1) {
            // F : T, T => T
            static_assert(std::is_invocable_v<F, T, T>);
            static_assert(std::is_same_v<std::invoke_result_t<F, T, T>, T>);

            // G0 : T, T => T
            static_assert(std::is_invocable_v<G0, T, T>);
            static_assert(std::is_same_v<std::invoke_result_t<G0, T, T>, T>);

            // G1 : T, T => T
            static_assert(std::is_invocable_v<G1, T, T>);
            static_assert(std::is_same_v<std::invoke_result_t<G1, T, T>, T>);

            using TensorT = Tensor<T, Dimensions...>;
            using Storage = TensorStorage<T, Dimensions...>;
            using GradFn = std::function<void(const TensorT&)>;

            auto ptr_output = std::make_shared<Storage>();
            for (auto i = std::size_t{0}; i < ptr_output->NElements; ++i) {
                ptr_output->get_flat(i) = f(x0.get_flat(i), x1.get_flat(i));
            }

            auto grad_fn = [
                c_x0 = x0,
                c_x1 = x1,
                c_g0 = std::forward<G0>(g0),
                c_g1 = std::forward<G1>(g1)
            ](const TensorT& t) mutable {
                assert(t.m_grad);
                assert(c_x0.m_grad);
                assert(c_x1.m_grad);

                for (auto i = std::size_t{0}; i < t.NElements; ++i) {
                    const auto& x0_i = c_x0.get_flat(i);
                    const auto& x1_i = c_x1.get_flat(i);
                    const auto& g_i = t.m_grad->get_flat(i);
                    c_x0.m_grad->get_flat(i) += c_g0(x0_i, x1_i) * g_i;
                    c_x1.m_grad->get_flat(i) += c_g1(x0_i, x1_i) * g_i;
                }

                c_x0.backward_impl();
                c_x1.backward_impl();
            };

            auto ptr_grad_fn = std::make_shared<GradFn>(std::move(grad_fn));

            return Tensor{std::move(ptr_output), std::move(ptr_grad_fn)};
        };

    } // namespace detail

    // +x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(const Tensor<T, Dimensions...>& x) noexcept {
        return elementwise_unary_function(
            // f
            [](T t){ return t; },
            // df/dt
            [](T /* t */) { return T{1}; },
            x
        );
    }

    // -x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator-(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return -t; },
            // df/dt
            [](T /* t */) { return T{-1}; },
            x
        );
    }

    // 1 + x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(T l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_l = l](T t){ return c_l + t; },
            // df/dt
            [](T /* t */) { return T{1}; },
            r
        );
    }

    // x + 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(const Tensor<T, Dimensions...>& l, T r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_r = r](T t){ return t + c_r; },
            // df/dt
            [](T /* t */) { return T{1}; },
            l
        );
    }

    // 1 - x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator-(T l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_l = l](T t){ return c_l - t; },
            // df/dt
            [](T /* t */) { return T{-1}; },
            r
        );
    }

    // x - 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator-(const Tensor<T, Dimensions...>& l, T r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_r = r](T t){ return t - c_r; },
            // df/dt
            [](T /* t */) { return T{1}; },
            r
        );
    }

    // 1 * x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator*(T l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_l = l](T t){ return c_l * t; },
            // df/dt
            [c_l = l](T /* t */) { return c_l; },
            r
        );
    }

    // x * 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator*(const Tensor<T, Dimensions...>& l, T r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_r = r](T t){ return t * c_r; },
            // df/dt
            [c_r = r](T /* t */) { return c_r; },
            l
        );
    }

    // x / 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator/(const Tensor<T, Dimensions...>& l, T r) noexcept {
        return detail::elementwise_unary_function(
            // f
            [c_r = r](T t){ return t / c_r; },
            // df/dt
            [c_r_inv = (T{1} / r)](T /* t */) { return c_r_inv; },
            l
        );
    }

    // abs(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> abs(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return std::abs(t); },
            // df/dt
            [](T t) {
                if (t >= T{0}) {
                    return T{1};
                } else {
                    return T{-1};
                }
            },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> exp(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return std::exp(t); },
            // df/dt
            [](T t) { return std::exp(t); },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> log(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return std::log(t); },
            // df/dt
            [](T t) { return T{1} / t; },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> square(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return t * t; },
            // df/dt
            [](T t) { return T{2} * t; },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> sqrt(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return std::sqrt(t); },
            // df/dt
            [](T t) { return static_cast<T>(0.5) / std::sqrt(t); },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> sin(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return std::sin(t); },
            // df/dt
            [](T t) { return std::cos(t); },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> cos(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_unary_function(
            // f
            [](T t){ return std::cos(t); },
            // df/dt
            [](T t) { return std::-sin(x); },
            x
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_binary_function(
            // f
            [](T t0, T t1) { return t0 + t1; },
            // df/dt0
            [](T /* t0 */, T /* t1 */){ return T{1}; },
            // df/dt1
            [](T /* t0 */, T /* t1 */){ return T{1}; },
            l,
            r
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator-(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_binary_function(
            // f
            [](T t0, T t1) { return t0 - t1; },
            // df/dt0
            [](T /* t0 */, T /* t1 */){ return T{1}; },
            // df/dt1
            [](T /* t0 */, T /* t1 */){ return T{-1}; },
            l,
            r
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator*(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_binary_function(
            // f
            [](T t0, T t1) { return t0 * t1; },
            // df/dt0
            [](T /* t0 */, T t1){ return t1; },
            // df/dt1
            [](T t0, T /* t1 */){ return t0; },
            l,
            r
        );
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator/(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_binary_function(
            // f
            [](T t0, T t1) { return t0 / t1; },
            // df/dt0
            [](T /* t0 */, T t1){ return T{1} / t1; },
            // df/dt1
            [](T t0, T t1){ return -t1 / (t0 * t0); },
            l,
            r
        );
    }

    // y = a / b = a * b^-1
    // dy/da = 1/b
    // dy/db = -b / a^2

} // namespace scorch
