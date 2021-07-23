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
#include <random>
#include <type_traits>
#include <vector>

namespace scorch {

    namespace detail {

        std::random_device randDev;
        std::default_random_engine randEng{randDev()};

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
        struct IsScalarImpl<> {
            static constexpr bool Value = true;
        };

        template<std::size_t... Dimensions>
        constexpr bool IsScalar = IsScalarImpl<Dimensions...>::Value;

        template<std::size_t... Dimensions>
        struct NumElementsImpl {
            static constexpr std::size_t Value = (Dimensions * ...);
        };

        template<>
        struct NumElementsImpl<> {
            static constexpr std::size_t Value = 1;
        };

        template<std::size_t... Dimensions>
        constexpr std::size_t NumElements = NumElementsImpl<Dimensions...>::Value;

    } // namespace detail

    template<typename T, std::size_t... Dimensions>
    class TensorStorage {
    public:
        using ValueType = T;
        static constexpr bool Scalar = detail::IsScalar<Dimensions...>;

        TensorStorage() noexcept
            : m_value{} {

        };

        T operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) const noexcept {
            return m_value[flatten_index({indices...})];
        }
        T& operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) noexcept {
            return m_value[flatten_index({indices...})];
        }

        T item() const noexcept {
            assert(Scalar);
            return m_value[0];
        }

        T& item() noexcept {
            assert(Scalar);
            return m_value[0];
        }

        T get_flat(std::size_t flat_index) const noexcept {
            assert(flat_index < NElements);
            return m_value[flat_index];
        }
        T& get_flat(std::size_t flat_index) noexcept {
            assert(flat_index < NElements);
            return m_value[flat_index];
        }

        void fill(T fill_value) noexcept {
            for (auto& v : m_value) {
                v = fill_value;
            }
        }

        void zero() noexcept {
            for (auto& v : m_value) {
                v = T{0};
            }
        }

        void fma(const TensorStorage& x, T y) noexcept {
            for (auto i = std::size_t{0}; i < NElements; ++i) {
                get_flat(i) += x.get_flat(i) * y;
            }
        }

        void fma(T x, const TensorStorage& y) noexcept {
            for (auto i = std::size_t{0}; i < NElements; ++i) {
                get_flat(i) += x * y.get_flat(i);
            }
        }

        void fma(const TensorStorage& x, const TensorStorage& y) noexcept {
            for (auto i = std::size_t{0}; i < NElements; ++i) {
                get_flat(i) += x.get_flat(i) + y.get_flat(i);
            }
        }

        static constexpr std::size_t NDims = sizeof...(Dimensions);
        static constexpr std::array<std::size_t, NDims> Dims = {{Dimensions...}};
        static constexpr std::size_t NElements = detail::NumElements<Dimensions...>;

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
    TensorStorage<T, Dimensions...>& operator+=(TensorStorage<T, Dimensions...>& l, T r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) += r;
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator-=(TensorStorage<T, Dimensions...>& l, T r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) -= r;
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator*=(TensorStorage<T, Dimensions...>& l, T r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) *= r;
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator/=(TensorStorage<T, Dimensions...>& l, T r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) /= r;
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator+=(TensorStorage<T, Dimensions...>& l, const TensorStorage<T, Dimensions...>& r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) += r.get_flat(i);
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator-=(TensorStorage<T, Dimensions...>& l, const TensorStorage<T, Dimensions...>& r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) -= r.get_flat(i);
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator*=(TensorStorage<T, Dimensions...>& l, const TensorStorage<T, Dimensions...>& r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) *= r.get_flat(i);
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    TensorStorage<T, Dimensions...>& operator/=(TensorStorage<T, Dimensions...>& l, const TensorStorage<T, Dimensions...>& r) noexcept {
        for (auto i = std::size_t{0}; i < l.NElements; ++i) {
            l.get_flat(i) /= r.get_flat(i);
        }
        return l;
    }

    template<typename T, std::size_t... Dimensions>
    class Tensor {
    public:
        using ValueType = T;
        static constexpr bool Scalar = detail::IsScalar<Dimensions...>;

        using Storage = TensorStorage<T, Dimensions...>;
        using GradFunction = std::function<void(const Tensor<T, Dimensions...>&)>;

        Tensor(std::shared_ptr<Storage> storage, std::shared_ptr<GradFunction> grad_fn) noexcept
            : m_storage(std::move(storage))
            , m_grad(std::make_shared<Storage>())
            , m_grad_fn(std::move(grad_fn)) {
        }

        Tensor() noexcept
            : m_storage(std::make_shared<Storage>())
            , m_grad(std::make_shared<Storage>())
            , m_grad_fn(nullptr) {

        }


        Tensor(T scalar_value) noexcept
            : m_storage(std::make_shared<Storage>())
            , m_grad(std::make_shared<Storage>())
            , m_grad_fn(nullptr) {

            static_assert(Scalar);
            m_storage->item() = scalar_value;
        }

        void backward() {
            static_assert(Scalar);
            m_grad->item() = T{1};
            backward_impl();
        }

        T operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) const noexcept {
            assert(m_storage);
            return (*m_storage)(indices...);
        }

        T& operator()(detail::MapNonTypeToType<Dimensions, std::size_t>... indices) noexcept {
            assert(m_storage);
            assert(!requires_grad());
            return (*m_storage)(indices...);
        }

        T get_flat(std::size_t flat_index) const noexcept {
            assert(m_storage);
            return m_storage->get_flat(flat_index);
        }

        T item() const noexcept {
            static_assert(Scalar);
            return m_storage->item();
        }

        Storage& value_mut() noexcept {
            assert(m_storage);
            return *m_storage;
        }

        const Storage& value() const noexcept {
            assert(m_storage);
            return *m_storage;
        }

        const Storage& grad() const noexcept {
            assert(m_grad);
            return *m_grad;
        }

        Storage& grad_mut() noexcept {
            assert(m_grad);
            return *m_grad;
        }

        bool requires_grad() const noexcept {
            return static_cast<bool>(m_grad_fn);
        }

        static constexpr std::size_t NDims = Storage::NDims;
        static constexpr std::array<std::size_t, NDims> Dims = Storage::Dims;
        static constexpr std::size_t NElements = Storage::NElements;

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
    Tensor<T, Dimensions...> copy(Tensor<T, Dimensions...>& src) noexcept {
        auto t = Tensor<T, Dimensions...>{};
        for (auto i = std::size_t{0}; i < t.NElements; ++i) {
            t.value_mut().get_flat(i) = src.get_flat(i);
        }
        return t;
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> rand(T low_value = T{0}, T high_value = T{1}) noexcept {
        auto t = Tensor<T, Dimensions...>{};
        auto d = std::uniform_real_distribution<T>(low_value, high_value);
        for (auto i = std::size_t{0}; i < t.NElements; ++i) {
            t.value_mut().get_flat(i) = d(detail::randEng);
        }
        return t;
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> rand_like(const Tensor<T, Dimensions...>& /* unused */, T low_value = T{0}, T high_value = T{1}) noexcept {
        auto t = Tensor<T, Dimensions...>{};
        auto d = std::uniform_real_distribution<T>(low_value, high_value);
        for (auto i = std::size_t{0}; i < t.NElements; ++i) {
            t.value_mut().get_flat(i) = d(detail::randEng);
        }
        return t;
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> zeros() noexcept {
        return Tensor<T, Dimensions...>{};
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> zeros_like(const Tensor<T, Dimensions...>& /* unused */) noexcept {
        return Tensor<T, Dimensions...>{};
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> ones() noexcept {
        auto t = Tensor<T, Dimensions...>{};
        t.value_mut().fill(T{1});
        return t;
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> ones_like(const Tensor<T, Dimensions...>& /* unused */) noexcept {
        auto t = Tensor<T, Dimensions...>{};
        t.value_mut().fill(T{1});
        return t;
    }

    template<typename T, std::size_t... Dimensions>
    std::ostream& operator<<(std::ostream& o, const TensorStorage<T, Dimensions...>& t) noexcept {
        if constexpr (t.Scalar) {
            o << t.item();
            return o;
        }
        auto fmt = std::ios{nullptr};
        fmt.copyfmt(o);
        auto flags = std::ios_base::fmtflags{o.flags()};
        o << std::setprecision(4) << std::fixed;
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

            if (row + 1 < numRows) {
                o << '\n';
            }

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

    namespace detail {

        template<typename T>
        struct IsTensorImpl {
            static constexpr bool Value = false;
        };

        template<typename T, std::size_t... Dimensions>
        struct IsTensorImpl<Tensor<T, Dimensions...>> {
            static constexpr bool Value = true;
        };

        template<typename T>
        constexpr bool IsTensor = IsTensorImpl<T>::Value;

    } // namespace detail

    template<typename... Parameters>
    class Optimizer {
    public:
        static_assert((detail::IsTensor<Parameters> && ...));

        Optimizer(const Parameters&... parameters)
            : m_parameters(parameters...) {

        }

        virtual void zero_grad() = 0;

        virtual void step() = 0;

        std::tuple<Parameters...>& parameters() noexcept {
            return m_parameters;
        }

        template<typename F>
        void visit_parameters(F&& f) {
            std::apply(
                [c_f = std::forward<F>(f)](auto&... args){
                    (c_f(args), ...);
                },
                m_parameters
            );
        }

    private:
        std::tuple<Parameters...> m_parameters;
    };

    template<typename T, typename... Parameters>
    class SGD : public Optimizer<Parameters...> {
    public:
        SGD(T step_size, T momentum, const Parameters&... parameters)
            : m_step_size(step_size)
            , m_momentum(momentum)
            , Optimizer(parameters...) {

        }

        void zero_grad() override {
            this->visit_parameters([this](auto& param){
                param.grad_mut() *= this->m_momentum;
            });
        }

        void step() override {
            this->visit_parameters([this](auto& param){
                param.value_mut().fma(-this->m_step_size, param.grad());
            });
        }

    private:
        T m_step_size;
        T m_momentum;
    };



    //------------------------------------------

    namespace detail {

        template<typename SequenceA, typename SequenceB>
        struct SameEndingImpl {
            static constexpr bool Value = false;
        };

        template<std::size_t S>
        struct SameEndingImpl<std::index_sequence<S>, std::index_sequence<S>> {
            static constexpr bool Value = true;
        };

        template<std::size_t S, std::size_t... A, std::size_t... B>
        struct SameEndingImpl<std::index_sequence<S, A...>, std::index_sequence<S, B...>> {
            static constexpr bool Value = SameEndingImpl<
                std::index_sequence<A...>,
                std::index_sequence<B...>,
            >::Value;
        };

        template<typename IndexSequenceA, typename IndexSequenceB>
        constexpr bool SameEnding = SameEndingImpl<IndexSequenceA, IndexSequenceB>::Value;

        template<typename F, typename G, typename T, std::size_t... Dimensions>
        Tensor<T, Dimensions...> elementwise_tensor(F&& f, G&& g, const Tensor<T, Dimensions...>& x) {
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
        Tensor<T, Dimensions...> elementwise_tensor_tensor(F&& f, G0&& g0, G1&& g1, const Tensor<T, Dimensions...>& x0, const Tensor<T, Dimensions...>& x1) {
            // TODO:
            // - take dimensions for x0 and x1 separately,
            // - assert that they have the same ending
            // - loop over any outer dimensions, like matvecmul

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

        template<typename F, typename GT, typename GS, typename T, std::size_t... Dimensions>
        Tensor<T, Dimensions...> elementwise_scalar_tensor(F&& f, GT&& gradient_tensor, GS&& gradient_scalar, const Tensor<T>& scalar, const Tensor<T, Dimensions...>& tensor) {
            using TensorT = Tensor<T, Dimensions...>;
            using Storage = TensorStorage<T, Dimensions...>;
            using GradFn = std::function<void(const TensorT&)>;

            // F : T, T => T
            static_assert(std::is_invocable_v<F, T, T>);
            static_assert(std::is_same_v<std::invoke_result_t<F, T, T>, T>);

            // GT : T, T => T
            static_assert(std::is_invocable_v<GT, T, T>);
            static_assert(std::is_same_v<std::invoke_result_t<GT, T, T>, T>);

            // GS : T, T => T
            static_assert(std::is_invocable_v<GS, T, T>);
            static_assert(std::is_same_v<std::invoke_result_t<GS, T, T>, T>);

            auto ptr_output = std::make_shared<Storage>();
            for (auto i = std::size_t{0}; i < ptr_output->NElements; ++i) {
                ptr_output->get_flat(i) = f(scalar.item(), tensor.get_flat(i));
            }

            auto grad_fn = [
                c_scalar = scalar,
                c_tensor = tensor,
                c_gradient_tensor = std::forward<GT>(gradient_tensor),
                c_gradient_scalar = std::forward<GS>(gradient_scalar)
            ](const TensorT& t) mutable {
                assert(t.m_grad);
                assert(c_scalar.m_grad);
                assert(c_tensor.m_grad);

                for (auto i = std::size_t{0}; i < t.NElements; ++i) {
                    const auto s = c_scalar.item();
                    const auto& t_i = c_tensor.get_flat(i);
                    const auto& g_i = t.m_grad->get_flat(i);
                    c_tensor.m_grad->get_flat(i) += c_gradient_tensor(s, t_i) * g_i;
                    c_scalar.m_grad->item() += c_gradient_scalar(s, t_i) * g_i;
                }

                c_scalar.backward_impl();
                c_tensor.backward_impl();
            };

            auto ptr_grad_fn = std::make_shared<GradFn>(std::move(grad_fn));

            return Tensor{std::move(ptr_output), std::move(ptr_grad_fn)};
        }

    } // namespace detail

    // UNARY OPERATORS

    // +x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(const Tensor<T, Dimensions...>& x) noexcept {
        return elementwise_tensor(
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
        return detail::elementwise_tensor(
            // f
            [](T t){ return -t; },
            // df/dt
            [](T /* t */) { return T{-1}; },
            x
        );
    }

    // BINARY OPERATORS WITH CONSTANTS

    // 1 + x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(T l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_tensor(
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
        return detail::elementwise_tensor(
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
        return detail::elementwise_tensor(
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
        return detail::elementwise_tensor(
            // f
            [c_r = r](T t){ return t - c_r; },
            // df/dt
            [](T /* t */) { return T{1}; },
            l
        );
    }

    // 1 * x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator*(T l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_tensor(
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
        return detail::elementwise_tensor(
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
        return detail::elementwise_tensor(
            // f
            [c_r = r](T t){ return t / c_r; },
            // df/dt
            [c_r_inv = (T{1} / r)](T /* t */) { return c_r_inv; },
            l
        );
    }

    // x ^ 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator^(const Tensor<T, Dimensions...>& l, T r) noexcept {
        return detail::elementwise_tensor(
            // f
            [c_r = r](T t){ return std::pow(t, c_r); },
            // df/dt
            [c_r = r](T t) { return c_r * std::pow(t, c_r - T{1}); },
            l
        );
    }

    // BINARY OPERATORS WITH SCALARS

    // 1 + x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator+(const Tensor<T>& scalar, const Tensor<T, Dimensions...>& tensor) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return s + t; },
            // df/dt
            [](T /* s */, T /* t */){ return T{1}; },
            // df/ds
            [](T /* s */, T /* t */){ return T{1}; },
            scalar,
            tensor
        );
    }

    // x + 1
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 1),Tensor<T, Dimensions...>>
    operator+(const Tensor<T, Dimensions...>& tensor, const Tensor<T>& scalar) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return s + t; },
            // df/dt
            [](T /* s */, T /* t */){ return T{1}; },
            // df/ds
            [](T /* s */, T /* t */){ return T{1}; },
            scalar,
            tensor
        );
    }

    // 1 - x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator-(const Tensor<T>& scalar, const Tensor<T, Dimensions...>& tensor) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return s - t; },
            // df/dt
            [](T /* s */, T /* t */){ return T{-1}; },
            // df/ds
            [](T /* s */, T /* t */){ return T{1}; },
            scalar,
            tensor
        );
    }

    // x - 1
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 1),Tensor<T, Dimensions...>>
    operator-(const Tensor<T, Dimensions...>& tensor, const Tensor<T>& scalar) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return t - s; },
            // df/dt
            [](T /* s */, T /* t */){ return T{1}; },
            // df/ds
            [](T /* s */, T /* t */){ return T{-1}; },
            scalar,
            tensor
        );
    }

    // 1 * x
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator*(const Tensor<T>& scalar, const Tensor<T, Dimensions...>& tensor) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return s * t; },
            // df/dt
            [](T s, T /* t */){ return s; },
            // df/ds
            [](T /* s */, T t){ return t; },
            scalar,
            tensor
        );
    }

    // x * 1
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 1),Tensor<T, Dimensions...>>
    operator*(const Tensor<T, Dimensions...>& tensor, const Tensor<T>& scalar) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return s * t; },
            // df/dt
            [](T s, T /* t */){ return s; },
            // df/ds
            [](T /* s */, T t){ return t; },
            scalar,
            tensor
        );
    }

    // x / 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator/(const Tensor<T, Dimensions...>& tensor, const Tensor<T>& scalar) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return t / s; },
            // df/dt
            [](T s, T /* t */){ return T{1} / s; },
            // df/ds
            [](T s, T t){ return -t / (s * s); },
            scalar,
            tensor
        );
    }

    // x ^ 1
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> operator^(const Tensor<T, Dimensions...>& tensor, const Tensor<T>& scalar) noexcept {
        return detail::elementwise_scalar_tensor(
            // f
            [](T s, T t){ return std::pow(t, s); },
            // df/dt
            [](T s, T t){ return s * std::pow(t, s - T{1}); },
            // df/ds
            [](T s, T t){ return std::pow(t, s) * std::log(t); },
            scalar,
            tensor
        );
    }

    // ELEMENTWISE UNARY FUNCTIONS

    // abs(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> abs(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
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

    // exp(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> exp(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return std::exp(t); },
            // df/dt
            [](T t) { return std::exp(t); },
            x
        );
    }

    // log(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> log(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return std::log(t); },
            // df/dt
            [](T t) { return T{1} / t; },
            x
        );
    }

    // square(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> square(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return t * t; },
            // df/dt
            [](T t) { return T{2} * t; },
            x
        );
    }

    // sqrt(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> sqrt(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return std::sqrt(t); },
            // df/dt
            [](T t) { return static_cast<T>(0.5) / std::sqrt(t); },
            x
        );
    }

    // sin(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> sin(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return std::sin(t); },
            // df/dt
            [](T t) { return std::cos(t); },
            x
        );
    }

    // cos(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> cos(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return std::cos(t); },
            // df/dt
            [](T t) { return -std::sin(t); },
            x
        );
    }

    // cos(x)
    template<typename T, std::size_t... Dimensions>
    Tensor<T, Dimensions...> sigmoid(const Tensor<T, Dimensions...>& x) noexcept {
        return detail::elementwise_tensor(
            // f
            [](T t){ return T{1} / (T{1} + std::exp(-t)); },
            // df/dt
            [](T t) {
                const auto s = T{1} / (T{1} + std::exp(-t));
                return s * (T{1} - s);
            },
            x
        );
    }

    // ELEMENTWISE BINARY OPERATIONS WITH TWO TENSORS

    // x + y
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 0), Tensor<T, Dimensions...>>
    operator+(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_tensor_tensor(
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

    // x - y
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 0), Tensor<T, Dimensions...>>
    operator-(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_tensor_tensor(
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

    // x * y
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 0), Tensor<T, Dimensions...>>
    operator*(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_tensor_tensor(
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

    // x / y
    template<typename T, std::size_t... Dimensions>
    std::enable_if_t<(sizeof...(Dimensions) > 0), Tensor<T, Dimensions...>>
    operator/(const Tensor<T, Dimensions...>& l, const Tensor<T, Dimensions...>& r) noexcept {
        return detail::elementwise_tensor_tensor(
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

    template<typename T, std::size_t... Dimensions>
    Tensor<T> sum(const Tensor<T, Dimensions...> x) noexcept {
        using OutputTensorT = Tensor<T>;
        using OutputStorage = TensorStorage<T>;
        using GradFn = std::function<void(const OutputTensorT&)>;

        auto ptr_output = std::make_shared<OutputStorage>();
        for (auto i = std::size_t{0}; i < x.NElements; ++i) {
            ptr_output->item() += x.get_flat(i);
        }

        auto grad_fn = [
            c_x = x
        ](const OutputTensorT& t) mutable {
            assert(t.m_grad);
            assert(c_x.m_grad);

            const auto& g = t.m_grad->item();
            for (auto i = std::size_t{0}; i < c_x.NElements; ++i) {
                const auto& x_i = c_x.get_flat(i);
                c_x.m_grad->get_flat(i) += g;
            }

            c_x.backward_impl();
        };

        auto ptr_grad_fn = std::make_shared<GradFn>(std::move(grad_fn));

        return OutputTensorT{std::move(ptr_output), std::move(ptr_grad_fn)};
    }

    template<typename T, std::size_t... Dimensions>
    Tensor<T> mean(const Tensor<T, Dimensions...> x) noexcept {
        using OutputTensorT = Tensor<T>;
        using OutputStorage = TensorStorage<T>;
        using GradFn = std::function<void(const OutputTensorT&)>;

        const auto k = T{1} / static_cast<T>(x.NElements);

        auto ptr_output = std::make_shared<OutputStorage>();
        for (auto i = std::size_t{0}; i < x.NElements; ++i) {
            ptr_output->item() += x.get_flat(i);
        }

        ptr_output->item() *= k;

        auto grad_fn = [
            c_x = x
        ](const OutputTensorT& t) mutable {
            assert(t.m_grad);
            assert(c_x.m_grad);
            const auto kk = T{1} / static_cast<T>(c_x.NElements);

            const auto& g = t.m_grad->item();
            for (auto i = std::size_t{0}; i < c_x.NElements; ++i) {
                const auto& x_i = c_x.get_flat(i);
                c_x.m_grad->get_flat(i) += g * kk;
            }

            c_x.backward_impl();
        };

        auto ptr_grad_fn = std::make_shared<GradFn>(std::move(grad_fn));

        return OutputTensorT{std::move(ptr_output), std::move(ptr_grad_fn)};
    }

    namespace detail {

        // WANTS:
        // - new last dimension plus list of all dimensions to substituted last dimension

        template<std::size_t... Dimensions>
        struct LastDimensionImpl;

        template<std::size_t D0, std::size_t D1, std::size_t... Rest>
        struct LastDimensionImpl<D0, D1, Rest...> {
            static constexpr std::size_t Value = LastDimensionImpl<D1, Rest...>::Value;
        };

        template<std::size_t D>
        struct LastDimensionImpl<D> {
            static constexpr std::size_t Value = D;
        };

        template<std::size_t... Dimensions>
        constexpr std::size_t LastDimension = LastDimensionImpl<Dimensions...>::Value;

        template<std::size_t NewLastDimension, typename InputDimensionsSequence, typename OutputDimensionsSequence>
        struct ReplaceLastDimensionImpl;

        template<
            std::size_t NewLastDimension,
            std::size_t InputDimension0,
            std::size_t InputDimension1,
            std::size_t... OtherInputDimensions,
            std::size_t... OutputDimensions
        >
        struct ReplaceLastDimensionImpl<
            NewLastDimension,
            std::index_sequence<
                InputDimension0,
                InputDimension1,
                OtherInputDimensions...
            >,
            std::index_sequence<
                OutputDimensions...
            >
        > {
            using Type = typename ReplaceLastDimensionImpl<
                NewLastDimension,
                std::index_sequence<
                    InputDimension1,
                    OtherInputDimensions...
                >,
                std::index_sequence<
                    InputDimension0,
                    OutputDimensions...
                >
            >::Type;
        };

        template<
            std::size_t NewLastDimension,
            std::size_t InputDimension,
            std::size_t... OutputDimensions
        >
        struct ReplaceLastDimensionImpl<
            NewLastDimension,
            std::index_sequence<InputDimension>,
            std::index_sequence<OutputDimensions...>
        > {
            using Type = std::index_sequence<OutputDimensions..., InputDimension>;
        };

        template<std::size_t NewLastDimension, std::size_t... Dimensions>
        using ReplaceLastDimension = typename ReplaceLastDimensionImpl<
            NewLastDimension,
            std::index_sequence<Dimensions...>,
            std::index_sequence<>
        >::Type;

        template<typename T, typename NewDimensionsSequence>
        struct ReplaceLastTensorDimensionImpl;

        template<typename T, std::size_t... NewDimensions>
        struct ReplaceLastTensorDimensionImpl<T, std::index_sequence<NewDimensions...>> {
            using Type = Tensor<T, NewDimensions...>;
        };

        template<typename T, std::size_t NewLastDimension, std::size_t... Dimensions>
        using ReplaceLastTensorDimension = typename ReplaceLastTensorDimensionImpl<
            T,
            ReplaceLastDimension<NewLastDimension, Dimensions...>
        >::Type;

    } // namespace detail

    template<typename T, std::size_t F_out, std::size_t F_in, std::size_t... Dimensions>
    auto matvecmul(const Tensor<T, Dimensions...>& input, const Tensor<T, F_out, F_in>& matrix) noexcept {
        static_assert(detail::LastDimension<Dimensions...> == F_in);
        using InputTensorT = Tensor<T, Dimensions...>;
        using OutputTensorT = detail::ReplaceLastTensorDimension<T, F_out, Dimensions...>;
        static_assert(detail::IsTensor<OutputTensorT>);
        using OutputStorage = typename OutputTensorT::Storage;
        using GradFn = std::function<void(const OutputTensorT&)>;

        auto ptr_output = std::make_shared<OutputStorage>();
        for (auto b = std::size_t{0}, bEnd = (input.NElements / F_in); b < bEnd; ++b) {
            for (auto i = std::size_t{0}; i < F_out; ++i) {
                auto& y_i = ptr_output->get_flat(b * F_out + i);
                for (auto j = std::size_t{0}; j < F_in; ++j) {
                    const auto& x_j = input.get_flat(b * F_in + j);
                    const auto& m_i_j = matrix(i, j);
                    y_i += m_i_j * x_j;
                }
            }
        }

        auto grad_fn = [
            c_input = input,
            c_matrix = matrix
        ](const OutputTensorT& t) mutable {
            assert(t.m_grad);
            assert(c_matrix.m_grad);
            assert(c_input.m_grad);

            for (auto b = std::size_t{0}, bEnd = (c_input.NElements / F_in); b < bEnd; ++b) {
                for (auto i = std::size_t{0}; i < F_out; ++i) {
                    const auto& g = t.grad().get_flat(b * F_out + i);
                    for (auto j = std::size_t{0}; j < F_in; ++j) {
                        const auto& x_j = c_input.get_flat(b * F_in + j);
                        const auto& m_i_j = c_matrix(i, j);
                        c_matrix.grad_mut()(i, j) += x_j * g;
                        c_input.grad_mut().get_flat(b * F_in + j) += m_i_j * g;
                    }
                }
            }

            c_input.backward_impl();
            c_matrix.backward_impl();
        };

        auto ptr_grad_fn = std::make_shared<GradFn>(std::move(grad_fn));

        return OutputTensorT{std::move(ptr_output), std::move(ptr_grad_fn)};
    }

    template<typename T, std::size_t F_out, std::size_t F_in, std::size_t... Dimensions>
    auto operator%(const Tensor<T, Dimensions...>& input, const Tensor<T, F_out, F_in>& matrix) noexcept {
        return matvecmul(input, matrix);
    }

} // namespace scorch
