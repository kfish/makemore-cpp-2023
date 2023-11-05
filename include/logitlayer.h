#pragma once

#include <cmath>
#include <functional>
#include <numeric>

#include "array.h"
#include "arraymath.h"
#include "tuple.h"
#include "mac.h"
#include "randomdata.h"
#include "value.h"

namespace ai {

template <typename T, size_t Nin>
class LogitNeuron {
    public:
        LogitNeuron()
            : weights_(randomArray<T, Nin>())
        {
        }

        Value<T> operator()(const std::array<Value<T>, Nin>& x) const {
            Value<T> zero = make_value<T>(0.0);
            Value<T> y = mac(weights_, x, zero);
            return expr(exp(y), "n");
        }

        const std::array<Value<T>, Nin>& weights() const {
            return weights_;
        }

        void adjust_weights(const T& learning_rate) {
            for (const auto& w : weights_) {
                w->adjust(learning_rate);
            }
        }

        void adjust(const T& learning_rate) {
            adjust_weights(learning_rate);
        }

    private:
        std::array<Value<T>, Nin> weights_{};
};

template <typename T, size_t Nin>
static inline std::ostream& operator<<(std::ostream& os, const LogitNeuron<T, Nin>& n) {
    return os << "LogitNeuron(" << PrettyArray(n.weights()) << ")";
}

template <typename T, size_t Nin, size_t Nout>
class LogitLayer {
    public:
        std::array<Value<T>, Nout> operator()(const std::array<Value<T>, Nin>& x) const {
            std::array<Value<T>, Nout> counts;
            std::transform(std::execution::par_unseq, neurons_.begin(), neurons_.end(),
                    counts.begin(), [&](const auto& n) { return n(x); });

            return norm(counts);
        }
        const std::array<LogitNeuron<T, Nin>, Nout> neurons() const {
            return neurons_;
        }

        const Value<T>& weight(int row, int col) const {
            return neurons_[row].weights()[col];
        }

        void adjust(const T& learning_rate) {
            for (auto & n : neurons_) {
                n.adjust(learning_rate);
            }
        }

    private:
        std::array<LogitNeuron<T, Nin>, Nout> neurons_{};
};

template <typename T, size_t Nin, size_t Nout>
static inline std::ostream& operator<<(std::ostream& os, const LogitLayer<T, Nin, Nout>& l) {
    return os << "LogitLayer(" << PrettyArray(l.neurons()) << ")";
}

}
