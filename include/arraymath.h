#pragma once

#include <numeric>
#include <array>
#include <execution>

namespace ai {

// Static inline function to generate a random std::array<T, N>
template <typename T, size_t N>
static inline std::array<Value<T>, N> zeroArray() {
    std::array<Value<T>, N> arr;
    for (auto& element : arr) {
        element = make_value(0.0);
    }
    return arr;
}

template <typename T, std::size_t N>
T sum(const std::array<T, N>& arr) {
    static_assert(N > 0, "Array cannot be empty");
    // Start accumulating from the first element
    return std::reduce(std::execution::seq, arr.begin() + 1, arr.end(), arr[0]);
}

template <typename T, std::size_t N>
T mean(const std::array<T, N>& arr) {
    return sum(arr) / N;
}

template <typename T, std::size_t N>
std::array<T, N> norm(const std::array<T, N>& a) {
    auto s = sum(a);

    std::array<T, N> output{};
    std::transform(std::execution::par_unseq, a.begin(), a.end(), output.begin(),
            [&](const T& v) { return v / s; });

    return output;
}

} // namespace ai
