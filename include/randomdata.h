#pragma once

#include <random>
#include <array>
#include <iostream>
#include <type_traits>

#include "value.h"

namespace ai {

static inline std::mt19937 static_mt19937() {
    static unsigned int seed = 42;
    static thread_local std::mt19937 gen(seed++);
    seed = gen(); // update seed for next time
    return gen;
}

// Static inline function to generate a random T
template <typename T>
static inline Value<T> randomValue() {
    std::mt19937& gen = static_mt19937();
    std::uniform_real_distribution<T> dist(-1.0, 1.0);
    return make_value(dist(gen));
}

// Static inline function to generate a random std::array<T, N>
template <typename T, size_t N>
static inline std::array<Value<T>, N> randomArray() {
    std::array<Value<T>, N> arr;
    for (auto& element : arr) {
        element = randomValue<T>();
    }
    return arr;
}

} // namespace ai
