#pragma once

#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "randomdata.h"

namespace ai {

std::vector<std::string> read_file(const std::string& filename) {
    std::vector<std::string> lines;
    std::ifstream file(filename);
    std::string line;

    if (!file) {
        // If the file couldn't be opened, throw an exception with the error message
        throw std::ios_base::failure("Error opening file: " + filename);
    }

    while (std::getline(file, line)) {
        lines.push_back(line);
    }

    return lines;
}

std::vector<int> minibatch_indices(int N, int batchSize) {
    std::unordered_set<int> uniqueIndices;
    std::vector<int> indices;
    std::mt19937 gen(static_mt19937());
    std::uniform_int_distribution<> dis(1, N);

    while (uniqueIndices.size() < static_cast<size_t>(batchSize)) {
        int index = dis(gen);
        if (uniqueIndices.insert(index).second) { // Check if the index is newly inserted
            indices.push_back(index);
        }
    }

    return indices;
}

template <typename T>
std::vector<T> extract_minibatch(const std::vector<T>& dataset, int batchSize) {
    std::vector<T> result(batchSize);

    auto indices = minibatch_indices(dataset.size(), batchSize);
    for (int ix = 0; ix < batchSize; ++ix) {
        result[ix] = dataset[indices[ix]];
    }

    return result;
}

} // namespace ai
