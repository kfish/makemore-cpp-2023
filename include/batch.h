#pragma once

#include <fstream>
#include <string>
#include <vector>

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
