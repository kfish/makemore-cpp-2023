#include <iostream>
#include <fstream>
#include <string>
#include <array>
#include <cctype>

std::array<std::array<int, 27>, 27> generate_bigram_distribution(const std::string& filename) {
    std::array<std::array<int, 27>, 27> bigram_freq = {0};
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return bigram_freq;
    }

    std::string word;

    auto to_ix = [](char c) { return 1 + c - 'a'; };

    while (file >> word) {
        char prev_char = '.';
        for (char c : word) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;  // Skip non-alphabetical characters

            ++bigram_freq[prev_char == '.' ? 0 : to_ix(prev_char)][to_ix(c)];
            prev_char = c;
        }
        if (prev_char != '.') {
            ++bigram_freq[to_ix(prev_char)][0];  // Increment frequency for the end token
        }
    }

    return bigram_freq;
}

int main(int argc, char *argv[]) {
    const std::string filename = argv[1];
    auto bigram_freq = generate_bigram_distribution(filename);

    for (const auto& row : bigram_freq) {
        for (int freq : row) {
            std::cout << freq << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}
