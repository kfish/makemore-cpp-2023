#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <cctype>

Eigen::MatrixXi generate_bigram_distribution(const std::string& filename) {
    Eigen::MatrixXi bigram_freq = Eigen::MatrixXi::Zero(27, 27);  // Initialize 27x27 matrix with zeros
    std::ifstream file(filename);

    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return bigram_freq;
    }

    std::string word;
    while (file >> word) {
        int prev_index = 0;  // Using 0 for '.'
        for (char c : word) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;  // Skip non-alphabetical characters

            int curr_index = c - 'a' + 1;  // 1-26 for a-z
            ++bigram_freq(prev_index, curr_index);
            prev_index = curr_index;
        }
        if (prev_index != 0) {
            ++bigram_freq(prev_index, 0);  // Increment frequency for the end token
        }
    }

    return bigram_freq;
}

int main(int argc, char *argv[]) {
    const std::string filename = argv[1];
    Eigen::MatrixXi bigram_freq = generate_bigram_distribution(filename);

    std::cout << bigram_freq << std::endl;

    return 0;
}
