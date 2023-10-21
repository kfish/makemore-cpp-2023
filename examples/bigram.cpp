#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cctype>
#include "matplotlibcpp.h"  // Make sure this is the version from the Cryoris fork

namespace plt = matplotlibcpp;

Eigen::MatrixXd generate_bigram_distribution(const std::string& filename) {
    Eigen::MatrixXd bigram_freq = Eigen::MatrixXd::Zero(27, 27);
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return bigram_freq;
    }
    std::string word;
    while (file >> word) {
        int prev_index = 0;
        for (char c : word) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;
            int curr_index = c - 'a' + 1;
            ++bigram_freq(prev_index, curr_index);
            prev_index = curr_index;
        }
        if (prev_index != 0) {
            ++bigram_freq(prev_index, 0);
        }
    }
    return bigram_freq;
}

int main(int argc, char *argv[]) {
    const std::string filename = argv[1];

    // Generate the bigram_freq matrix
    Eigen::MatrixXd bigram_freq = generate_bigram_distribution(filename);

    // Flattening the Eigen::MatrixXd to Eigen::VectorXd using Eigen::Map
    //Eigen::Map<Eigen::VectorXd> data(bigram_freq.data(), bigram_freq.size());

    // Plotting
    plt::imshow(bigram_freq, {{"cmap", "Blues"}});
    plt::title("Bigram Frequency Distribution");
    plt::xlabel("Next Character (Index)");
    plt::ylabel("Current Character (Index)");
    plt::colorbar();
    plt::show();

    return 0;
}
