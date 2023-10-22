#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <cctype>
#include "matplotlibcpp.h"  // Make sure this is the version from the Cryoris fork

#include "matplotlib_main.h"

namespace plt = matplotlibcpp;

int c_to_i(char c) {
    return c - 'a' + 1;
}

char i_to_c(int i) {
    return i ? 'a' + i - 1 : '.';
}

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
            int curr_index = c_to_i(c);
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

    // Plotting
    plt::figure_size(1024, 1024);
    plt::imshow(bigram_freq, {{"cmap", "Blues"}});

    for (int i=0; i<27; ++i) {
        for (int j=0; j<27; ++j) {
            char label[3] = {i_to_c(i), i_to_c(j), '\0'};
            plt::text(j, i, label, {{"ha", "center"}, {"va", "bottom"}, {"color", "grey"}, {"fontsize", "8"}});

            double v = bigram_freq(i, j);
            std::string value = std::to_string(static_cast<int>(v)); // Convert to int to avoid decimal points
            plt::text(j, i, value, {{"ha", "center"}, {"va", "top"}, {"color", "grey"}, {"fontsize", "8"}});
        }
    }

    plt::axis("off");

    return matplotlib_main(argc, argv);
}
