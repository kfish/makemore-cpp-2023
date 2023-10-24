#include <cctype>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <Eigen/Dense>

#include "matplotlibcpp.h"  // Make sure this is the version from the Cryoris fork

#include "matplotlib_main.h"
#include "multinomial.h"

#include <iomanip>

namespace plt = matplotlibcpp;

int c_to_i(char c) {
    return (c == '.') ? 0 : c - 'a' + 1;
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

Eigen::MatrixXd generate_probability_distributions(const Eigen::MatrixXd& freq_matrix) {
    Eigen::MatrixXd prob_matrix = freq_matrix; // Copy the matrix; will modify in place
    Eigen::VectorXd row_sums = freq_matrix.rowwise().sum(); // Sum along each row

    // Normalize each row
    for (int i = 0; i < freq_matrix.rows(); ++i) {
        if(row_sums(i) > 0) { // Guard against division by zero
            prob_matrix.row(i) /= row_sums(i);
        }
    }

    return prob_matrix;
}

double evaluate_model(const std::string& filename, const Eigen::MatrixXd& prob_matrix) {
    double log_likelihood = 0.0;
    int n = 0;

    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return 0.0;
    }
    std::string word;
    while (file >> word) {
        int prev_index = 0;
        for (char c : word) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;
            int curr_index = c_to_i(c);
            log_likelihood += log(prob_matrix(prev_index, curr_index));
            ++n;
            prev_index = curr_index;
        }
        log_likelihood += log(prob_matrix(prev_index, 0));
        ++n;
    }
    double nll = -log_likelihood / (double)n;

    std::cerr << "EVAL: n=" << n << " log_likelihood=" << log_likelihood << " nll=" << nll << std::endl;

    return nll;
}

int main(int argc, char *argv[]) {
    const std::string filename = argv[1];

    // Generate the bigram_freq matrix
    Eigen::MatrixXd bigram_freq = generate_bigram_distribution(filename);

    auto prob_matrix = generate_probability_distributions(bigram_freq);

    auto multinomial = MultinomialSampler(prob_matrix);

    auto generate = [&]() {
        int ix = 0;
        do {
            ix = multinomial(ix);
            std::cout << i_to_c(ix);
        } while (ix);
        std::cout << std::endl;
    };

    for (int i=0; i<50; ++i) {
        generate();
    }

    evaluate_model(filename, prob_matrix);

    // Plotting
    plt::figure_size(1024, 1024);
    plt::imshow(bigram_freq, {{"cmap", "Blues"}});

    for (int i=0; i<27; ++i) {
        for (int j=0; j<27; ++j) {
            char label[3] = {i_to_c(i), i_to_c(j), '\0'};
            plt::text(j, i, label, {{"ha", "center"}, {"va", "bottom"}, {"color", "grey"}, {"fontsize", "8"}});

            //double v = bigram_freq(i, j);
            //std::string value = std::to_string(static_cast<int>(v)); // Convert to int to avoid decimal points
            double v = prob_matrix(i, j);
            std::string value = static_cast<std::ostringstream&&>(std::ostringstream() << std::fixed << std::setprecision(2) << v).str();

            plt::text(j, i, value, {{"ha", "center"}, {"va", "top"}, {"color", "grey"}, {"fontsize", "8"}});
        }
    }

    plt::axis("off");

    return matplotlib_main(argc, argv);
}
