#include <iostream>

#include "matplotlib_main.h"
#include "onehot.h"

int c_to_i(char c) {
    return (c == '.') ? 0 : c - 'a' + 1;
}

char i_to_c(int i) {
    return i ? 'a' + i - 1 : '.';
}

static inline Eigen::VectorXd encode_onehot(char c) {
    return OneHot(27, c_to_i(c));
}

static inline Eigen::MatrixXd encode_onehot(const std::string& word) {
    Eigen::MatrixXd matrix(word.size(), 27);

    for (size_t i = 0; i < word.size(); ++i) {
        matrix.row(i) = encode_onehot(tolower(word[i]));
    }

    return matrix;
}

int main(int argc, char *argv[]) {
    std::string xs = ".emma";

    auto xenc = encode_onehot(xs);

    plt::figure_size(640, 180);
    plt::imshow(xenc);

    return matplotlib_main(argc, argv);
}
