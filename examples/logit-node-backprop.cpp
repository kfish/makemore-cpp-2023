
#include <iostream>
#include <ostream>
#include <fstream>

#include "multinomial.h"
#include "logitnode.h"
#include "onehot.h"
#include "graph.h"
#include "pretty.h"

#define DEBUG

using namespace ai;

int c_to_i(char c) {
    return (c >= 'a' && c <= 'z') ? c - 'a' + 1 : 0;
}

char i_to_c(int i) {
    return i ? 'a' + i - 1 : '.';
}

std::string string_pair(int row, int col) {
    char pair[3];
    pair[0] = i_to_c(row);
    pair[1] = i_to_c(col);
    pair[2] = '\0';
    return std::string(pair);
}

static inline Eigen::VectorXd encode_onehot(size_t i) {
    return OneHot(27, i);
}

static std::array<Node, 27> onehots{};

static inline void cache_onehots() {
    for (int row=0; row < 27; ++row) {
        onehots[row] = make_node(encode_onehot(row));
    }
}

template <typename F>
Node make_nll(const F& f, const std::string& filename, int max_words = 100)
{
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return make_node(0.0);
    }

    Node loss = make_node(0.0);
    int n = 0;

    cache_onehots();

    std::string word;
    int num_words = 0;

    while (num_words < max_words && file >> word) {
        ++num_words;
        int prev_index = 0;
        for (char c : word) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;
            int curr_index = c_to_i(c);

            //auto prev = encode_onehot<double>(prev_index);
            //std::cerr << PrettyArray(prev) << std::endl;
            auto result = f(onehots[prev_index]);
            loss = loss + log(column(result, curr_index));

            ++n;

            prev_index = curr_index;
        }
        if (prev_index != 0) {
            //auto prev = encode_onehot<double>(prev_index);
            auto result = f(onehots[prev_index]);
            loss = loss + log(column(result, 0));
            ++n;
        }
    }

    std::cerr << "Calculated loss=" << loss << std::endl;
    std::cerr << "Read n=" << n << " bigrams" << std::endl;

    auto nll = -loss / n;

    std::cerr << "nll: " << nll << std::endl;

    return nll;
}


Eigen::MatrixXd prob_matrix = Eigen::MatrixXd::Zero(27, 27);

void extract_probability_matrix(const LogitNode<27>& layer) {

    for (size_t row=0; row < 27; ++row) {
        Node output = layer(onehots[row]);
        for (size_t col=0; col < 27; ++col) {
            prob_matrix(row, col) = output->data()(col);
        }
    }

}

int main(int argc, char *argv[]) {
    if (argc == 0) {
        std::cout << "Usage: " << argv[0] << " FILE" << std::endl;
        std::cout << std::endl;
        std::cout << "  FILE: line delimited list of words" << std::endl;
        exit(1);
    }

    const std::string filename = argv[1];

    LogitNode<27> layer;
    //std::cerr << "Initial Weights: " << PrettyMatrix(layer.weights()->data()) << std::endl;

    cache_onehots();

    auto nll = make_nll(layer, filename, 100000);
    auto topo = topo_sort(nll);

#if 0
    backward(nll);
    std::cout << Graph(nll) << std::endl;
    return 0;
#endif

    for (int iter=0; iter<300; ++iter) {
        //auto nll = make_nll(layer, filename, 500000);
        backward_presorted(nll, topo);

        std::cerr << "Iter " << iter << ": " << nll << std::endl;

        //std::cerr << "WGrad: " << PrettyMatrix(layer.weights()->grad()) << std::endl;

        layer.adjust(50.0);

        //std::cerr << "Weights: " << PrettyMatrix(layer.weights()->data()) << std::endl;

        forward_presorted(topo);
    }

    //auto prob_matrix = extract_probability_matrix(layer);
    extract_probability_matrix(layer);
    auto multinomial = MultinomialSampler(prob_matrix);

    auto generate = [&]() {
        int ix = 0;
        do {
            ix = multinomial(ix);
            std::cerr << i_to_c(ix);
        } while (ix);
        std::cerr << std::endl;
    };

    for (int i=0; i<50; ++i) {
        generate();
    }

    //std::cout << Graph(nll) << std::endl;

    return 0;
}
