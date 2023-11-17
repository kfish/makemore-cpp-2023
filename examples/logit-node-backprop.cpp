
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

static std::array<Node, 27> log_likelihoods{};
static std::array<Node, 27> onehots{};

static inline void cache_onehots() {
    for (int row=0; row < 27; ++row) {
        onehots[row] = make_node(encode_onehot(row));
    }
}

template <typename F>
void recalc_log_likelihoods(const F& f) {
    for (int row=0; row < 27; ++row) {
        const auto output = f(onehots[row]);
#ifdef DEBUG
        //std::cerr << "log_likelihood(" << i_to_c(row) << "): " << output << std::endl;
#endif
        std::ostringstream oss;
        oss << "log(" << i_to_c(row) << ")";
        log_likelihoods[row] = label(log(output), oss.str());
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

    std::array<Eigen::RowVectorXd, 27> counts;
    for (int row=0; row<27; ++row) {
        counts[row] = Eigen::RowVectorXd::Zero(27);
    }

    recalc_log_likelihoods(f);

    std::string word;
    int num_words = 0;

    while (num_words < max_words && file >> word) {
        ++num_words;
        int prev_index = 0;
        for (char c : word) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;
            int curr_index = c_to_i(c);

#if 1
            //auto prev = encode_onehot<double>(prev_index);
            //std::cerr << PrettyArray(prev) << std::endl;
            auto result = f(onehots[prev_index]);
            loss = loss + log(column(result, curr_index));
#else
            //loss = loss + column(log_likelihoods[prev_index], curr_index);
            ++counts[prev_index](curr_index);
#endif
            ++n;

            prev_index = curr_index;
        }
        if (prev_index != 0) {
#if 1
            //auto prev = encode_onehot<double>(prev_index);
            auto result = f(onehots[prev_index]);
            loss = loss + log(column(result, 0));
#else
            //loss = loss + column(log_likelihoods[prev_index], 0);
            ++counts[prev_index](0);
#endif
            ++n;
        }
    }


#if 0 // no cache
#if 1

#if 1
    for (size_t row=0; row < 27; ++row) {
        auto row_node = make_node(counts[row]);
        for (size_t col=0; col < 27; ++col) {
            if (counts[row](col) > 0) {
                //std::cerr << "make_nll: " << i_to_c(row) << i_to_c(col)
                //    << ": " << (int)(counts[row](col)) << std::endl;
                loss = loss + (column(row_node, col)) * column(log_likelihoods[row], col);
            }
        }
    }
#else
    size_t row = c_to_i('e');
        auto row_node = make_node(counts[row]);
        for (int col=c_to_i('.'); col < c_to_i('b'); ++col) {
            if (counts[row][col] > 0) {
                //std::cerr << "make_nll: " << i_to_c(row) << i_to_c(col)
                //    << ": " << counts[row][col] << std::endl;
                loss = loss + (column(row_node, col)) * column(log_likelihoods[row], col);
            }
        }
#endif

#else
    //loss = (log_likelihoods.cwiseProduct(counts)).sum();
    for (size_t row=0; row < 27; ++row) {
#ifdef DEBUG
        std::cerr << "Adding " << i_to_c(row) << " row=" << row << ":"
            << "\n\tcounts[row]=" << counts[row]
            << "\n\tlog_likelihoods[row]=" << log_likelihoods[row]
            << std::endl;
        auto d = dot(make_node(counts[row]), log_likelihoods[row]);
        std::cerr << "\n\tDot product counts . log_likelihoods = " << d << std::endl;
#endif
        loss = loss + dot(make_node(counts[row]), log_likelihoods[row]);
    }
#endif
#endif

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
    //recalc_log_likelihoods(layer);

    auto nll = make_nll(layer, filename, 100000);

#if 0
    backward(nll);
    std::cout << Graph(nll) << std::endl;
    return 0;
#endif

    for (int iter=0; iter<300; ++iter) {
        //auto nll = make_nll(layer, filename, 500000);
        backward(nll);

        std::cerr << "Iter " << iter << ": " << nll << std::endl;

        //std::cerr << "WGrad: " << PrettyMatrix(layer.weights()->grad()) << std::endl;

        layer.adjust(50.0);

        //std::cerr << "Weights: " << PrettyMatrix(layer.weights()->data()) << std::endl;

        forward(nll);

        //recalc_log_likelihoods(layer);
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
