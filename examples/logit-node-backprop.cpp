
#include <iostream>
#include <ostream>

#include "batch.h"
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

int process_word_bigram(const std::string& input,
        std::function<void(const Eigen::VectorXd&, int)> train) {
    int prev_index = 0;
    int n = 0;
    for (char c : input) {
        c = std::tolower(c);
        if (c < 'a' || c > 'z') continue;
        int curr_index = c_to_i(c);

        train(encode_onehot(prev_index), curr_index);
        ++n;

        prev_index = curr_index;
    }
    if (prev_index != 0) {
        train(encode_onehot(prev_index), 0);
        ++n;
    }

    return n;
}

template <typename F>
Node make_nll(const F& f, const std::string& filename, int start_word = 0, int max_words = 100)
{
    auto words = read_file(filename);

    Node loss = make_node(0.0);
    int n = 0;

    std::string word;

    auto loss_func = [&](const Eigen::VectorXd& context, int curr_index) -> void {
        auto result = f(make_node(context));
        loss = loss + log(column(result, curr_index));
    };

    int end_word = std::min(static_cast<int>(words.size()), start_word + max_words);
    for (int num_words = 0; num_words < end_word; ++num_words) {
        if (num_words < start_word)
            continue;

        n += process_word_bigram(words[num_words], loss_func);
    }

    std::cerr << "Calculated loss=" << loss << std::endl;
    std::cerr << "Read n=" << n << " bigrams" << std::endl;

    auto nll = -loss / n;

    std::cerr << "nll: " << nll << std::endl;

    return nll;
}

Eigen::MatrixXd extract_probability_matrix(const LogitNode<27, 27>& layer) {
    Eigen::MatrixXd prob_matrix = Eigen::MatrixXd::Zero(27, 27);

    for (size_t row=0; row < 27; ++row) {
        Node output = layer(onehots[row]);
        for (size_t col=0; col < 27; ++col) {
            prob_matrix(row, col) = output->data()(col);
        }
    }

    return prob_matrix;
}

int main(int argc, char *argv[]) {
    if (argc == 0) {
        std::cout << "Usage: " << argv[0] << " FILE" << std::endl;
        std::cout << std::endl;
        std::cout << "  FILE: line delimited list of words" << std::endl;
        exit(1);
    }

    const std::string filename = argv[1];

    LogitNode<27, 27> layer;
    std::cout << "Model: " << layer.model_params() << " params" << std::endl;

    cache_onehots();

    auto nll = make_nll(layer, filename, 0, 100000);
    auto topo = topo_sort(nll);

    std::cerr << "nll: " << count_params_presorted(topo) << " params" << std::endl;

    for (int iter=0; iter<300; ++iter) {
        backward_presorted(nll, topo);

        std::cerr << "Iter " << iter << ": " << nll << std::endl;

        layer.adjust(50.0);

        forward_presorted(topo);
    }

    auto prob_matrix = extract_probability_matrix(layer);
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
