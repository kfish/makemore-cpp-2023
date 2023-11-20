
#include <iostream>
#include <ostream>
#include <fstream>

#include "multinomial.h"
//#include "logitnode.h"
#include "logitmlp.h"
#include "onehot.h"
#include "graph.h"
#include "pretty.h"

#define DEBUG

using namespace ai;

#define CONTEXT_LENGTH 3

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

int process_word(const std::string& input,
        std::function<void(const Eigen::VectorXd&, int)> train) {
    std::string_view inputView(input);
    Eigen::VectorXd contextVec = Eigen::VectorXd::Zero(CONTEXT_LENGTH * 27);

    // Function to update the context vector for a new character
    auto updateContext = [&](char newChar) {
        // Shift context to the left by 27 elements
        contextVec.segment(0, (CONTEXT_LENGTH - 1) * 27) = contextVec.segment(27, (CONTEXT_LENGTH - 1) * 27);

        // Set the last 27 elements to the new character encoding
        contextVec.segment((CONTEXT_LENGTH - 1) * 27, 27) = encode_onehot(c_to_i(newChar));
    };

    int n = 0;

    for (size_t i = 0; i <= inputView.size(); ++i) {
        char nextChar = i < inputView.size() ? tolower(inputView[i]) : '.';
        train(contextVec, c_to_i(nextChar));
        ++n;

        // Update the context vector for the next iteration
        if (i < inputView.size()) {
            updateContext(nextChar);
        }
    }

    return n;
}

std::string generate_word(std::function<char(const Eigen::VectorXd&)> sample_model) {
    Eigen::VectorXd contextVec = Eigen::VectorXd::Zero(CONTEXT_LENGTH * 27);
    std::string result;

    // Function to update the context vector for a new character
    auto updateContext = [&](char newChar) {
        // Shift context to the left by 27 elements
        contextVec.segment(0, (CONTEXT_LENGTH - 1) * 27) = contextVec.segment(27, (CONTEXT_LENGTH - 1) * 27);

        // Set the last 27 elements to the new character encoding
        contextVec.segment((CONTEXT_LENGTH - 1) * 27, 27) = encode_onehot(c_to_i(newChar));
    };

    for(;;) {
        char nextChar = sample_model(contextVec);
        if (nextChar == '.') break;

        result.push_back(nextChar);
        updateContext(nextChar);
    }

    return result;
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
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open file." << std::endl;
        return make_node(0.0);
    }

    Node loss = make_node(0.0);
    int n = 0;

    std::string word;

    auto loss_func = [&](const Eigen::VectorXd& context, int curr_index) -> void {
        auto result = f(make_node(context));
        loss = loss + log(column(result, curr_index));
    };

    int end_word = start_word + max_words;
    for (int num_words = 0; num_words < end_word && file >> word; ++num_words) {
        if (num_words < start_word)
            continue;

        //n += process_word_bigram(word, loss_func);
        n += process_word(word, loss_func);
    }

    std::cerr << "Calculated loss=" << loss << std::endl;
    std::cerr << "Read n=" << n << " bigrams" << std::endl;

    auto nll = -loss / n;

    std::cerr << "nll: " << nll << std::endl;

    return nll;
}

int main(int argc, char *argv[]) {
    if (argc == 0) {
        std::cout << "Usage: " << argv[0] << " FILE" << std::endl;
        std::cout << std::endl;
        std::cout << "  FILE: line delimited list of words" << std::endl;
        exit(1);
    }

    const std::string filename = argv[1];

    //LogitNode<CONTEXT_LENGTH*27, 27> layer;
    LogitMLP<CONTEXT_LENGTH*27, 27> layer;

    cache_onehots();

    int train_eval_split = 5000;

    // TRAIN
    std::cerr << "Start TRAIN..." << std::endl;
    auto train_nll = make_nll(layer, filename, 0, train_eval_split);
    auto train_topo = topo_sort(train_nll);

    for (int iter=0; iter<300; ++iter) {
        backward_presorted(train_nll, train_topo);

        std::cerr << "Iter " << iter << ": " << train_nll << std::endl;

        layer.adjust(50.0);

        forward_presorted(train_topo);
    }

    // EVALUATE
    std::cerr << "Start EVAL ..." << std::endl;
    auto eval_nll = make_nll(layer, filename, train_eval_split, 10000);
    std::cerr << "EVAL: " << eval_nll << std::endl;
    //auto eval_topo = topo_sort(eval_nll);
    //forward_presorted(eval_topo);

#if 0
Eigen::MatrixXd prob_matrix = Eigen::MatrixXd::Zero(CONTEXT_LENGTH*27, 27);

void extract_probability_matrix(const LogitNode<CONTEXT_LENGTH*27, 27>& layer) {

    for (size_t row=0; row < 27; ++row) {
        Node output = layer(onehots[row]);
        for (size_t col=0; col < 27; ++col) {
            prob_matrix(row, col) = output->data()(col);
        }
    }

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
#else
    std::mt19937 rng = static_mt19937();

    auto sample_model = [&](const Eigen::VectorXd& context) {
        Node input = make_node(context);
        Node output = layer(input);
        Eigen::RowVectorXd row = output->data();
        std::vector<double> prob_vector(row.data(), row.data() + row.size());
        std::discrete_distribution<int> dist(prob_vector.begin(), prob_vector.end());
        int ix = dist(rng);
        return i_to_c(ix);
    };

    for (int i=0; i<50; ++i) {
        auto word = generate_word(sample_model);
        std::cerr << word << std::endl;
    }
#endif

    //std::cout << Graph(nll) << std::endl;

    return 0;
}
