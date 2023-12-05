
#include <iostream>
#include <fstream>

#include "batch.h"
#include "modelsampler.h"
#include "logitmlp.h"
#include "onehot.h"
#include "graph.h"
#include "pretty.h"

#define DEBUG

using namespace ai;

#define CONTEXT_LENGTH 3

#define MINIBATCH

using Model = LogitMLP<CONTEXT_LENGTH, 27, 10, 300, 27>;

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

int process_word(const std::string& input,
        std::function<void(const Eigen::MatrixXd&, int)> train) {
    std::string_view inputView(input);
    Eigen::MatrixXd context = Eigen::MatrixXd::Zero(CONTEXT_LENGTH, 27);

    // Function to update the context vector for a new character
    auto updateContext = [&](char newChar) {
        context.block(0, 0, CONTEXT_LENGTH - 1, 27) = context.block(1, 0, CONTEXT_LENGTH - 1, 27);
        context.row(CONTEXT_LENGTH - 1) = encode_onehot(c_to_i(newChar));
    };

    int n = 0;

    for (size_t i = 0; i <= inputView.size(); ++i) {
        char nextChar = i < inputView.size() ? tolower(inputView[i]) : '.';
        train(context, c_to_i(nextChar));
        ++n;

        // Update the context vector for the next iteration
        if (i < inputView.size()) {
            updateContext(nextChar);
        }
    }

    return n;
}

std::string generate_word(std::function<char(const Eigen::MatrixXd&)> sample_model) {
    Eigen::MatrixXd context = Eigen::MatrixXd::Zero(CONTEXT_LENGTH, 27);
    std::string result;

    // Function to update the context vector for a new character
    auto updateContext = [&](char newChar) {
        context.block(0, 0, CONTEXT_LENGTH - 1, 27) = context.block(1, 0, CONTEXT_LENGTH - 1, 27);
        context.row(CONTEXT_LENGTH - 1) = encode_onehot(c_to_i(newChar));
    };

    for(;;) {
        char nextChar = sample_model(context);
        if (nextChar == '.') break;

        result.push_back(nextChar);
        updateContext(nextChar);
    }

    return result;
}

template <typename F>
Node make_nll(const F& f, const std::string& filename, int max_words = 100)
{
    auto all_words = read_file(filename);
    auto words = extract_minibatch(all_words, max_words);

    Node loss = make_node(0.0);
    int n = 0;

    std::string word;

    auto loss_func = [&](const Eigen::MatrixXd& context, int curr_index) -> void {
        //std::cout << "loss_func: input is " << context.rows() << " x " << context.cols() << std::endl;
        auto result = f(make_node(context));
        loss = loss + log(select_column(result, curr_index));
    };

    int end_word = std::min(static_cast<int>(words.size()), max_words);
    for (int num_words = 0; num_words < end_word; ++num_words) {
        n += process_word(words[num_words], loss_func);
    }

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

    std::ofstream loss_output("loss.tsv");

    Model layer;

    std::cout << "Model: " << layer.model_params() << " params" << std::endl;

    int eval_count = 10000;

    // TRAIN
    std::cerr << "Start TRAIN..." << std::endl;

#ifdef MINIBATCH
    int batch_count = 100;
#else
    int train_count = 25000;

    auto train_nll = make_nll(layer, filename, train_count);
    auto train_topo = topo_sort(train_nll);

    std::cerr << "train_nll: " << count_params_presorted(train_topo) << " params" << std::endl;
#endif

    for (int iter=0; iter<1000; ++iter) {
#ifdef MINIBATCH
        auto train_nll = make_nll(layer, filename, batch_count);
        auto train_topo = topo_sort(train_nll);
#endif

        forward_presorted(train_topo);
        backward_presorted(train_nll, train_topo);

        std::cerr << "Iter " << iter << ": " << train_nll << std::endl;
        double result = train_nll->data()(0, 0);
        loss_output << iter << '\t' << result << '\n';

        layer.adjust(5.0);
    }


    // EVALUATE
    std::cerr << "Start EVAL ..." << std::endl;
    auto eval_nll = make_nll(layer, filename, eval_count);
    std::cerr << "EVAL: " << eval_nll << ": "
        << count_params(eval_nll) << " params" << std::endl;
    //auto eval_topo = topo_sort(eval_nll);
    //forward_presorted(eval_topo);

    auto model_sampler = ModelSampler(layer);

    auto sample_model = [&](const Eigen::MatrixXd& context) -> char {
        int ix = model_sampler(context);
        return i_to_c(ix);
    };

    for (int i=0; i<50; ++i) {
        auto word = generate_word(sample_model);
        std::cerr << word << std::endl;
    }

    //std::cout << Graph(nll) << std::endl;

    return 0;
}
