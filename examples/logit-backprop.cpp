#include <iostream>

#include "batch.h"
#include "multinomial.h"
#include "logitlayer.h"
#include "onehot.h"
#include "graph.h"

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

template <typename T>
static inline std::array<Value<T>, 27> encode_onehot(size_t i) {
    std::array<T, 27> arr{};
    arr[i] = 1.0;
    return value_array(arr);
}

static std::array<std::array<double, 27>, 27> grads{};
static std::array<std::array<Value<double>, 27>, 27> likelihoods{};
static std::array<std::array<Value<double>, 27>, 27> log_likelihoods{};
static std::array<std::array<int, 27>, 27> counts{};
static std::array<std::array<Value<double>, 27>, 27> onehots{};

static inline void cache_onehots() {
    for (int row=0; row < 27; ++row) {
        onehots[row] = encode_onehot<double>(row);
    }
}

template <typename F>
void recalc_log_likelihoods(const F& f) {
    for (size_t row=0; row < 27; ++row) {
        //const std::array<Value<double>, 27> input = encode_onehot<double>(row);
        const std::array<Value<double>, 27>& input = onehots[row];
        const std::array<Value<double>, 27> output = f(input);
        for (size_t col=0; col < 27; ++col) {
            grads[row][col] = f.weight(row, col)->grad();
            likelihoods[row][col] = output[col];
            //log_likelihoods[row][col] = log(output[col]);
            log_likelihoods[row][col] = expr(log(output[col]), string_pair(row, col));
        }
    }
}

template <typename F>
Value<double> make_nll(const F& f, const std::string& filename, int max_words = 100)
{
    auto words = read_file(filename);

    Value<double> loss = make_value(0.0);
    int n = 0;

    recalc_log_likelihoods(f);

    for (size_t row=0; row < 27; ++row) {
        for (size_t col=0; col < 27; ++col) {
            counts[row][col] = 0;
        }
    }

    int num_words = 0;
    int end_word = std::min(static_cast<int>(words.size()), max_words);

    while (num_words < end_word) {
        ++num_words;
        int prev_index = 0;
        for (char c : words[num_words]) {
            c = std::tolower(c);
            if (c < 'a' || c > 'z') continue;
            int curr_index = c_to_i(c);

#if 0
            auto prev = encode_onehot<double>(prev_index);
            //std::cout << PrettyArray(prev) << std::endl;
            auto result = f(prev);
            loss += log(result[curr_index]);
#else
            //loss += log_likelihoods[prev_index][curr_index];
            ++counts[prev_index][curr_index];
#endif
            ++n;

            prev_index = curr_index;
        }
        if (prev_index != 0) {
#if 0
            auto prev = encode_onehot<double>(prev_index);
            auto result = f(prev);
            loss += log(result[0]);
#else
            //loss += log_likelihoods[prev_index][0];
            ++counts[prev_index][0];
#endif
            ++n;
        }
    }

    for (size_t row=0; row < 27; ++row) {
        for (size_t col=0; col < 27; ++col) {
            if (counts[row][col] > 0) {
                std::cerr << "make_nll: " << i_to_c(row) << i_to_c(col)
                    << ": " << counts[row][col] << std::endl;
                loss = loss + (counts[row][col] * log_likelihoods[row][col]);
            }
        }
    }

    auto nll = -loss / n;

    return nll;
}

Eigen::MatrixXd prob_matrix = Eigen::MatrixXd::Zero(27, 27);

//Eigen::MatrixXd extract_probability_matrix(const LogitLayer<double, 27, 27>& layer) {
void extract_probability_matrix(const LogitLayer<double, 27, 27>& layer) {

    for (size_t row=0; row < 27; ++row) {
        const std::array<Value<double>, 27> input = encode_onehot<double>(row);
        const std::array<Value<double>, 27> output = layer(input);
        for (size_t col=0; col < 27; ++col) {
            prob_matrix(row, col) = output[col]->data();
        }
    }

#if 0
    std::cerr << "cnts 0 0 : " << counts[0][0] << std::endl;
    std::cerr << "prob 0 0 : " << prob_matrix(0, 0) << std::endl;
    std::cerr << "prob 0 1 : " << prob_matrix(0, 1) << std::endl;
    std::cerr << "grad 0 0 : " << grads[0][0] << std::endl;
    std::cerr << "grad 0 1 : " << grads[0][1] << std::endl;

    for (int i=0; i<27; ++i) {
        //std::cerr << "prob 0 " << i << " " << i_to_c(i) << " : " << prob_matrix(0, i) << std::endl;
    }
#endif

    //return prob_matrix;
}

int main(int argc, char *argv[]) {
    if (argc == 0) {
        std::cout << "Usage: " << argv[0] << " FILE" << std::endl;
        std::cout << std::endl;
        std::cout << "  FILE: line delimited list of words" << std::endl;
        exit(1);
    }

    const std::string filename = argv[1];

    cache_onehots();

    LogitLayer<double, 27, 27> layer;
    std::cout << "Model: " << layer.model_params() << " params" << std::endl;

    //recalc_log_likelihoods(layer);

    auto nll = make_nll(layer, filename, 500000);

    std::cerr << "nll: " << count_params(nll) << " params" << std::endl;

#if 0
    backward(nll);
    std::cout << Graph(nll) << std::endl;
    return 0;
#endif

    for (int iter=0; iter<300; ++iter) {
        //auto nll = make_nll(layer, filename, 500000);
        backward(nll);

        std::cerr << "Iter " << iter << ": " << nll << std::endl;
        //std::cerr << "\tem: " << layer.weight(c_to_i('e'), c_to_i('m')) << std::endl;

        layer.adjust(50.0);

        //
        forward(nll);
        recalc_log_likelihoods(layer);

        //std::cerr << "POST Iter " << iter << ": " << nll << std::endl;
        //std::cerr << "\tem: " << layer.weight(c_to_i('e'), c_to_i('m')) << std::endl;

        //auto prob_matrix = extract_probability_matrix(layer);
        //extract_probability_matrix(layer);
        //std::cerr << prob_matrix << std::endl;

    }


    //auto prob_matrix = extract_probability_matrix(layer);
    extract_probability_matrix(layer);
    auto multinomial = MultinomialSampler(prob_matrix);

    //return 0;

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
