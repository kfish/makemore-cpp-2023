#pragma once

#include "node.h"

namespace ai {

template <size_t ContextLength, size_t N, size_t E, size_t H, size_t M>
class LogitMLP {
    public:
        LogitMLP()
            : C_(make_node(Eigen::MatrixXd(N, E))),
            hidden_(make_node(Eigen::MatrixXd(ContextLength*E, H))),
            weights_(make_node(Eigen::MatrixXd(H, M))),
            bias_(make_node(Eigen::RowVectorXd(M)))

        {
        }

        const Node& weights() const {
            return weights_;
        }

        Node operator()(const Node& input) const {
            //std::cout << "LogitMLP(): input is " << input->rows() << " x " << input->cols() << std::endl;
            // input is a ContextLength * N matrix; transpose it to a row vector to select a row of weights_
            //return normalize_rows(exp(row_vectorize(input * C_) * hidden_ * weights_ + bias_));
            return normalize_rows(exp(tanh(row_vectorize(input * C_) * hidden_) * weights_ + bias_));
        }

        void adjust(double learning_rate) {
            C_->adjust(learning_rate);
            hidden_->adjust(learning_rate);
            weights_->adjust(learning_rate);
            bias_->adjust(learning_rate);
        }

    private:
        Node C_;
        Node hidden_;
        Node weights_;
        Node bias_;
};

} // namespace ai
