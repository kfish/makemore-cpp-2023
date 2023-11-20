#pragma once

#include "node.h"

namespace ai {

template <size_t N, size_t M>
class LogitMLP {
    public:
        LogitMLP()
            : weights_(make_node(Eigen::MatrixXd(N, M))),
            bias_(make_node(Eigen::RowVectorXd(M)))

        {
        }

        const Node& weights() const {
            return weights_;
        }

        Node operator()(const Node& input) const {
            // input is a column vector of length N; transpose it to a row vector to select a row of weights_
            return normalize_rows(exp((transpose(input) * weights_)) + bias_);
        }

        void adjust(double learning_rate) {
            weights_->adjust(learning_rate);
        }

    private:
        Node weights_;
        Node bias_;
};

} // namespace ai
