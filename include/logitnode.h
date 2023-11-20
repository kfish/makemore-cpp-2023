#pragma once

#include "node.h"

namespace ai {

template <size_t N, size_t M>
class LogitNode {
    public:
        LogitNode()
            : weights_(make_node(Eigen::MatrixXd(N, M)))
        {
        }

        const Node& weights() const {
            return weights_;
        }

        Node operator()(const Node& input) const {
            // input is a column vector; transpose it to a row vector to select a row of weights_
            return normalize_rows(exp(transpose(input) * weights_));
        }

        void adjust(double learning_rate) {
            weights_->adjust(learning_rate);
        }

    private:
        Node weights_;
};

} // namespace ai
