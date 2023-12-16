#pragma once

#include "node.h"

namespace ai {

template <size_t ContextLength, size_t N, size_t E, size_t H, size_t M>
class LogitMLP {
    public:
        LogitMLP()
            : C_(make_node(Eigen::MatrixXd(N, E))),
            W1_(make_node(Eigen::MatrixXd(ContextLength*E, H))),
            B1_(make_node(Eigen::RowVectorXd(H))),
            W2_(make_node(Eigen::MatrixXd(H, M))),
            B2_(make_node(Eigen::RowVectorXd(M)))
        {
        }

        size_t model_params() const {
            return count_params(C_) + count_params(W1_) + count_params(W2_) + count_params(B2_);
        }

        Node internal(const Node& input) const {
            return tanh(row_vectorize(input * C_) * W1_ + B1_) * W2_ + B2_;
        }

        Node operator()(const Node& input) const {
            return softmax(internal(input));
        }

        void adjust(double learning_rate) {
            C_->adjust(learning_rate);
            W1_->adjust(learning_rate);
            B1_->adjust(learning_rate);
            W2_->adjust(learning_rate);
            B2_->adjust(learning_rate);
        }

    private:
        Node C_;
        Node W1_;
        Node B1_;
        Node W2_;
        Node B2_;
};

} // namespace ai
