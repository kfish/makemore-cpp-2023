#pragma once

#include "node.h"

namespace ai {

template <size_t N>
class LogitNode {
    public:
        LogitNode()
            //: weights_(make_node(Eigen::MatrixXd(N, N)))
        {
            //Eigen::MatrixXd init = Eigen::MatrixXd::Zero(N, N);
            Eigen::MatrixXd init = Eigen::MatrixXd::Constant(N, N, 0.01);
            weights_ = make_node(init);
        }

        const Node& weights() const {
            return weights_;
        }

        Node operator()(const Node& input) const {
            // input is a column vector; transpose it to a row vector to select a row of weights_
            auto iw = transpose(input) * weights_;
#if 1
            std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                << "LogitNode(): transpose(input) * weights_ has dim (" << iw->data().rows() << ", " << iw->data().cols() << ")";
            auto eiw = exp(iw);
            //std::cerr << "\n\texp(iTw) = " << eiw << std::endl;
            //std::cerr << "\n\tnormalize(exp(iTw)) = " << normalize_rows(eiw) << std::endl;
#endif
            //return normalize_rows(exp(iw));
            return normalize_rows(eiw);
            //return normalize_cols(exp(weights_ * input));
            //return normalize_rows(exp(input * weights_));
            //return (exp(input * weights_));
        }

        void adjust(double learning_rate) {
            weights_->adjust(learning_rate);
        }

    private:
        Node weights_;
};

} // namespace ai
