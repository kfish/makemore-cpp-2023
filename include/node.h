#pragma once

#include <assert.h>
#include <cmath>
#include <Eigen/Dense>
#include <memory>
#include <set>

#include "pretty.h"

//#define DEBUG

namespace ai {

template<typename Derived>
class NodeBase {
    public:
        Derived data_;
        Derived grad_;

        // Constructor that initializes data_ and grad_
        NodeBase(const Derived& data, const Derived& grad)
            : data_(data), grad_(grad) {}

        // Return the number of rows in data_
        auto rows() const -> decltype(data_.rows()) {
            return data_.rows();
        }

        // Return the number of columns in data_
        auto cols() const -> decltype(data_.cols()) {
            return data_.cols();
        }

        void adjust(double learning_rate) {
            data_ -= learning_rate * grad_;
        }

        // Zero out the gradient
        void zerograd() {
            grad_.setZero();
        }

        // Initialize the gradient to 1.0
        void initgrad() {
            // If the Derived is a scalar (1x1 matrix), set it to 1.0
            if constexpr (Derived::RowsAtCompileTime == 1 && Derived::ColsAtCompileTime == 1) {
                grad_(0, 0) = 1.0;
            } else {
                // For vectors or matrices, set to an array or matrix of ones
                grad_.setOnes();
            }
        }
};

class NodeValue {
    public:
        using ptr = std::shared_ptr<NodeValue>;

    private:
        NodeBase<Eigen::MatrixXd> base_;

    public:
        Eigen::MatrixXd& data() {
            return base_.data_;
        }

        Eigen::MatrixXd& grad() {
            return base_.grad_;
        }

        const Eigen::MatrixXd& data() const {
            return base_.data_;
        }

        const Eigen::MatrixXd& grad() const {
            return base_.grad_;
        }

        // Forward rows() call to base_
        auto rows() const -> decltype(base_.rows()) {
            return base_.rows();
        }

        // Forward cols() call to base_
        auto cols() const -> decltype(base_.cols()) {
            return base_.cols();
        }

    private:
        // Constructor for a double scalar
        NodeValue(double value, const std::string& label="")
            : base_(Eigen::MatrixXd::Constant(1, 1, value), Eigen::MatrixXd::Zero(1, 1)),
              label_(label) {}

        // Constructor for an Eigen::VectorXd
        NodeValue(const Eigen::VectorXd& vec, const std::string& label="")
            : base_(vec, Eigen::VectorXd::Zero(vec.size())),
              label_(label) {}

        // Constructor for an Eigen::RowVectorXd
        NodeValue(const Eigen::RowVectorXd& vec, const std::string& label="")
            : base_(vec, Eigen::RowVectorXd::Zero(vec.size())),
              label_(label) {}

        // Constructor for an Eigen::MatrixXd
        NodeValue(const Eigen::MatrixXd& mat, const std::string& label="")
            : base_(mat, Eigen::MatrixXd::Zero(mat.rows(), mat.cols())),
              label_(label) {}

        static ptr make_empty(int rows, int cols) {
            return ptr(new NodeValue(Eigen::MatrixXd(rows, cols)));
        }

        static ptr make_empty_copy(const ptr& x) {
            return ptr(new NodeValue(Eigen::MatrixXd(x->rows(), x->cols())));
        }

    public:
        static ptr add_label(ptr unlabelled, const std::string& label)
        {
            unlabelled->label_ = label;
            return unlabelled;
        }

        template <typename... Args>
        static ptr make(Args&&... args) {
            return ptr(new NodeValue(std::forward<Args>(args)...));
        }

        const std::string& label() const {
            return label_;
        }

        const std::set<ptr>& children() const {
            return prev_;
        }

        const std::string& op() const {
            return op_;
        }


        // Adjust the Node's data by the gradient scaled by the learning rate
        void adjust(double learning_rate) {
            base_.adjust(learning_rate);
        }

        // Zero out the gradient
        void zerograd() {
            base_.zerograd();
        }

        // Initialize the gradient to 1.0
        void initgrad() {
            base_.initgrad();
        }

        friend void forward(const ptr& node) {
            std::vector<NodeValue*> topo;
            std::set<NodeValue*> visited;

            std::function<void(const ptr&)> build_topo = [&](const ptr& v) {
                if (!visited.contains(v.get())) {
                    visited.insert(v.get());
                    for (auto && c : v->children()) {
                        build_topo(c);
                    }
                    topo.push_back(v.get());
                }
            };

            build_topo(node);

            int n=0;

            for (auto it = topo.begin(); it != topo.end(); ++it) {
                const NodeValue* v = *it;
                auto f = v->forward_;
                if (f) {
                    ++n;
                    f();
                }
            }
        }

        friend void backward(const ptr& node) {
            std::vector<NodeValue*> topo;
            std::set<NodeValue*> visited;

            std::function<void(const ptr&)> build_topo = [&](const ptr& v) {
                if (!visited.contains(v.get())) {
                    visited.insert(v.get());
                    for (auto && c : v->children()) {
                        build_topo(c);
                    }
                    topo.push_back(v.get());
                }
            };

            build_topo(node);

            // Zero gradients first
            for (auto & v : topo) {
                v->zerograd();
            }

            node->initgrad();

            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                const NodeValue* v = *it;
                auto f = v->backward_;
                if (f) f();
            }
        }

        // operator+
        friend ptr operator+(const ptr& a, const ptr& b) {
            auto out = make_empty_copy(a);
            out->prev_ = {a, b};
            out->op_ = "+";

            out->forward_ = [=]() {
                out->data() = a->data() + b->data();
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "BACKWARD operator+:"
                    << "\n\tout->grad(): " << out->grad()
                    << "\n\ta->grad(): " << a->grad()
                    << "\n\tb->grad(): " << b->grad()
                    << std::endl;
#endif
                a->grad() += out->grad();
                b->grad() += out->grad();
            };

            out->forward_();
            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator+(const ptr& a, N n) { return a + make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator+(N n, const ptr& a) { return make(n) + a; }

        friend ptr operator+(const ptr& a, const Eigen::VectorXd& v) { return a + make(v); }
        friend ptr operator+(const Eigen::VectorXd& v, const ptr& a) { return make(v) + a; }
        friend ptr operator+(const ptr& a, const Eigen::MatrixXd& m) { return a + make(m); }
        friend ptr operator+(const Eigen::MatrixXd& m, const ptr& a) { return make(m) + a; }

        // unary operator-
        friend ptr operator-(const ptr& a) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "neg";

            out->forward_ = [=]() {
                out->data() = -a->data();
            };

            out->backward_ = [=]() {
                a->grad() -= out->grad();
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "BACKWARD operator- (unary):"
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        // operator-
        friend ptr operator-(const ptr& a, const ptr& b) {
            auto out = make_empty_copy(a);
            out->prev_ = {a, b};
            out->op_ = "-";

            out->forward_ = [=]() {
                out->data() = a->data() - b->data();
            };

            out->backward_ = [=]() {
                a->grad() += out->grad();
                b->grad() += -out->grad();
            };

            out->forward_();
            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator-(const ptr& a, N n) { return a - make(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator-(N n, const ptr& a) { return make(n) - a; }

        friend ptr operator-(const ptr& a, const Eigen::VectorXd& v) { return a - make(v); }
        friend ptr operator-(const Eigen::VectorXd& v, const ptr& a) { return make(v) - a; }
        friend ptr operator-(const ptr& a, const Eigen::MatrixXd& m) { return a - make(m); }
        friend ptr operator-(const Eigen::MatrixXd& m, const ptr& a) { return make(m) - a; }

        // operator*
        friend ptr operator*(const ptr& a, double m) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "*";

            out->forward_ = [=]() {
                out->data() = a->data() * m;
            };

            out->backward_ = [=]() {
                // Gradient with respect to weights is the upstream gradient times the input transposed
                a->grad() += out->grad() * m;
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "BACKWARD OPERATOR* (scalar)"
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(const ptr& a, N n) { return a * static_cast<double>(n); }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator*(N n, const ptr& a) { return a * static_cast<double>(n); }

        friend ptr operator*(const ptr& a, const ptr& b) {
            auto out = make_empty(a->rows(), b->cols());
            out->prev_ = {a, b};
            out->op_ = "*";

            out->forward_ = [=]() {
                out->data() = a->data() * b->data();
                //std::cerr << "forward*: a=" << PrettyMatrix(a->data()) << " * b=" << PrettyMatrix(b->data()) << std::endl;
                //std::cerr << "\tout=" << PrettyMatrix(out->data()) << std::endl;
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "BACKWARD OPERATOR*: "
                    << "out->op: " << out->op() << "\t"
                    << "out->grad():(" << out->grad().rows() << ", " << out->grad().cols() << ")\t"
                    << "a->op: " << a->op() << "\t"
                    //<< "a->data():(" << a->data().rows() << ", " << a->data().cols() << ")\t"
                    << "a->data(): " << a->data() << "\n\t"
                    << "a->grad():(" << a->grad().rows() << ", " << a->grad().cols() << ")\t"
                    << "b->op: " << b->op() << "\t"
                    //<< "b->data():(" << b->data().rows() << ", " << b->data().cols() << ")\t"
                    << "b->data(): " << b->data() << "\n\t"
                    //<< "b->grad():(" << b->grad().rows() << ", " << b->grad().cols() << ")\t"
                    ;
#endif

                // Gradient with respect to weights is the upstream gradient times the input transposed
                a->grad() += out->grad() * b->data().transpose();

                // Gradient with respect to input is the transposed weights times the upstream gradient
                b->grad() += a->data().transpose() * out->grad();
#ifdef DEBUG
                std::cerr
                    << "a->grad(): " << a->grad() << "\n\t"
                    << "b->grad(): " << b->grad() << "\n"
                    << "<BACKWARD OP* DONE>" << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        friend ptr operator*(const ptr& a, const Eigen::VectorXd& v) { return a * make(v); }
        friend ptr operator*(const Eigen::VectorXd& v, const ptr& a) { return make(v) * a; }
        friend ptr operator*(const ptr& a, const Eigen::MatrixXd& m) { return a * make(m); }
        friend ptr operator*(const Eigen::MatrixXd& m, const ptr& a) { return make(m) * a; }

        // operator/
        friend ptr operator/(const ptr& a, double m) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "/";

            out->forward_ = [=]() {
                out->data() = a->data() / m;
            };

            out->backward_ = [=]() {
                a->grad() += out->grad() * (1.0 / m);
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "BACKWARD operator/ (by scalar m, a->grad() = out->grad() / m):"
                    << "\n\tm: " << m
                    << "\n\tout->grad(): " << out->grad()
                    << "\n\ta->grad(): " << a->grad()
                    << std::endl;
#endif

            };

            out->forward_();
            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr operator/(const ptr& a, N n) { return a / static_cast<double>(n); }

        // dot
        friend ptr dot(const ptr& a, const ptr& b) {
            auto out = make(0.0);

            out->prev_ = {a, b};
            out->op_ = "dot";

            out->forward_ = [=]() {
                double result = (a->data().array() * b->data().array()).sum();
                out->data()(0, 0) = result;
            };

            out->backward_ = [=]() {
                a->grad() += out->grad()(0, 0) * b->data();
                b->grad() += out->grad()(0, 0) * a->data();
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "DOT"
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        // transpose
        friend ptr transpose(const ptr& a) {
            auto out = make_empty(a->cols(), a->rows());

            out->prev_ = {a};
            out->op_ = "transpose";

            out->forward_ = [=]() {
                out->data() = a->data().transpose();
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "TRANSPOSE: ";
#endif
                a->grad() += out->grad().transpose();
#ifdef DEBUG
                std::cerr
                    << a->grad()
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        friend ptr select_row(const ptr& a, int ix) {
            auto out = make_empty(1, a->cols());

            out->prev_ = {a};
            out->op_ = "row";

            out->forward_ = [=]() {
                out->data() = a->data().row(ix);
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "ROW: ";
#endif
                a->grad().row(ix) += out->grad();
#ifdef DEBUG
                std::cerr
                    << a->grad()
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        // TODO: RENAME TO select_col
        friend ptr column(const ptr& a, int ix) {
            auto out = make_empty(a->rows(), 1);

            out->prev_ = {a};
            out->op_ = "col";

            out->forward_ = [=]() {
                out->data() = a->data().col(ix);
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "COL: ";
#endif
                a->grad().col(ix) += out->grad();
#ifdef DEBUG
                std::cerr
                    << a->grad()
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        // normalize
        friend ptr normalize_rows(const ptr& a) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "normalize_rows";

            out->forward_ = [=]() {
#if 0
                Eigen::RowVectorXd rowSums = a->data().rowwise().sum();
                Eigen::MatrixXd replicated_row_sums = rowSums.replicate(1, a->data().cols());
                out->data() = a->data().array() / replicated_row_sums.array();
#else
                // Calculate the sum of each row
                Eigen::VectorXd rowSums = a->data().rowwise().sum();

                // Use broadcasting for normalization
                for (int i = 0; i < a->data().rows(); ++i) {
                    if (rowSums(i) != 0) { // Avoid division by zero
                        out->data().row(i) = a->data().row(i).array() / rowSums(i);
                    }
                }
#endif
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "NORMALIZE_ROWS: ";
#endif
                int rows = a->data().rows();
                int cols = a->data().cols();

                for (int i = 0; i < rows; ++i) {
                    double rowSum = a->data().row(i).sum();
                    double rowSumSquared = rowSum * rowSum;

                    for (int j = 0; j < cols; ++j) {
                        double grad = 0.0;
                        for (int k = 0; k < cols; ++k) {
                            if (j == k) {
                                grad += out->grad()(i, k) * (1 / rowSum - a->data()(i, j) / rowSumSquared);
                            } else {
                                grad -= out->grad()(i, k) * a->data()(i, k) / rowSumSquared;
                            }
                        }
                        a->grad()(i, j) += grad;
                    }
                }

#ifdef DEBUG
                std::cerr
                    << "DONE: "
                    << a->grad()
                    << std::endl;
#endif
            };

            out->forward_();
            return out;
        }

        friend ptr normalize_cols(const ptr& a) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "normalize_cols";

            out->forward_ = [=]() {
                Eigen::VectorXd colSums = a->data().colwise().sum();

                // Create a diagonal matrix from colSums for broadcasting
                Eigen::MatrixXd colSumsDiag = colSums.asDiagonal();

                // Use matrix multiplication for normalization
                out->data() = a->data() * colSumsDiag.inverse();

                //std::cerr << "norm: a=" << PrettyMatrix(a->data()) << std::endl;
                //std::cerr << "norm: out=" << PrettyMatrix(out->data()) << std::endl;
            };

            out->backward_ = [=]() {
#ifdef DEBUG
                std::cerr << basename(__FILE__) << ":" << __LINE__ << ": "
                    << "NORMALIZE_COLS: ";
#endif

                Eigen::VectorXd colSums = a->data().colwise().sum();

                for (int i = 0; i < a->data().cols(); ++i) {
                    // Extract the i-th column of the gradient of the result
                    Eigen::VectorXd grad_col = out->grad().col(i);
                    Eigen::VectorXd input_col = a->data().col(i);

                    // Compute the gradient effect for each element
                    for (int j = 0; j < a->data().rows(); ++j) {
                        // Compute the derivative of the normalization operation
                        double grad_effect = 0.0;
                        for (int k = 0; k < a->data().rows(); ++k) {
                            if (j == k) {
                                grad_effect += grad_col(k) * (colSums(i) - input_col(k)) / (colSums(i) * colSums(i));
                            } else {
                                grad_effect -= grad_col(k) * input_col(j) / (colSums(i) * colSums(i));
                            }
                        }
                        a->grad()(j, i) += grad_effect;
                    }
                }
#ifdef DEBUG
                std::cerr
                    << a->grad()
                    << std::endl;
#endif
            };

            out->forward_();

            return out;
        }

        // log
        friend ptr log(const ptr& a) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "log";

            out->forward_ = [=]() {
                // Ensure the data is positive since log is undefined for non-positive values
                assert((a->data().array() > 0.0).all() && "Logarithm is undefined for non-positive values.");

                out->data() = a->data().array().log().matrix();
            };

            out->backward_ = [=]() {
                // Element-wise 
                a->grad() += (out->grad().array() / a->data().array()).matrix();
            };

            out->forward_();

            return out;
        }

        friend ptr exp(const ptr& a) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "exp";

            out->forward_ = [=]() {
                out->data() = a->data().array().exp().matrix();
                //std::cerr << "exp: out=" << PrettyMatrix(out->data()) << std::endl;
            };

            out->backward_ = [=]() {
                // Element-wise product: out->data .* out->grad
                a->grad() += out->data().cwiseProduct(out->grad());
            };

            out->forward_();
            return out;
        }

#if 0
        // pow
        friend ptr pow(const ptr& a, double exp_value) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "pow";

            out->forward_ = [=]() {
                out->data() = a->data().array().pow(exp_value);
            };

            out->backward_ = [=]() {
                // Gradient of base when exponent is a scalar
                a->grad() += (out->grad().array() * exp_value * a->data().array().pow(exp_value - 1)).matrix();
            };

            out->forward_();
            return out;
        }

        template<typename N, std::enable_if_t<std::is_arithmetic<N>::value, int> = 0>
        friend ptr pow(const ptr& a, N n) { return pow(a, static_cast<double>(n)); }

        friend ptr pow(const ptr& a, const ptr& b) {
            auto out = make_empty_copy(a);

            out->prev_ = {a, b};
            out->op_ = "pow";

            out->forward_ = [=]() {
                if (b->data().size() == 1) { // Exponent is a scalar
                    double exp_value = b->data()(0, 0);
                    out->data() = a->data().array().pow(exp_value);
                } else {
                    assert(a->rows() == b->rows() && a->cols() == b->cols() &&
                           "Dimensions of base and exponent must match for element-wise pow.");

                    // Apply element-wise exponentiation
                    out->data() = a->data().array().pow(b->data().array());
                }
            };

            out->backward_ = [=]() {
                if (b->data().size() == 1) { // Exponent is a scalar
                    double exp_value = b->data()(0, 0);

                    // Gradient of base when exponent is a scalar
                    a->grad() += (out->grad().array() * exp_value * a->data().array().pow(exp_value - 1)).matrix();

                    // Gradient of exponent when it is a scalar
                    // Sum all the gradients since the exponent is a scalar and affects all elements of the base
                    b->grad()(0, 0) += (out->grad().array() * a->data().array().pow(exp_value) * a->data().array().log()).sum();
                } else {
                    // Gradient of base for element-wise exponent
                    a->grad() += (out->grad().array() * b->data().array() * a->data().array().pow(b->data().array() - 1)).matrix();

                    // Gradient of exponent for element-wise exponent
                    b->grad() += (out->grad().array() * a->data().array().log() * a->data().array().pow(b->data().array())).matrix();
                }
            };

            out->forward_();
            return out;
        }
#endif

    private:
        std::string label_{};
        std::set<ptr> prev_{};
        std::string op_{""};

        std::function<void()> forward_{};
        std::function<void()> backward_{};
};

using Node = typename NodeValue::ptr;

#if 0
template <typename T, typename... Args>
static Node make_node(const T& data, Args&&... args) {
    return NodeValue::make(data, std::forward<Args>(args)...);
}
#else
template <typename... Args>
static Node make_node(Args&&... args) {
    return NodeValue::make(std::forward<Args>(args)...);
}
#endif

static inline Node label(Node unlabelled, const std::string& label) {
    return NodeValue::add_label(unlabelled, label);
}

static inline std::ostream& operator<<(std::ostream& os, const NodeValue& value) {
    auto old_precision = os.precision();
    os << std::fixed << std::setprecision(8);
    os << "NodeValue("
        << "label=" << value.label() << ", "
        << "data=" << value.data() << ", "
        << "grad=" << value.grad() << ", "
        << "dim=(" << value.data().rows() << ", " << value.data().cols() << "), "
        << "op=" << value.op()
        << ")";
    os << std::setprecision(old_precision) << std::defaultfloat;
    return os;
}

static inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<NodeValue>& value) {
    return os << value.get() << "=&" << *value;
}

};
