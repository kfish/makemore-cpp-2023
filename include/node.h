#pragma once

#include <assert.h>
#include <cmath>
#include <Eigen/Dense>
#include <memory>
#include <unordered_map>
#include <vector>
#include <stack>
#include <set>

#include "pretty.h"

namespace ai {

template<typename Derived>
class NodeBase {
    public:
        Derived data_;
        Derived grad_;

        // Constructor that initializes data_ and grad_
        NodeBase(const Derived& data, const Derived& grad)
            : data_(data), grad_(grad) {}

        // Return size (rows * cols) of data_
        auto size() const -> decltype(data_.size()) {
            return data_.size();
        }

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

        // Forward size() call to base_
        auto size() const -> decltype(base_.size()) {
            return base_.size();
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

        const std::vector<ptr>& children() const {
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

        friend std::vector<NodeValue*> topo_sort(const ptr& node) {
            std::vector<NodeValue*> topo;
            std::unordered_map<NodeValue*, bool> visited;
            std::stack<NodeValue*> stack;

            // Start by pushing the root node onto the stack
            stack.push(node.get());

            while (!stack.empty()) {
                NodeValue* v = stack.top();
                stack.pop();

                if (visited[v]) {
                    // Node already visited, skip
                    continue;
                }

                // Mark the node as visited
                visited[v] = true;

                // Push the node into the topological order
                topo.push_back(v);

                // Iterate over the children in reverse order to maintain the correct ordering
                // when they are popped from the stack, because the stack is LIFO.
                const auto& childrenNodes = v->children();
                for (auto it = childrenNodes.rbegin(); it != childrenNodes.rend(); ++it) {
                    if (!visited[it->get()]) {
                        stack.push(it->get());
                    }
                }
            }

            // The topo vector will be in reverse order, so reverse it before returning
            std::reverse(topo.begin(), topo.end());

            return topo;
        }

        friend size_t count_params_presorted(const std::vector<NodeValue*>& topo) {
            size_t num_params = 0;

            for (auto it = topo.begin(); it != topo.end(); ++it) {
                const NodeValue* v = *it;
                num_params += v->size();
            }

            return num_params;
        }

        friend size_t count_params(const ptr& node) {
            std::vector<NodeValue*> topo = topo_sort(node);

            return count_params_presorted(topo);
        }

        friend void forward_presorted(const std::vector<NodeValue*>& topo) {
            for (auto it = topo.begin(); it != topo.end(); ++it) {
                const NodeValue* v = *it;
                auto f = v->forward_;
                if (f) f();
            }
        }

        friend void forward(const ptr& node) {
            std::vector<NodeValue*> topo = topo_sort(node);

            forward_presorted(topo);
        }

        friend void backward_presorted(const ptr& node, const std::vector<NodeValue*>& topo) {
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

        friend void backward(const ptr& node) {
            std::vector<NodeValue*> topo = topo_sort(node);

            backward_presorted(node, topo);
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
                a->grad() += out->grad() * m;
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
            };

            out->backward_ = [=]() {
                // Gradient with respect to weights is the upstream gradient times the input transposed
                a->grad() += out->grad() * b->data().transpose();

                // Gradient with respect to input is the transposed weights times the upstream gradient
                b->grad() += a->data().transpose() * out->grad();
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
               a->grad() += out->grad().transpose();
            };

            out->forward_();
            return out;
        }

        // row_vectorize: to row vector
        friend ptr row_vectorize(const ptr& a) {
            auto out = make_empty(1, a->size());

            out->prev_ = {a};
            out->op_ = "row_vect";

            out->forward_ = [=]() {
                out->data() = Eigen::Map<Eigen::MatrixXd>(a->data().data(), 1, a->size());
            };

            out->backward_ = [=]() {
                a->grad() = Eigen::Map<Eigen::MatrixXd>(out->grad().data(), a->rows(), a->cols());
            };

            out->forward_();
            return out;
        }

        // column_vectorize: to column vector
        friend ptr column_vectorize(const ptr& a) {
            auto out = make_empty(a->size(), 1);

            out->prev_ = {a};
            out->op_ = "col_vect";

            out->forward_ = [=]() {
                out->data() = Eigen::Map<Eigen::MatrixXd>(a->data().data(), a->size(), 1);
            };

            out->backward_ = [=]() {
                a->grad() = Eigen::Map<Eigen::MatrixXd>(out->grad().data(), a->rows(), a->cols());
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
                a->grad().row(ix) += out->grad();
            };

            out->forward_();
            return out;
        }

        friend ptr select_column(const ptr& a, int ix) {
            auto out = make_empty(a->rows(), 1);

            out->prev_ = {a};
            out->op_ = "col";

            out->forward_ = [=]() {
                out->data() = a->data().col(ix);
            };

            out->backward_ = [=]() {
                a->grad().col(ix) += out->grad();
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
                // Calculate the sum of each row
                Eigen::VectorXd rowSums = a->data().rowwise().sum();

                // Use broadcasting for normalization
                for (int i = 0; i < a->data().rows(); ++i) {
                    if (rowSums(i) != 0) { // Avoid division by zero
                        out->data().row(i) = a->data().row(i).array() / rowSums(i);
                    }
                }
            };

            out->backward_ = [=]() {
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

        friend ptr tanh(const ptr& a) {
            auto out = make_empty_copy(a);

            out->prev_ = {a};
            out->op_ = "tanh";

            out->forward_ = [out, a]() {
                out->data() = a->data().array().tanh();
            };

            out->backward_ = [out, a]() {
                Eigen::MatrixXd tanhOfA = out->data();
                Eigen::MatrixXd derivative = 1.0 - tanhOfA.array().square();
                a->grad() += (derivative.array() * out->grad().array()).matrix();
            };

            out->forward_();
            return out;
        }

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

    private:
        std::string label_{};
        std::vector<ptr> prev_{};
        std::string op_{""};

        std::function<void()> forward_{};
        std::function<void()> backward_{};
};

using Node = typename NodeValue::ptr;

template <typename... Args>
static Node make_node(Args&&... args) {
    return NodeValue::make(std::forward<Args>(args)...);
}

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
