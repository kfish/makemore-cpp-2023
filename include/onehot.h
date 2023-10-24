#include <Eigen/Dense>

static inline Eigen::VectorXd OneHot(int length, int hot_index) {
    Eigen::VectorXd vec = Eigen::VectorXd::Zero(length);
    if(hot_index < length && hot_index >= 0) {
        vec(hot_index) = 1.0;
    } else {
        throw std::out_of_range("Hot index is out of range");
    }
    return vec;
}
