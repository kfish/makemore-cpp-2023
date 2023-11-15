#pragma once

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>

namespace ai {

// Define the global format
//Eigen::IOFormat PrettyMatrixFormat(4, 0, ", ", "\n", "", "", "[", "]");

class PrettyMatrix {
public:
    // Constructor that takes a reference to an Eigen matrix
    PrettyMatrix(const Eigen::MatrixXd& matrix) : matrix_(matrix) {}

    // Friend function to overload operator<<
    friend std::ostream& operator<<(std::ostream& os, const PrettyMatrix& wrapper) {
        os << std::fixed;
        os << "(" << wrapper.matrix_.rows() << "," << wrapper.matrix_.cols() << "):[\n";
        for (int i = 0; i < wrapper.matrix_.rows(); ++i) {
            for (int j = 0; j < wrapper.matrix_.cols(); ++j) {
                os << std::setw(8) << std::setprecision(6) << wrapper.matrix_(i, j);
                if (j < wrapper.matrix_.cols() - 1) os << ", ";
            }
            if (i < wrapper.matrix_.rows() - 1) os << "\n";
        }
        os << "]\n";
        os << std::defaultfloat;
        return os;
    }

private:
    const Eigen::MatrixXd& matrix_; // Reference to an Eigen matrix
};

} // namespace ai
