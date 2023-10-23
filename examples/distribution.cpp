#include <iostream>
#include <random>
#include <Eigen/Dense>

int main(int argc, char *argv[]) {
    std::random_device rd;
    std::mt19937 generator(rd());

    //std::vector<double> probs = {0.6064, 0.3033, 0.0903};
    //std::discrete_distribution<int> dist(probs.begin(), probs.end());

    Eigen::MatrixXd prob_matrix(1, 3);
    prob_matrix << 0.6064, 0.3033, 0.0903;
    std::discrete_distribution<int> dist(prob_matrix.row(0).data(), prob_matrix.row(0).data() + prob_matrix.cols());

    for (int i=0; i<50; ++i) {
        std::cout << dist(generator) << '\t';
    }
    std::cout << std::endl;
}
