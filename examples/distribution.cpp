#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
    std::random_device rd;
    std::mt19937 generator(rd());

    std::vector<double> probs = {0.6064, 0.3033, 0.0903};
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    for (int i=0; i<50; ++i) {
        std::cout << dist(generator) << '\t';
    }
    std::cout << std::endl;
}
