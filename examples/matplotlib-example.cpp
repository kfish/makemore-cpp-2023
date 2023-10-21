#include "matplotlib_main.h"

namespace plt = matplotlibcpp;

int main(int argc, char *argv[]) {
    std::vector<double> y = {1.0, 2.9, 3.8};

    plt::plot(y);

    return matplotlib_main(argc, argv);
}
