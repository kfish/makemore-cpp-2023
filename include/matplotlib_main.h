#pragma once

#include <iostream>
#include <unistd.h> // for getopt

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int matplotlib_main(int argc, char *argv[]) {
    int opt;
    std::string output_file;

    while ((opt = getopt(argc, argv, "o:")) != -1) {
        switch (opt) {
        case 'o':
            output_file = std::string(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " [-o output_file]" << std::endl;
            return 1;
        }
    }

    if (!output_file.empty()) {
        plt::save(output_file);
    } else {
        plt::show();
    }

    return 0;
}
