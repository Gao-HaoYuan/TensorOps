#include "mini_gtest.h"

using namespace minitest;

int main(int argc, char **argv) {
    std::string filter;
    if (argc > 1 && std::string(argv[1]) == "--filter" && argc > 2) {
        filter = argv[2];
    }

    run_all(filter);
    
    return 0;
}