#include <iostream>
#include <fstream>
#include <cstring>
#include <random>
#include <sstream>
#include <thread>

#include "data_loader.h"
#include "utils.h"
#include "time_measure.h"

std::string FILENAME = "combined_data_1.txt";

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    std::ifstream file(FILENAME);
    if (!file) {
        exit(1);
    }
    const std::size_t N = 10;
    std::stringstream string;
    auto b = get_current_time_fenced();
    string << file.rdbuf();

    auto [R, Q, P, bu, bi, mean] = setUp<double, N>(string, dist, gen);
    std::shuffle ( R.begin(), R.end(), std::mt19937(std::random_device()()));

    auto c = get_current_time_fenced();
    std::cout << "Start training" << std::endl;
    hyperParams params{0.01, mean, 10, 0, static_cast<size_t>(R.size() * 0.8), 1};
    fit(R, Q, P, bu, bi, params);
    auto d = get_current_time_fenced();

    std::cout << "Start testing" << std::endl;
    params.start = static_cast<size_t>(R.size() * 0.8);
    params.end = R.size();
    std::cout << "RMSE: " << calcRMSE(R, Q, P, bu, bi, params) << std::endl;
    auto e = get_current_time_fenced();

    std::cout << "SetUp: " << to_ms(c - b)/1e6 << std::endl;
    std::cout << "Train: " << to_ms(d - c)/1e6 << std::endl;
    std::cout << "Test: " << to_ms(e - d)/1e6 << std::endl;

    return 0;
}
