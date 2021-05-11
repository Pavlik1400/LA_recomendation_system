#ifndef FUNKSVD_DATA_LOADER_H
#define FUNKSVD_DATA_LOADER_H


#include "utils.h"
template<typename T, std::size_t N>
std::array<T, N> randomArray(std::normal_distribution<double> &dist, std::mt19937 &gen){
    std::array<T, N> array{};
    for(std::size_t i = 0; i < N; i++) {
        array[i] = dist(gen);
    }
    return array;
}
template<typename T, std::size_t N>
std::tuple<ratingVector<3>, hashMapWeight<T, N>, hashMapWeight<T, N>, hashMapBias<T>, hashMapBias<T>, T> setUp(std::stringstream &string, std::normal_distribution<double> &dist, std::mt19937 &gen){

    ratingVector<3> R;
    hashMapWeight<T, N> P;
    hashMapWeight<T, N> Q;
    hashMapBias<T> bu;
    hashMapBias<T> bi;
    std::string line;
    std::size_t product_id = 0;
    double mean = 0;
    while (std::getline(string, line)) {
        if (line.back() == ':') {
            product_id = std::stoul(line.substr(0, line.length() - 1));
            if (Q.find(product_id) != Q.end()) {
                bi.emplace(product_id, dist(gen));
                Q.emplace(product_id, randomArray<T, N>(dist, gen));
            }
            continue;
        }
        std::string s_user_id;
        std::string s_rating;

        std::stringstream ss(line);

        std::getline(ss, s_user_id, ',');
        std::getline(ss, s_rating, ',');

        auto user_id = std::stoul(s_user_id);
        auto rating = std::stoul(s_rating);
        if (P.find(user_id) != P.end()) {
            bu.emplace(user_id, dist(gen));
            P.emplace(user_id, randomArray<T, N>(dist, gen));
        }
        mean += rating;
        R.emplace_back(std::array<std::size_t, 3>{product_id, user_id, rating});
    }
    mean /= R.size();
    return {R, Q, P, bu, bi, mean};
}

void combineData(const std::vector<std::string> &paths, const std::string &output){

    std::ofstream ofile(output, std::ios::app);
    for(auto &file: paths){
        std::ifstream ifile(file);
        ofile << ifile.rdbuf();
        ifile.close();
    }
}
#endif //FUNKSVD_DATA_LOADER_H
