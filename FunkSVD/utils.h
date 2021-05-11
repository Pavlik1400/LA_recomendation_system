
#ifndef FUNKSVD_UTILS_H
#define FUNKSVD_UTILS_H

#include <functional>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <unordered_map>

//#define SINGEL
#ifndef SINGEL
template<typename T, std::size_t N>
using hashMapWeight = tbb::concurrent_unordered_map<std::size_t, std::array<T, N>>;
template<typename T>
using hashMapBias = tbb::concurrent_unordered_map<std::size_t, T>;
#else
template<typename T, std::size_t N>
using hashMapWeight = std::unordered_map<std::size_t, std::array<T, N>>;
template<typename T>
using hashMapBias = std::unordered_map<std::size_t, T>;
#endif
template<std::size_t N>
using ratingVector = std::vector<std::array<std::size_t, N>>;

typedef struct {
    double lr_all, mean;
    std::size_t epochs;
    std::size_t start, end, step;
} hyperParams;

template<typename T, std::size_t N>
inline T dot_product(std::array<T, N> &vector1, std::array<T, N> &vector2){
    T accum = 0;
    for(std::size_t i = 0; i < N; i++){
        accum = std::fma(vector1[i], vector2[i], accum);
    }
    return accum;
}


template<typename T, std::size_t N>
void fit(const ratingVector<3> &R, hashMapWeight<T, N> &Q, hashMapWeight<T, N> &P, hashMapBias<T> &bu, hashMapBias<T> &bi, const hyperParams &params){
    for(std::size_t j = 0; j < params.epochs; j++) {
        tbb::parallel_for(params.start, params.end, params.step, [&](std::size_t it){
            auto product_id = R[it][0];
            auto user_id = R[it][1];
            auto rating = R[it][2];

            auto q_i = Q[product_id];
            auto p_u = P[user_id];
            auto error = (rating - dot_product(q_i, p_u) - params.mean - bu[user_id] - bi[product_id]) * params.lr_all;
            T prevPu;
            for (std::size_t i = 0; i < N; i++) {
                prevPu = p_u[i];
                p_u[i] += error * q_i[i];
                q_i[i] += error * prevPu;
            }
            bu[user_id] += error;
            bi[product_id] += error;
            Q[product_id] = q_i;
            P[user_id] = p_u;
        });
    }
}


template<typename T, std::size_t N>
T calcRMSE(const ratingVector<3> &R, hashMapWeight<T, N> &Q, hashMapWeight<T, N> &P, hashMapBias<T> &bu, hashMapBias<T> &bi, const hyperParams &params){
    T accum = 0;
    std::size_t count = 0;
    for(std::size_t it = params.start; it < params.end; it += params.step){
        auto error = R[it][2] - dot_product(Q[R[it][0]], P[R[it][1]]) - params.mean - bu[R[it][1]] - bi[R[it][0]];
        accum += error * error;
        count++;
    }
    accum /= count;
    return std::sqrt(accum);
}


#endif //FUNKSVD_UTILS_H
