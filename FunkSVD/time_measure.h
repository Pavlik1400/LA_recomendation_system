#ifndef LAB1_TIME_MEASURE_H
#define LAB1_TIME_MEASURE_H

#include <atomic>
#include <chrono>

#ifdef WIN32
typedef std::chrono::steady_clock clock_type;
#else
typedef std::chrono::high_resolution_clock clock_type;
#endif

inline clock_type::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = clock_type::now();
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

template<class D>
inline long long to_ms(const D &d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

#endif //LAB1_TIME_MEASURE_H
