cmake_minimum_required(VERSION 3.19)
project(FunkSVD)

set(CMAKE_CXX_STANDARD 17)


find_package(LibArchive REQUIRED)
find_package(ICU REQUIRED COMPONENTS uc i18n)
include_directories(${Boost_INCLUDE_DIRS} ${LibArchive_INCLUDE_DIRS})
set (Boost_USE_STATIC_LIBS OFF)
set (Boost_USE_MULTITHREAD ON)
find_package(Boost REQUIRED COMPONENTS
        filesystem locale system)



set(SOURCES main.cpp data_loader.h utils.h)
add_executable(${PROJECT_NAME} ${SOURCES})
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME} TBB::tbb Threads::Threads ${Boost_LIBRARIES} ${ICU_LIBRARIES} ${LibArchive_LIBRARIES})
target_compile_options(${PROJECT_NAME} PRIVATE ${TBB_CXX_STD_FLAG})