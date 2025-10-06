cmake_minimum_required(VERSION 3.16)
project(openfhe_iris_project LANGUAGES CXX)

# =====================================================
# Compiler settings
# =====================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# =====================================================
# Locate OpenFHE
# =====================================================
# Adjust path if needed
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# =====================================================
# Source files
# =====================================================
set(SRC_FILES
    fhe_svm_iris_full.cpp
    fhe_svm_iris_multiclass.cpp
)

# =====================================================
# Executables
# =====================================================
add_executable(fhe_svm_iris_full fhe_svm_iris_full.cpp)
add_executable(fhe_svm_iris_multiclass fhe_svm_iris_multiclass.cpp)

# =====================================================
# Link OpenFHE libraries
# =====================================================
target_link_libraries(fhe_svm_iris_full
    PUBLIC
        OPENFHEcore
        OPENFHEpke
        OPENFHEbinfhe
)

target_link_libraries(fhe_svm_iris_multiclass
    PUBLIC
        OPENFHEcore
        OPENFHEpke
        OPENFHEbinfhe
)

# =====================================================
# Output directory
# =====================================================
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

message(STATUS "✅ Configuration complete.")
message(STATUS "You can now run:  cd build && make -j$(nproc)")
