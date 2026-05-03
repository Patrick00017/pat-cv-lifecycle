#
# image_classification

set(TEST_TARGET image_classification)
add_executable(${TEST_TARGET} image_classification.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)