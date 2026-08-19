add_executable(test_weights test_weights.cpp)
target_link_libraries(test_weights PRIVATE nanoinfer_core)

add_test(NAME weights_roundtrip COMMAND test_weights)
