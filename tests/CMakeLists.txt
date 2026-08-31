# Shared test support: header search path for npy.hpp and reference.hpp, plus the
# absolute path to the reference data so tests work from any build directory.
add_library(nanoinfer_test_support INTERFACE)
target_include_directories(nanoinfer_test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(nanoinfer_test_support INTERFACE
  NANOINFER_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(nanoinfer_test_support INTERFACE nanoinfer_core)

function(nanoinfer_add_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE nanoinfer_test_support)
  add_test(NAME ${name} COMMAND ${name})
  # 77 means the reference dump is missing. ctest reports a skip, not a failure.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

nanoinfer_add_test(test_weights test_weights.cpp)
nanoinfer_add_test(test_layernorm test_layernorm.cu)
nanoinfer_add_test(test_gelu test_gelu.cu)
nanoinfer_add_test(test_embedding test_embedding.cu)
nanoinfer_add_test(test_residual test_residual.cu)
