function(add_coverage_flags_if_enabled target)
  if (${POPART_ENABLE_COVERAGE} AND (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU|Clang|AppleClang"))
    # The same set of flags is applicable for both g++ and clang, however 
    # parsing the coverage output from g++ MUST be parsed by `gcov`, while
    # the output from clang++ MUST be parsed by `llvm-cov gcov`.
    target_compile_options(${target} PRIVATE --coverage)
    target_link_options(${target} PRIVATE --coverage)
  endif()
endfunction()
