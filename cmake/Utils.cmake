function(add_coverage_flags_if_enabled target)
  if (${POPART_ENABLE_COVERAGE} AND (${CMAKE_CXX_COMPILER_ID} MATCHES "GNU|Clang|AppleClang"))
    # The same set of flags is applicable for both g++ and clang, however 
    # parsing the coverage output from g++ MUST be parsed by `gcov`, while
    # the output from clang++ MUST be parsed by `llvm-cov gcov`.
    target_compile_options(${target} PRIVATE --coverage)
    target_link_options(${target} PRIVATE --coverage)
  endif()
endfunction()

function(get_include_directories_from_target target out)
  set(dirs "")

  get_target_property(tmp ${target} INCLUDE_DIRECTORIES)
  list(APPEND dirs ${tmp})

  get_target_property(popart-link-libraries ${target} LINK_LIBRARIES)
  foreach(lib IN LISTS popart-link-libraries)
    if(TARGET ${lib})
      get_target_property(tmp ${lib} INTERFACE_INCLUDE_DIRECTORIES)
      if(tmp)
        list(APPEND dirs ${tmp})
      endif()
      get_target_property(tmp ${lib} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
      if(tmp)
        list(APPEND dirs ${tmp})
      endif()
    endif()
  endforeach()

  set(${out}
      "${dirs}"
      PARENT_SCOPE)
endfunction()
