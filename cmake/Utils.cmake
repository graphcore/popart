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
