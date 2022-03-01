# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
cmake_minimum_required(VERSION 3.4.2)

set(BUILD_DOCS OFF CACHE BOOL "Build documentation")

set(popart_docs__find_poplar_enable_script_failure_str "\
Error building PopART docs: Cannot find Poplar's enable.sh script. Try setting \
POPLAR_INSTALL_DIR. If you are building the poplar view, this is a bug."
CACHE INTERNAL "")

function(find_poplar_enable_script result_var)
  if(DEFINED POPLAR_INSTALL_DIR AND EXISTS ${POPLAR_INSTALL_DIR})
    # If building in popart view or out-of-view.
    set(${result_var} "${POPLAR_INSTALL_DIR}/enable.sh" PARENT_SCOPE)

  elseif(EXISTS "${CMAKE_BINARY_DIR}/../../activate.sh" AND
      EXISTS "${CMAKE_BINARY_DIR}/../../install/poplar")
    # Assume building in poplar view. There is no enable.sh, so we activate the
    # buildtree instead.
    set(${result_var} "${CMAKE_BINARY_DIR}/../../activate.sh" PARENT_SCOPE)

  else()
    # Error, don't know where else to look.
    message(FATAL_ERROR ${popart_docs__find_poplar_enable_script_failure_str})
  endif()
endfunction()

if (BUILD_DOCS)
  # TODO(T54155): Replace with find_package once sphinx_resources supports it
  set(sphinx_resources_dir "${CMAKE_SOURCE_DIR}/../sphinx_resources/common")
  if(NOT EXISTS "${sphinx_resources_dir}")
    message(FATAL_ERROR "PopART cannot build docs, ${sphinx_resources_dir} not found")
  else()
    message("PopART found ${sphinx_resources_dir}")
  endif()

  find_program(DOXYGEN doxygen)
  if(NOT DOXYGEN)
    message(FATAL_ERROR "PopART cannot build docs, doxygen not found")
  else()
    message("PopART found ${DOXYGEN}")
  endif()

  find_program(VIRTUALENV virtualenv)
  if(NOT VIRTUALENV)
    message(FATAL_ERROR "PopART cannot build docs, virtualenv not found")
  else()
    message("PopART found ${VIRTUALENV}")
  endif()

  # Create gen_docs.sh
  find_poplar_enable_script(POPLAR_ENABLE_SCRIPT)
  configure_file(./gen_docs.sh.in gen_docs.sh @ONLY)
  configure_file(./active_buildir.sh.in active_buildir.sh @ONLY)

  # Create a virtual environment so we can run sphinx-build. We used to do this
  # inside of gen_docs.sh but it is better to do this once here so that every
  # gen_docs.sh call can make use of this environment.
  set(venv "${CMAKE_CURRENT_BINARY_DIR}/venv")
  set(venv_activate "${venv}/bin/activate")
  set(venv_requirements "${sphinx_resources_dir}/requirements.txt")

  add_custom_command(
    OUTPUT ${venv}
    COMMAND ${VIRTUALENV} -p python3 ${venv}
    DEPENDS ${CMAKE_BINARY_DIR}/docs
  )

  add_custom_command(
    OUTPUT venv.stamp
    COMMAND ${CMAKE_COMMAND} -E copy ${venv_requirements} requirements.txt
    COMMAND . ${venv_activate} && pip install -r requirements.txt
    COMMAND . ${venv_activate} && pip install numpy
    COMMAND . ${venv_activate} && pip install typing_extensions
    DEPENDS ${venv} ${venv_requirements}
  )

  add_custom_target(SphinxRequirements ALL
    DEPENDS venv.stamp
    SOURCES ${venv_requirements}
  )

  # Creating folders.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND cmake -E make_directory ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Creating build folder for 'docs'"
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/latex
    COMMAND cmake -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/latex
    COMMENT "Creating build folder for 'docs/latex'"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/doxygen/latex
    COMMAND cmake -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/html/doxygen/latex
    COMMENT "Creating build folder for 'docs/html/doxygen/latex'"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Generate supported PopART ops rst doc.
  set(SUPPORTED_POPART_OPS_GEN ${CMAKE_CURRENT_BINARY_DIR}/popart_supported_ops_gen.rst)
  add_custom_command(
    OUTPUT ${SUPPORTED_POPART_OPS_GEN}
    COMMAND
        bash ${CMAKE_CURRENT_BINARY_DIR}/active_buildir.sh
        python3 gen_popart_supported_ops.py ${SUPPORTED_POPART_OPS_GEN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating PopART supported ops in ${SUPPORTED_POPART_OPS_GEN}"
    DEPENDS popart-py
            ${CMAKE_CURRENT_BINARY_DIR}/active_buildir.sh
  )

  # Generate supported PopXL ops rst doc.
  set(SUPPORTED_POPXL_OPS_GEN ${CMAKE_CURRENT_BINARY_DIR}/popxl_supported_ops_gen.rst)
  add_custom_command(
    OUTPUT ${SUPPORTED_POPXL_OPS_GEN}
    COMMAND
        bash ${CMAKE_CURRENT_BINARY_DIR}/active_buildir.sh
        python3 gen_popxl_supported_ops.py ${SUPPORTED_POPXL_OPS_GEN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating PopXL supported ops in ${SUPPORTED_POPXL_OPS_GEN}"
    DEPENDS popart-py
            ${CMAKE_CURRENT_BINARY_DIR}/active_buildir.sh
  )

  # For a given document directory DOC_SOURCE_DIR, get the source files,
  # returned in DOC_SOURCE_FILES.
  function(get_doc_files DOC_SOURCE_DIR DOC_SOURCE_FILES)
    file(GLOB GLOB_RESULT CONFIGURE_DEPENDS
      # This assumes all .rst files are in the root dir, and other files are
      # in directories, which is true at time of writing but may need revisiting
      # is the document structure changes.
      ${DOC_SOURCE_DIR}/*.rst
      ${DOC_SOURCE_DIR}/**/*.png
      ${DOC_SOURCE_DIR}/**/*.py
      ${DOC_SOURCE_DIR}/**/*.cpp
      ${DOC_SOURCE_DIR}/**/*.in
    )
    # Set the result value.
    set(${DOC_SOURCE_FILES} ${GLOB_RESULT} PARENT_SCOPE)
  endfunction()

  set(USER_GUIDE_SOURCE_FILES "")
  get_doc_files(${CMAKE_CURRENT_SOURCE_DIR}/user_guide USER_GUIDE_SOURCE_FILES)

  # Log the document sources.
  message("USER_GUIDE_SOURCE_FILES: ${USER_GUIDE_SOURCE_FILES}")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/user_guide
    COMMAND cmake -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/user_guide
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying user-guide sources to ${CMAKE_CURRENT_BINARY_DIR}/user_guide"
    DEPENDS ${SUPPORTED_POPART_OPS_GEN}
            ${USER_GUIDE_SOURCE_FILES}
  )

  set(POPXL_SOURCE_FILES "")
  get_doc_files(${CMAKE_CURRENT_SOURCE_DIR}/popxl POPXL_SOURCE_FILES)

  # Log the document sources.
  message("POPXL_SOURCE_FILES: ${POPXL_SOURCE_FILES}")

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/popxl
    COMMAND cmake -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/popxl ${CMAKE_CURRENT_BINARY_DIR}/popxl
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying PopXL sources to ${CMAKE_CURRENT_BINARY_DIR}/popxl"
    DEPENDS ${SUPPORTED_POPXL_OPS_GEN}
            ${POPXL_SOURCE_FILES}
  )

  # Create Doxyfile file.
  list(APPEND DOXYGEN_INPUT0 "")
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include)
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include/op)
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include/popx)

  string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT0}")
  configure_file(api_cpp/Doxyfile.in Doxyfile @ONLY)

  # Run doxygen.
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/doxygen
    COMMAND BUILDDIR=${CMAKE_CURRENT_BINARY_DIR} ${DOXYGEN} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    COMMENT "Run doxygen"
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
            ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  # Helper function to build html and pdf docs.
  function(build_popart_doc DOC_SOURCE_DIR DOC_TARGET DOC_TITLE DOC_VERSION)
    # Use sphinx-build to build a documentation item.
    set(DOC_TARGET_HTML "${CMAKE_CURRENT_BINARY_DIR}/html/${DOC_TARGET}")
    set(DOC_TARGET_PDF "${CMAKE_CURRENT_BINARY_DIR}/latex/${DOC_TARGET}.pdf")

    # These dependencies are evaluated at build time. Adding these files as
    # dependencies will ensure the docs are rebuilt when they change.
    set(DOC_SOURCE_FILES "")
    get_doc_files(${DOC_SOURCE_DIR}/ DOC_SOURCE_FILES)

    # Log the document sources.
    message("DOC_SOURCE_FILES for ${DOC_TITLE}: ${DOC_SOURCE_FILES}")

    add_custom_command(
      OUTPUT ${DOC_TARGET_HTML}
             ${DOC_TARGET_PDF}
      COMMAND . ${venv_activate} &&
                bash ${CMAKE_CURRENT_BINARY_DIR}/active_buildir.sh
                bash ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh
                  "${DOC_SOURCE_DIR}"
                  "${DOC_TARGET_HTML}"
                  "${DOC_TARGET_PDF}"
                  "${DOC_TITLE}"
                  "${DOC_VERSION}"
      COMMENT "Building ${DOC_TITLE} (html/pdf)"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS SphinxRequirements
              ${SUPPORTED_POPART_OPS_GEN}
              ${SUPPORTED_POPXL_OPS_GEN}
              ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh
              ${CMAKE_CURRENT_BINARY_DIR}/active_buildir.sh
              ${PROJECT_BINARY_DIR}/enable.sh
              ${DOC_SOURCE_DIR}
              ${DOC_SOURCE_FILES}
              ${CMAKE_CURRENT_BINARY_DIR}/doxygen
              # Adding popart-py as a dependency will ensure docs are rebuilt
              # when PopART changes.
              popart-py
      VERBATIM)
  endfunction()

  build_popart_doc("${CMAKE_CURRENT_BINARY_DIR}/user_guide" "popart_user_guide" "PopART User Guide" "${DOCS_VERSION}")
  build_popart_doc("${CMAKE_CURRENT_SOURCE_DIR}/api_cpp" "popart_api_cpp" "PopART C++ API" "${DOCS_VERSION}")
  build_popart_doc("${CMAKE_CURRENT_SOURCE_DIR}/api_python" "popart_api_python" "PopART Python API" "${DOCS_VERSION}")
  build_popart_doc("${CMAKE_CURRENT_BINARY_DIR}/popxl" "popxl_user_guide" "PopXL User Guide and API" "${DOCS_VERSION}")

  add_custom_target(ug_zip ALL
    COMMAND zip -r ${PROJECT_BINARY_DIR}/${USER_GUIDE_HTML_NAME} . -i *
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/popart_user_guide
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/popart_user_guide
  )

  add_custom_target(python_zip ALL
    COMMAND zip -r ${PROJECT_BINARY_DIR}/${PYTHON_API_HTML_NAME} . -i *
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_python
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_python
  )

  add_custom_target(cpp_zip ALL
    COMMAND zip -r ${PROJECT_BINARY_DIR}/${CPP_API_HTML_NAME} . -i *
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_cpp
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_cpp
  )

  add_custom_target(ir_zip ALL
    COMMAND zip -r ${PROJECT_BINARY_DIR}/${POPXL_USER_GUIDE_HTML_NAME} . -i *
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/popxl_user_guide
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html/popxl_user_guide
  )

  # Make a copy of the support_ops doc to be uploaded to Artifactory so
  # it can be used to builed the externally published docs. This does not
  # need to be installed as part of the SDK package.
  add_custom_target(supported_popart_ops_copy ALL
    COMMAND cp ${SUPPORTED_POPART_OPS_GEN} ${PROJECT_BINARY_DIR}/${SUPPORTED_POPART_OPS_GEN_FILE_NAME}
    DEPENDS ${SUPPORTED_POPART_OPS_GEN}
  )
  add_custom_target(supported_popxl_ops_copy ALL
    COMMAND cp ${SUPPORTED_POPXL_OPS_GEN} ${PROJECT_BINARY_DIR}/${SUPPORTED_POPXL_OPS_GEN_FILE_NAME}
    DEPENDS ${SUPPORTED_POPXL_OPS_GEN}
  )

  add_custom_target(pdf_copy ALL
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf ${PROJECT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf ${PROJECT_BINARY_DIR}/${CPP_API_PDF_NAME}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf ${PROJECT_BINARY_DIR}/${PYTHON_API_PDF_NAME}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popxl_user_guide.pdf ${PROJECT_BINARY_DIR}/${POPXL_USER_GUIDE_PDF_NAME}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf
            ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf
            ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf
            ${CMAKE_CURRENT_BINARY_DIR}/latex/popxl_user_guide.pdf
  )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION docs
            COMPONENT popart-docs)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popxl_user_guide.pdf
    DESTINATION docs/pdf
    COMPONENT popart-docs)

else()
  install(FILES)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/user_guide/files
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/user_guide)

set(PYTEST_BASETEMP ${CMAKE_CURRENT_BINARY_DIR}/pytest_tmp)
file(MAKE_DIRECTORY ${PYTEST_BASETEMP})

add_test(NAME popart_examples_test COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_BINARY_DIR}/user_guide/files/tests/test_examples.py --basetemp=${PYTEST_BASETEMP}/popart_examples_test)
