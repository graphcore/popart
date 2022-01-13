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
  find_package(gcdoc REQUIRED)

  list(APPEND DOXYGEN_INPUT0 "")
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include)
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include/op)
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include/popx)

  string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT0}")
  configure_file(api_cpp/Doxyfile.in Doxyfile @ONLY)

  find_poplar_enable_script(POPLAR_ENABLE_SCRIPT)
  configure_file(./gen_docs.sh.in gen_docs.sh @ONLY)

  set(SUPPORTED_OPS_GEN ${CMAKE_CURRENT_BINARY_DIR}/supported_ops_gen.rst)

  list(APPEND SUPPORTED_OPS_PYTHONPATH ${PROJECT_BINARY_DIR}/python/popart)
  list(APPEND SUPPORTED_OPS_PYTHONPATH ${PROJECT_BINARY_DIR}/python/popart/popart_core/)
  list(APPEND SUPPORTED_OPS_PYTHONPATH ${PROJECT_BINARY_DIR}/python/popart._internal.ir/)
  foreach(opset IN LISTS POPART_OPSETS)
    list(APPEND SUPPORTED_OPS_PYTHONPATH ${PROJECT_BINARY_DIR}/python/popart/${opset}/)
  endforeach()
  string(REPLACE ";" ":" SUPPORTED_OPS_PYTHONPATH "${SUPPORTED_OPS_PYTHONPATH}")
  add_custom_command(
    OUTPUT ${SUPPORTED_OPS_GEN}
    COMMAND python3 gen_supported_ops.py ${SUPPORTED_OPS_PYTHONPATH} ${SUPPORTED_OPS_GEN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating supported ops in ${SUPPORTED_OPS_GEN}"
    DEPENDS popart-py
  )
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/user_guide
    COMMAND cmake -E copy_directory user_guide ${CMAKE_CURRENT_BINARY_DIR}/user_guide
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Copying user-guide sources to ${CMAKE_CURRENT_BINARY_DIR}/user_guide"
    DEPENDS ${SUPPORTED_OPS_GEN}
  )

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

  # Note : To keep the temporary files add the '-save-temps' option to the GCDOC command

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pdfgen
    COMMAND ${GCDOC} -v -r ${DOCS_VERSION} --onesided -b pdf ${CMAKE_CURRENT_BINARY_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf
    COMMAND ${GCDOC} -v -x ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile -r ${DOCS_VERSION} --onesided -b pdf ${CMAKE_CURRENT_SOURCE_DIR}/api_cpp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf
    COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh ${GCDOC} -v -r ${DOCS_VERSION} --onesided -b pdf ${CMAKE_CURRENT_SOURCE_DIR}/api_python ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf
    COMMENT "Building PDF documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh
    ${PROJECT_BINARY_DIR}/enable.sh
    ${CMAKE_CURRENT_BINARY_DIR}/latex
    ${SUPPORTED_OPS_GEN}
    ${CMAKE_CURRENT_BINARY_DIR}/user_guide
    popart-py
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/popart_user_guide
    COMMAND ${GCDOC} -v -r ${DOCS_VERSION} -b html ${CMAKE_CURRENT_BINARY_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/html/popart_user_guide
    COMMENT "Building html user guide documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${SUPPORTED_OPS_GEN}
    ${CMAKE_CURRENT_BINARY_DIR}/user_guide
    popart-py
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_cpp
    COMMAND ${GCDOC} -v -x ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile -r ${DOCS_VERSION} -b html ${CMAKE_CURRENT_SOURCE_DIR}/api_cpp ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_cpp
    COMMENT "Building html C++ API documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    popart-py
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_python
    COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh ${GCDOC} -v -r ${DOCS_VERSION} -b html ${CMAKE_CURRENT_SOURCE_DIR}/api_python ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_python
    COMMENT "Building html Python API documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh
    ${PROJECT_BINARY_DIR}/enable.sh
    popart-py
  )

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

  # Make a copy of the support_ops doc to be uploaded to Artifactory so
  # it can be used to builed the externally published docs. This does not
  # need to be installed as part of the SDK package.
  add_custom_target(supported_ops_copy ALL
    COMMAND cp ${SUPPORTED_OPS_GEN} ${PROJECT_BINARY_DIR}/${SUPPORTED_OPS_GEN_FILE_NAME}
    DEPENDS ${SUPPORTED_OPS_GEN}
  )

  add_custom_target(pdf_copy ALL
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf ${PROJECT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf ${PROJECT_BINARY_DIR}/${CPP_API_PDF_NAME}
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf ${PROJECT_BINARY_DIR}/${PYTHON_API_PDF_NAME}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/pdfgen
  )

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION docs
            COMPONENT popart-docs)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf
    ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf
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
