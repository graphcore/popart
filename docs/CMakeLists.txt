cmake_minimum_required(VERSION 3.4.2)

set(BUILD_DOCS OFF CACHE BOOL "Build documentation")

if (BUILD_DOCS)
  find_package(gcdoc REQUIRED)

  list(APPEND DOXYGEN_INPUT0 "")
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include)

  string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT0}")
  configure_file(api_cpp/Doxyfile.in Doxyfile @ONLY)

  configure_file(./gen_docs.sh.in gen_docs.sh @ONLY)

  set(SUPPORTED_OPS_GEN ${CMAKE_CURRENT_SOURCE_DIR}/supported_ops_gen.rst)

  message(STATUS "Popart_core lib : " $<TARGET_FILE:popart_core>)
  message(STATUS "Python executable : " ${Python3_EXECUTABLE})

  add_custom_command(
    OUTPUT ${SUPPORTED_OPS_GEN}
    COMMAND python3 gen_supported_ops.py "${PROJECT_BINARY_DIR}/python:${PROJECT_BINARY_DIR}/popart/python" ${SUPPORTED_OPS_GEN}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Creating supported ops"
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
    COMMAND ${GCDOC} -v -r ${VERSION} -b pdf ${CMAKE_CURRENT_SOURCE_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_user_guide.pdf
    COMMAND ${GCDOC} -v -x ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile -r ${VERSION} -b pdf ${CMAKE_CURRENT_SOURCE_DIR}/api_cpp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_cpp.pdf
    COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh ${GCDOC} -v -r ${VERSION} -b pdf ${CMAKE_CURRENT_SOURCE_DIR}/api_python ${CMAKE_CURRENT_BINARY_DIR}/latex/popart_api_python.pdf
    COMMENT "Building PDF documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh
    ${PROJECT_BINARY_DIR}/enable.sh
    ${CMAKE_CURRENT_BINARY_DIR}/latex
    ${SUPPORTED_OPS_GEN}
    popart_examples
    popart_core
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/popart_user_guide
    COMMAND ${GCDOC} -v -r ${VERSION} -b singlehtml ${CMAKE_CURRENT_SOURCE_DIR}/user_guide ${CMAKE_CURRENT_BINARY_DIR}/html/popart_user_guide
    COMMENT "Building html user guide documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${SUPPORTED_OPS_GEN}
    popart_examples
    popart_core
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_cpp
    COMMAND ${GCDOC} -v -x ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile -r ${VERSION} -b singlehtml ${CMAKE_CURRENT_SOURCE_DIR}/api_cpp ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_cpp
    COMMENT "Building html C++ API documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    ${SUPPORTED_OPS_GEN}
    popart_examples
    popart_core
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_python
    COMMAND bash ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh ${GCDOC} -v -r ${VERSION} -b singlehtml ${CMAKE_CURRENT_SOURCE_DIR}/api_python ${CMAKE_CURRENT_BINARY_DIR}/html/popart_api_python
    COMMENT "Building html Python API documentation"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/gen_docs.sh
    ${PROJECT_BINARY_DIR}/enable.sh
    ${SUPPORTED_OPS_GEN}
    popart_examples
    popart_core
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

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/
    DESTINATION docs/popart_api_cpp/doxygen
    COMPONENT popart-docs)

else()
  install(FILES)
endif()
