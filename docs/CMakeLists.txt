cmake_minimum_required(VERSION 3.4.2)

set(BUILD_DOCS OFF CACHE BOOL "Build documentation")

if (BUILD_DOCS)
  find_package(gcdoc REQUIRED)

  list(APPEND DOXYGEN_INPUT0 "")
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../willow/include)
  list(APPEND DOXYGEN_INPUT0 ${CMAKE_CURRENT_SOURCE_DIR}/../${INSTALL_PYDIR})

  string(REPLACE ";" " " DOXYGEN_INPUT "${DOXYGEN_INPUT0}")
  configure_file(user_guide/Doxyfile.in Doxyfile @ONLY)

  set(RELEASE "0.2.2")

  set(SUPPORTED_OPS_GEN ${CMAKE_CURRENT_BINARY_DIR}/supported_ops_gen.rst)

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

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pdf_user_guide
    COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/user_guide
    ${CMAKE_CURRENT_BINARY_DIR}/pdf_user_guide
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html_user_guide
    COMMAND cp -R ${CMAKE_CURRENT_SOURCE_DIR}/user_guide
    ${CMAKE_CURRENT_BINARY_DIR}/html_user_guide
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Note : To keep the temporay files add the '-save-temps' option to the GCDOC command

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/pdfgen
    COMMAND ${GCDOC} -v -x ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                     -r ${RELEASE}
		     -b pdf
         ${CMAKE_CURRENT_BINARY_DIR}/pdf_user_guide
         ${CMAKE_CURRENT_BINARY_DIR}/latex/popart.pdf
    COMMENT "Building PDF documentation from 'user_guide' directory"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/latex
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    ${CMAKE_CURRENT_BINARY_DIR}/pdf_user_guide
	    ${SUPPORTED_OPS_GEN}
	    popart_examples
            popart_core
  )

  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMAND ${GCDOC} -v -x ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                     -r ${RELEASE}
		     -b html
         ${CMAKE_CURRENT_BINARY_DIR}/html_user_guide
         ${CMAKE_CURRENT_BINARY_DIR}/html
    COMMENT "Building html documentation from 'user_guide' directory"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    ${CMAKE_CURRENT_BINARY_DIR}/html/doxygen/latex
    ${CMAKE_CURRENT_BINARY_DIR}/html_user_guide
	    ${SUPPORTED_OPS_GEN}
	    popart_examples
            popart_core
  )

  add_custom_target(html_zip ALL
    COMMAND zip -r ${PROJECT_BINARY_DIR}/${USER_GUIDE_HTML_NAME} . -i *
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
    VERBATIM
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/html
  )

  add_custom_target(pdf_copy ALL
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/latex/popart.pdf ${PROJECT_BINARY_DIR}/${USER_GUIDE_PDF_NAME}
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/pdfgen
  )

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION docs
          COMPONENT popart-docs)

        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/latex/popart.pdf DESTINATION docs/pdf
          COMPONENT popart-docs)

        install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html/doxygen/html
          DESTINATION docs/doxygen
          COMPONENT popart-docs)

else()
  install(FILES)
endif()
