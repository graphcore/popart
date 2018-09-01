include(GNUInstallDirs)

# for more useful warnings
include(EnableCompilerWarnings)

# so that installed bins don't have link probs (copied from poplar)
if (APPLE)
  set(CMAKE_INSTALL_RPATH
      "@loader_path/../${CMAKE_INSTALL_LIBDIR};@loader_path/../../tbb/lib"
      CACHE STRING "Relative RPATH for OSX." FORCE)
else(UNIX)
  set(CMAKE_INSTALL_RPATH
      "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR};\$ORIGIN/../../tbb/lib" CACHE STRING
      "Relative RPATH for Unix systems." FORCE)
endif()

add_definitions(-std=c++11)
include_directories("include")
file(GLOB_RECURSE source_files src/*cpp)
add_library(neuralnet SHARED ${source_files})

MESSAGE( STATUS "BOOST_FILESYSTEM_LIB to be used : " ${BOOST_FILESYSTEM_LIB} )
MESSAGE( STATUS "BOOST_FILESYSTEM_INCLUDE_DIR to be used :  " ${BOOST_FILESYSTEM_INCLUDE_DIR} )

target_include_directories (neuralnet PUBLIC 
    # if the model checker is used: ${ONNX_UTILS_INCLUDE_DIR}
    ${ONNX_PB_INCLUDE_DIR}
    ${ONNX_CHECKER_INCLUDE_DIR}
    ${Protobuf_INCLUDE_DIR}
    ${BOOST_FILESYSTEM_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIRS}
    )

target_link_libraries(neuralnet
  # if the onnx mode checker is incorporated, this will be needed: ${ONNX_LIB}
  ${ONNX_LIB}
  ${CBLAS_LIBRARIES}
  ${ONNX_PROTO_LIB}
  ${Protobuf_LIBRARIES} 
  ${BOOST_FILESYSTEM_LIB}
  # filesystem (which is used) requires system
  ${BOOST_SYSTEM_LIB}
)

install(TARGETS neuralnet
        EXPORT neuralnet
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT neuralnet
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

