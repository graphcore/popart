include(GNUInstallDirs)

# for more useful warnings
include(EnableCompilerWarnings)

option(POPLAR_BACKEND "Build Poplar backend (Graphcore IPU)" ON)
add_definitions(-std=c++11)
include_directories("include")


file(GLOB sources src/*cpp)

if (POPLAR_BACKEND)
message( STATUS "Building WITH Poplar IPU backend")
find_package(Poplar REQUIRED)
# concatenate poplar backend sources to the list of files to compile
file(GLOB pop_backend_sources src/gcipu/*cpp)
set(sources ${sources} ${pop_backend_sources})
else()
message( STATUS "Building WITHOUT Poplar IPU backend")
endif()

add_library(willow SHARED ${sources})
message(STATUS "${sources}")
message( STATUS "BOOST_FILESYSTEM_LIB to be used : " ${BOOST_FILESYSTEM_LIB} )
message( STATUS "BOOST_FILESYSTEM_INCLUDE_DIR to be used :  " ${BOOST_FILESYSTEM_INCLUDE_DIR} )

target_include_directories (willow PUBLIC 
    ${ONNX_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIR}
    ${BOOST_FILESYSTEM_INCLUDE_DIR}
    ${CBLAS_INCLUDE_DIRS}
    ${POPLAR_INCLUDE_DIR}
    )

target_link_libraries(willow
  onnx
  ${CBLAS_LIBRARIES}
  ${BOOST_FILESYSTEM_LIB}
  ${BOOST_SYSTEM_LIB}
  ${POPLAR_LIB}
)

install(TARGETS willow
        EXPORT willow
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT willow
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

