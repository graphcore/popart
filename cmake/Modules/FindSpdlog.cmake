set(SPDLOG_HINT_PATH ${CMAKE_INSTALL_PREFIX}/../spdlog/include)
if(Spdlog_FOUND)
 return()
endif()

find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h HINT ${SPDLOG_HINT_PATH})
set(SPDLOG_INCLUDE_DIRS ${SPDLOG_INCLUDE_DIR} )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Spdlog DEFAULT_MSG SPDLOG_INCLUDE_DIR)
