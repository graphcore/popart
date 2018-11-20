
add_subdirectory(poponnx)
add_subdirectory(torch)

install(DIRECTORY .
	DESTINATION ${INSTALL_TESTS}
	FILES_MATCHING REGEX ".*\.(py|sh|md)$")
