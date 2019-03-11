add_subdirectory(poponnx)
add_subdirectory(torch)
add_subdirectory(onnx_backend)

install(DIRECTORY .
	DESTINATION ${INSTALL_TESTS}
	FILES_MATCHING REGEX ".*\.(py|sh|md)$")


