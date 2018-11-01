include(CPack)

set(CPACK_SOURCE_IGNORE_FILES ${INSTALL_TESTS})

add_test(NAME loader_test WORKING_DIRECTORY ${INSTALL_TESTS} COMMAND python basic/loader_test.py)

install(DIRECTORY .
	DESTINATION ${INSTALL_TESTS}
	FILES_MATCHING PATTERN "*_test.py")

install(FILES basic/example.py
	DESTINATION ${INSTALL_EXAMPLES})
