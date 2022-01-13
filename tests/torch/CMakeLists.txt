# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
include(CPack)

#------ download data script
SET(C10_DIR "" CACHE PATH "Directory to install Cifar-10 dataset to")
SET(C10_PYSTRING_DIR "tempfile.gettempdir()")
IF(C10_DIR)
  IF(IS_DIRECTORY ${C10_DIR})
    SET(C10_PYSTRING_DIR "\"${C10_DIR}\"")
  ELSE()
    MESSAGE(FATAL_ERROR "Directory ${C10_DIR} does not exist, invalid C10_DIR")
  ENDIF()
ENDIF()
MESSAGE(STATUS "C10_PYSTRING_DIR is ${C10_PYSTRING_DIR}")
configure_file(cifar10/c10datadir.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/cifar10/c10datadir.py
  @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cifar10/c10datadir.py
  DESTINATION ${INSTALL_TESTS}/torch/cifar10)

set(CIFAR10_TEST_DIR "${INSTALL_TESTS}/torch/cifar10/")

add_test(NAME download_cifar10_dataset_mkdir_and_check_space
  	WORKING_DIRECTORY "${CIFAR10_TEST_DIR}"
    COMMAND ${Python3_EXECUTABLE} -m download_dataset.mkdir_and_check_space)
set_tests_properties(download_cifar10_dataset_mkdir_and_check_space
    PROPERTIES FIXTURES_SETUP cifar10_dirmade
    SKIP_RETURN_CODE 2)

add_test(NAME download_cifar10_dataset_actual_download
  	WORKING_DIRECTORY "${CIFAR10_TEST_DIR}"
    COMMAND ${Python3_EXECUTABLE} -m download_dataset.actual_download)
set_tests_properties(download_cifar10_dataset_actual_download
    PROPERTIES FIXTURES_REQUIRED cifar10_dirmade
               FIXTURES_SETUP cifar10_downloaded
    SKIP_RETURN_CODE 2)

add_test(NAME download_cifar10_dataset_verify_tar_file
  	WORKING_DIRECTORY "${CIFAR10_TEST_DIR}"
    COMMAND ${Python3_EXECUTABLE} -m download_dataset.verify_tar_file)
set_tests_properties(download_cifar10_dataset_verify_tar_file
    PROPERTIES FIXTURES_REQUIRED cifar10_downloaded
               FIXTURES_SETUP cifar10_tar_verified
    SKIP_RETURN_CODE 2)

add_test(NAME download_cifar10_dataset_extract_tar_file
    WORKING_DIRECTORY "${CIFAR10_TEST_DIR}"
    COMMAND ${Python3_EXECUTABLE} -m download_dataset.extract_tar_file)
set_tests_properties(download_cifar10_dataset_extract_tar_file
    PROPERTIES FIXTURES_REQUIRED cifar10_tar_verified
               FIXTURES_SETUP cifar10_tar_extracted
    SKIP_RETURN_CODE 2)

add_test(NAME download_cifar10_dataset
	COMMAND ${Python3_EXECUTABLE} ${INSTALL_TESTS}/torch/cifar10/download_dataset.py)
set_tests_properties(download_cifar10_dataset
    PROPERTIES FIXTURES_REQUIRED cifar10_tar_extracted FIXTURES_SETUP cifar10)

#------ end of download data script


set(CPACK_SOURCE_IGNORE_FILES ${INSTALL_TESTS})

# popart.torch tests (require pytorch and torchvision)
add_test(NAME model0_train_cpu
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model0.py --device=cpu)
set_tests_properties(model0_train_cpu PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model0_train_ipumodel
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model0.py --device=ipu_model)
set_tests_properties(model0_train_ipumodel PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model0_evaluate
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model0_evaluate.py)
set_tests_properties(model0_evaluate PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model0_infer
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model0_infer.py)
set_tests_properties(model0_infer PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model1
	WORKING_DIRECTORY ${INSTALL_TESTS}
	COMMAND ${Python3_EXECUTABLE} torch/cifar10/model1.py)

set_tests_properties(model1 PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model2
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model2.py)
set_tests_properties(model2 PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model5
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model5.py)
set_tests_properties(model5 PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model6
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model6.py)
set_tests_properties(model6 PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_reduce_sum
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_reduce_sum.py)
set_tests_properties(model_reduce_sum PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model7
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model7.py)
set_tests_properties(model7 PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model8
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model8.py)
set_tests_properties(model8 PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_conv_bias
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_conv_bias.py)
set_tests_properties(model_conv_bias PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME reset_weights
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/reset_weights.py)
set_tests_properties(reset_weights PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_reshape
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_reshape.py)
set_tests_properties(model_reshape PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_cat
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_cat.py)
set_tests_properties(model_cat PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_softmax
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_softmax.py)
set_tests_properties(model_softmax PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_nll
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_nll.py)
set_tests_properties(model_nll PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_matmul_bcast
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_matmul_bcast.py)
set_tests_properties(model_matmul_bcast PROPERTIES FIXTURES_REQUIRED cifar10)

add_test(NAME model_lstm
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_lstm.py)
set_tests_properties(model_lstm PROPERTIES FIXTURES_REQUIRED cifar10)

# See D14275 this test is failing on OSX and needs investigating
# add_test(NAME model_instancenorm
# 		 WORKING_DIRECTORY ${INSTALL_TESTS}
# 		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_instancenorm.py)
# set_tests_properties(model_instancenorm PROPERTIES FIXTURES_REQUIRED cifar10)

# flaky test, T16816
# add_test(NAME model_loss_scaling
# 		 WORKING_DIRECTORY ${INSTALL_TESTS}
# 		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_loss_scaling.py)
# set_tests_properties(model_loss_scaling PROPERTIES FIXTURES_REQUIRED cifar10)

# see T7159
# add_test(NAME model_batchnorm
# 		 WORKING_DIRECTORY ${INSTALL_TESTS}
# 		 COMMAND ${Python3_EXECUTABLE} torch/cifar10/model_batchnorm.py --device=cpu)
# set_tests_properties(model_batchnorm PROPERTIES FIXTURES_REQUIRED cifar10)
