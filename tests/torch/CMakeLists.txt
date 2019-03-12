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

configure_file(cifar10/download_dataset.py.in
  ${CMAKE_BINARY_DIR}/cifar10/download_dataset.py
  @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/cifar10/download_dataset.py
  DESTINATION ${INSTALL_TESTS}/torch/cifar10/)
#------ end of download data script


set(CPACK_SOURCE_IGNORE_FILES ${INSTALL_TESTS})

# poponnx.torch tests (require pytorch and torchvision)
add_test(NAME model0_train_cpu
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND python3 torch/cifar10/model0.py --device=cpu)


add_test(NAME model0_train_ipumodel
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND python3 torch/cifar10/model0.py --device=ipu_model)

add_test(NAME model0_evaluate
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND python3 torch/cifar10/model0_evaluate.py)

add_test(NAME model0_infer
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND python3 torch/cifar10/model0_infer.py)

add_test(NAME model1
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model1.py)

add_test(NAME model2
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model2.py)

add_test(NAME model5
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model5.py)

add_test(NAME model6
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model6.py)

add_test(NAME model_reduce_sum
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_reduce_sum.py)

add_test(NAME model7
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model7.py)

add_test(NAME model8
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model8.py)

add_test(NAME model_conv_bias
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_conv_bias.py)

add_test(NAME reset_weights
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/reset_weights.py)

add_test(NAME model_reshape
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_reshape.py)

add_test(NAME model_cat
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_cat.py)

add_test(NAME model_softmax
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_softmax.py)

add_test(NAME model_nll
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_nll.py)

add_test(NAME model_matmul_bcast
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_matmul_bcast.py)

add_test(NAME model_lstm
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_lstm.py)

add_test(NAME model_instancenorm
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND python3 torch/cifar10/model_instancenorm.py)

# see T7159  
# add_test(NAME model_batchnorm
# 		 WORKING_DIRECTORY ${INSTALL_TESTS}
# 		 COMMAND python3 torch/cifar10/model_batchnorm.py --device=cpu)
