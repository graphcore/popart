include(CPack)

set(CPACK_SOURCE_IGNORE_FILES ${INSTALL_TESTS})

# poponnx.torch tests (require pytorch and torchvision)
add_test(NAME model0
		 WORKING_DIRECTORY ${INSTALL_TESTS}
		 COMMAND python3 torch/cifar10/model0.py .)

add_test(NAME model1
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model1.py .)

#add_test(NAME model2
#		WORKING_DIRECTORY ${INSTALL_TESTS}
#		COMMAND python3 torch/cifar10/model2.py .)

add_test(NAME model5
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model5.py .)

add_test(NAME model6
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model6.py .)

add_test(NAME model_reduce_sum
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_reduce_sum.py .)
		
add_test(NAME model7
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model7.py .)

add_test(NAME model_conv_bias
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/model_conv_bias.py .)

add_test(NAME reset_weights
		WORKING_DIRECTORY ${INSTALL_TESTS}
		COMMAND python3 torch/cifar10/reset_weights.py .)
