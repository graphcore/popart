Overview of the models
----------------------
Please refer the the model .py files to verify what is stated below

model0
------
output  = relu(conv(input)) and loss is l1 norm of output

model1
------
two image inputs and a label. Uses AveragePooling, and 2 losses : l1 and nll

model2
------
A simple model which include a linear layer

model5
------
Basic test of NLL (testing bug fix to issue reported in T5271)

model6
------
same as model1, but uses Subtract rather than Add

model_reduce_sum
------
output = sum(conv(input), dim=1) and loss is l1 norm of output

model7
------
Same setup as model2 but with a matmul at the end

model_conv_bias
------
output = conv(input) with bias and loss is l1 norm of output

reset_weights
------
Test updating the weights using a second model that differs from the first only in weights.
Create a pytorch model, export it to onnx and load it using poponnx.
Run a number of steps on the pytorch model, reexport the model to onnx, and use this onnx model
to reset the weights of the poponnx model.
