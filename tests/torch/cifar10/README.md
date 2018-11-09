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
