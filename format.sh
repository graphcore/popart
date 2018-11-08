echo "Inplace clang-formatting all .cpp files in listed directories,"
clang-format -i python/poponnx.cpp
clang-format -i willow/src/*cpp
clang-format -i willow/src/popx/*cpp

echo "inplace clang-formatting all .hpp files in listed directories,"
clang-format -i willow/include/willow/*hpp
clang-format -i willow/include/willow/popx/*hpp

echo "inplace yapfing all .py files in listed directories,"
python -m yapf -i tests/torch/cifar10/*py
python -m yapf -i tests/poponnx/*py
python -m yapf -i python/poponnx/torch/*py
python -m yapf -i python/poponnx/*py

echo "formatting complete."
