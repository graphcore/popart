echo "clang-format -i willow/src/*cpp"
clang-format -i willow/src/*cpp
echo "clang-format -i willow/src/popx/*cpp"
clang-format -i willow/src/popx/*cpp
echo "clang-format -i willow/include/willow/*hpp"
clang-format -i willow/include/willow/*hpp
echo "clang-format -i willow/include/willow/popx/*hpp"
clang-format -i willow/include/willow/popx/*hpp
echo "clang-format -i python/poponnx/*cpp"
clang-format -i tests/poponnx/*cpp
echo "python -m yapf -i tests/poponnx/*py"
python -m yapf -i tests/torch/*py
echo "python -m yapf -i tests/torch/cifar10/*py"
python -m yapf -i tests/torch/cifar10/*py
echo "python -m yapf -i python/poponnx/*py"
python -m yapf -i python/poponnx/*py
