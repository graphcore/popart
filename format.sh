echo "clang-format -i willow/src/*cpp"
clang-format -i willow/src/*cpp
echo "clang-format -i willow/src/popx/*cpp"
clang-format -i willow/src/popx/*cpp
echo "clang-format -i willow/include/willow/*hpp"
clang-format -i willow/include/willow/*hpp
echo "clang-format -i willow/include/willow/popx/*hpp"
clang-format -i willow/include/willow/popx/*hpp
echo "clang-format -i pywillow/*cpp"
clang-format -i pywillow/*cpp
echo "python -m yapf -i tests/basic/*py"
python -m yapf -i tests/basic/*py
echo "python -m yapf -i tests/cifar10/*py"
python -m yapf -i tests/cifar10/*py
echo "python -m yapf -i pywillow/*py"
python -m yapf -i pywillow/*py
