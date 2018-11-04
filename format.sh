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
echo "yapf -i tests/basic/*py"
yapf -i tests/basic/*py
echo "yapf -i tests/cifar10/*py"
yapf -i tests/cifar10/*py
echo "yapf -i pywillow/*py"
yapf -i pywillow/*py
