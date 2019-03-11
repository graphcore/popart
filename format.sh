#how to format recursively with clang-format, from
# https://stackoverflow.com/questions/28896909

# we encourage all commits to be done with clang-format version 8
# note that the linter does not use this script for formatting.
cf_version=$(python3 scripts/get_clang_format_version.py)
if [[ "${cf_version}" -lt 8 ]];
then 
echo "Clang-format version should be 8.0.0 or greater. On OS/X this is the default, on Ubuntu consider installing clang-format locally with linuxbrew if you cannot install to filesystem"
exit
fi

echo "Inplace clang-formatting all .cpp files in willow directory"
find willow/ -iname *.cpp | xargs clang-format -i

echo "Inplace clang-formatting all .cpp files in tests directory"
find tests/ -iname *.cpp | xargs clang-format -i

echo "Inplace clang-formatting all .hpp files in willow directory"
find willow/ -iname *.hpp | xargs clang-format -i

echo "Inplace clang-formatting python/poponnx.cpp"
clang-format -i python/poponnx.cpp

echo "inplace yapfing all .py files in listed directories"
python3 -m yapf -i tests/torch/cifar10/*py
python3 -m yapf -i tests/poponnx/*py
python3 -m yapf -i scripts/*py
python3 -m yapf -i tests/poponnx/operators_test/*py
python3 -m yapf -i python/poponnx/torch/*py
python3 -m yapf -i python/poponnx/*py

echo "formatting complete."

