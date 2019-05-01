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

printf "  -->  Inplace clang-formatting all .cpp files in willow directory\n"
find willow -iname *.cpp | xargs clang-format -i -verbose

printf "\n  -->  Inplace clang-formatting all .hpp files in willow directory\n"
find willow -iname *.hpp | xargs clang-format -i -verbose

printf "\n  -->  Inplace clang-formatting all .cpp/.hpp files in tests directory\n"
find tests -iname *.cpp | xargs clang-format -i -verbose
find tests -iname *.hpp | xargs clang-format -i -verbose

printf "\n  -->  Inplace clang-formatting all .cpp files in python directory\n"
find python -iname *.cpp | xargs clang-format -i -verbose

printf "\n  -->  Inplace yapfing all .py files in python/poponnx\n"
python3 -m yapf -i python/poponnx/*py


echo "  -->  Inplace yapfing all .py test files"
python3 -m yapf -i tests/torch/cifar10/*py
python3 -m yapf -i tests/poponnx/*py
python3 -m yapf -i tests/poponnx/operators_test/*py

echo "  -->  Inplace yapfing all .py scripts"
python3 -m yapf -i scripts/*py

echo "  -->  Inplace yapfing all .py files in python/poponnx/torch"
python3 -m yapf -i python/poponnx/torch/*py


printf "\nFormatting complete.\n"
