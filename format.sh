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

# number of threads used by xargs
PROC_COUNT=12

# run clang format on all .cpp and .hpp files in directory at $1
function format_cpp_files {
  printf "  -->  Inplace clang-formatting all .hpp files in $1 directory\n"
  find $1 -iname "*.hpp" | xargs -n 1 -P $PROC_COUNT clang-format -i
  printf "  -->  Inplace clang-formatting all .cpp files in $1 directory\n"
  find $1 -iname "*.cpp" | xargs -n 1 -P $PROC_COUNT clang-format -i
}

# run yapf on all .py files in directory at $1
function format_py_files {
  printf "  -->  Inplace yapfing all .py files in $1 directory\n"
  find $1 -iname "*.py" | xargs -n 1 -P $PROC_COUNT python3 -m yapf -i
}

# format all cpp and hpp files
format_cpp_files willow
format_cpp_files tests
format_cpp_files python

# format all py files
format_py_files python
format_py_files tests
format_py_files scripts

printf "\nFormatting complete.\n"
