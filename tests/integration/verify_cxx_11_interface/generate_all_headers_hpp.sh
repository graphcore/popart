PROJECT_SOURCE_DIR=$1
CMAKE_CURRENT_BINARY_DIR=$2
for header in $(find $PROJECT_SOURCE_DIR/willow/include -name '*hpp' | sort);
                              do printf "#include <$header>\n";
                            done > "$CMAKE_CURRENT_BINARY_DIR/all_headers.hpp"
