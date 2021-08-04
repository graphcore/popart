# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
PROJECT_SOURCE_DIR=$1
CMAKE_CURRENT_BINARY_DIR=$2
YEAR=$(date +"%Y")
OUT_FILE="$CMAKE_CURRENT_BINARY_DIR/all_headers.hpp"

echo "// Copyright (c) $YEAR Graphcore Ltd. All rights reserved." > $OUT_FILE
for header in $(find $PROJECT_SOURCE_DIR/willow/include -name '*hpp' | sort);
                              do printf "#include <$header>\n";
                            done >> $OUT_FILE
