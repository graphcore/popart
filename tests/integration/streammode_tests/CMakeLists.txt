# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_popart_py_unit_test(tensor_replication VARIANTS "${HW_EXCLUDE_C600}")
add_popart_py_unit_test(grouped_initialization_test VARIANTS "${HW_EXCLUDE_C600}" MATCHEXPR "test_onnx_checkpointing")
add_popart_py_unit_test(grouped_initialization_test VARIANTS "${HW_EXCLUDE_C600}" MATCHEXPR "test_locations")
add_popart_py_unit_test(test_broadcast_replication VARIANTS "Hw")

# test_grouped_initialization test starts here
set(config_options "config0" "config1" "config2" "config3" "config4" "config5" "config6" "config7" "config8" "config9" "config10")
set(locations "location0" "location1" "location2")
set(session_types "training" "inference")

foreach(config ${config_options})
    foreach(location ${locations})
        foreach(session_type ${session_types})
# linter complains if this is indented
add_popart_py_unit_test(grouped_initialization_test VARIANTS "${HW_EXCLUDE_C600}" MATCHEXPR "[${session_type}-${location}-${config}]")
        endforeach()
    endforeach()
endforeach()
# test_grouped_initialization test ends here
