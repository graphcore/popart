# Copyright (c) 2021 Graphcore Ltd. All rights reserved.

# Initialise variables.
# ------------------------------------------------------------------------------
# A path to the file which contains the metadata of all ops.
set(popart_op_metadata_path ${PROJECT_BINARY_DIR}/metadata/op.json)
# A path to the dir which contains all op bindings.
set(popart_op_bindings_dir ${CMAKE_CURRENT_BINARY_DIR}/bindings/op)
# A path to the dir which contains all op headers.
set(popart_op_headers_dir ${PROJECT_SOURCE_DIR}/willow/include/popart/op/)
# A path to the dir which contains Jinja2 templates for writing source files.
set(templates_dir ${CMAKE_CURRENT_SOURCE_DIR}/templates)
# Paths to the generating scripts.
set(gen_op_metadata_path ${PROJECT_SOURCE_DIR}/scripts/gen_op_metadata.py)
set(gen_op_bindings_path ${PROJECT_SOURCE_DIR}/scripts/gen_op_bindings.py)

# Store a list of (initially a selected set of) all op headers.
list(
  APPEND
  op_headers
  # ${popart_op_headers_dir}/accumulate.hpp # Manually bound
  ${popart_op_headers_dir}/add.hpp
  ${popart_op_headers_dir}/subtract.hpp
  ${popart_op_headers_dir}/sum.hpp
  # ${popart_op_headers_dir}/loop.hpp # Manually bound
  # ${popart_op_headers_dir}/call.hpp # Manually bound
  # ${popart_op_headers_dir}/matmul.hpp # Manually bound
  ${popart_op_headers_dir}/exchange/hostcopy.hpp
  ${popart_op_headers_dir}/incrementmod.hpp
  ${popart_op_headers_dir}/init.hpp
  ${popart_op_headers_dir}/mul.hpp
  ${popart_op_headers_dir}/div.hpp
  ${popart_op_headers_dir}/pow.hpp
  ${popart_op_headers_dir}/printtensor.hpp
  ${popart_op_headers_dir}/dropout.hpp
  ${popart_op_headers_dir}/gather.hpp
  ${popart_op_headers_dir}/scatter.hpp
  ${popart_op_headers_dir}/cast.hpp
  ${popart_op_headers_dir}/collectives/collectives.hpp
  ${popart_op_headers_dir}/collectives/replicatedallgather.hpp
  ${popart_op_headers_dir}/collectives/replicatedallreduce.hpp
  ${popart_op_headers_dir}/collectives/replicatedreducescatter.hpp
  ${popart_op_headers_dir}/copyvarupdate.hpp
  ${popart_op_headers_dir}/tiedgather.hpp
  ${popart_op_headers_dir}/groupnorm.hpp
  ${popart_op_headers_dir}/equal.hpp
  ${popart_op_headers_dir}/or.hpp
  ${popart_op_headers_dir}/and.hpp
  ${popart_op_headers_dir}/not.hpp
  ${popart_op_headers_dir}/relu.hpp
  ${popart_op_headers_dir}/dynamic/dynamicupdate.hpp
  ${popart_op_headers_dir}/dynamic/dynamicslice.hpp
  ${popart_op_headers_dir}/reshape.hpp
  ${popart_op_headers_dir}/gelu.hpp
  ${popart_op_headers_dir}/transpose.hpp
  ${popart_op_headers_dir}/slice.hpp
  ${popart_op_headers_dir}/split.hpp
  # ${popart_op_headers_dir}/ipucopy.hpp manual binding
  ${popart_op_headers_dir}/negate.hpp
  ${popart_op_headers_dir}/softmax.hpp
  ${popart_op_headers_dir}/nll.hpp
  ${popart_op_headers_dir}/detach.hpp
  ${popart_op_headers_dir}/randomuniform.hpp
  ${popart_op_headers_dir}/randomnormal.hpp
  ${popart_op_headers_dir}/tanh.hpp)
# file(GLOB_RECURSE op_headers ${popart_op_headers_dir}/*.hpp)

# Store lists of the locations of the sources and headers for generated op
# bindings.
set(op_binding_headers "")
set(op_binding_sources "")
foreach(op_header IN LISTS op_headers)
  string(REGEX REPLACE "^${popart_op_headers_dir}" ${popart_op_bindings_dir}
                       tmp ${op_header})
  string(REGEX REPLACE ".hpp$" ".gen.hpp" op_binding_header ${tmp})
  list(APPEND op_binding_headers ${op_binding_header})
  string(REGEX REPLACE ".hpp$" ".gen.cpp" op_binding_source ${tmp})
  list(APPEND op_binding_sources ${op_binding_source})
endforeach()
list(APPEND op_bindings ${op_binding_headers} ${op_binding_sources})

# Store a list of all include directories that are needed to build popart. NOTE:
# This list contains directories which don't get resolved until build time.
# Targets that rely on this list should DEPEND on `popart-only`.
set(popart_include_directories "")
get_include_directories_from_target(popart-only popart_include_directories)

# Custom target for generating op metadata.
# ------------------------------------------------------------------------------
# Command for parsing all op headers and generating a metadata file.
add_custom_command(
  OUTPUT ${popart_op_metadata_path}
  DEPENDS popart-only ${gen_op_metadata_path}
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMAND
    python3 ${gen_op_metadata_path} --headers "\"${op_headers}\"" --out
    ${popart_op_metadata_path} --include-directories
    "\"${popart_include_directories}\"")

# A corresponding target to the command above.
add_custom_target(popart_op_metadata ALL DEPENDS ${popart_op_metadata_path})

# Custom target for generating op binding source files.
# ------------------------------------------------------------------------------
# Command for parsing the metadata file and writing op bindings.
set(op_bindings__all ${popart_op_bindings_dir}/_all.gen.cpp)
add_custom_command(
  OUTPUT ${op_bindings} ${op_bindings__all}
         ${popart_op_bindings_dir}/../graph.gen.cpp
  DEPENDS popart_op_metadata
          ${gen_op_metadata_path}
          ${gen_op_bindings_path}
          ${templates_dir}/graph.cpp.j2
          ${templates_dir}/op/_op.cpp.j2
          ${templates_dir}/op/_op.hpp.j2
          ${templates_dir}/op/_all.cpp.j2
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMMAND python3 ${gen_op_bindings_path} --json_path
          ${popart_op_metadata_path} --out ${popart_op_bindings_dir})

# A corresponding target to the command above.
add_custom_target(popart_op_bindings ALL DEPENDS ${op_bindings})

# Python bindings for the PopART IR.
# ------------------------------------------------------------------------------
add_popart_python_module(
  popart_internal_ir
  ../popart/shared_cpp/np_utils.cpp
  ${op_binding_sources}
  ${op_bindings__all}
  ${popart_op_bindings_dir}/../graph.gen.cpp
  # patterns
  bindings/patterns/pattern.cpp
  bindings/patterns/patterns.cpp
  # End patterns
  bindings/debugcontext.cpp
  bindings/graph.cpp
  bindings/graphid.cpp
  bindings/ir.cpp
  bindings/op.cpp
  # Manual ops
  bindings/op/enums.cpp
  bindings/op/loop.cpp
  bindings/op/ipucopy.cpp
  bindings/op/call.cpp
  bindings/op/matmul.cpp
  bindings/op/accumulate.cpp
  bindings/op/manualbindops.cpp
  bindings/op/adamupdater.cpp
  bindings/op/optimizervalue.cpp
  bindings/op/varupdate.cpp
  bindings/op/accumulatorscale.cpp
  bindings/op/accumulatorzero.cpp
  # End ops
  bindings/scope.cpp
  bindings/tensor.cpp
  bindings/tensordata.cpp
  bindings/tensorinfo.cpp
  bindings/tensors.cpp
  bindings/basicoptionals.cpp
  bindings/opidentifier.cpp
  bindings/region.cpp
  bindings/tensorlocation.cpp
  bindings/bwdgraphinfo.cpp
  bindings/commgroup.cpp
  bindings/util.cpp
  # Transforms
  bindings/transforms/transform.cpp
  bindings/transforms/prune.cpp
  bindings/transforms/autodiff.cpp
  # end transforms
  bindings/op/enums.cpp
  bindings/op/optional.cpp
  popart._internal.ir.cpp)

target_include_directories(
  popart_internal_ir PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
                             "${CMAKE_CURRENT_BINARY_DIR}")

# The top-level "__init__.py" file for popart is in the main package directory
# "${CMAKE_CURRENT_BINARY_DIR}/../popart/popart/". The
# "_internal/ir/__init__.py" file should be in a child dir of the main dir for
# scripts, such as "docs/gen_supported_ops.py" to work, since they are executed
# from the build dir.
configure_file(
  __init__.py.in
  "${CMAKE_CURRENT_BINARY_DIR}/../popart/popart/_internal/ir/__init__.py" @ONLY)

# This works because "_internal/ir/__init__.py" imports the pybind .so using from ... import *
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/popart_internal_ir.pyi"
  DESTINATION "${INSTALL_PYDIR}/popart/_internal/ir/"
  RENAME "__init__.pyi"
  OPTIONAL)
