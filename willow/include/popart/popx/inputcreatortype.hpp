// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_POPX_INPUTCREATORTYPE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_POPX_INPUTCREATORTYPE_HPP_

namespace popart {
namespace popx {

enum class InputCreatorType {
  // Opx has a poplar call to a function that can
  // lay out the input tensor on the device
  CanCreate = 0,
  // Cannot create the input tensor, but can
  // allow an Opx downstream in the graph to
  // create it
  CanUnwind,
  // Can create or unwind
  CanCreateOrUnwind,
  // Cannot create tensor, nor can it allow a
  // a downstream Opx to create the tensor
  Deadend,
  // Has a potential creator, but can also allow an Opx downstream in the graph
  // to create it instead.
  CanDelegate,
  // Has a potential creator, but can also allow an Opx downstream in the graph
  // to create it instead (either propagated through the subgraph, or directly).
  CanDelegateOrUnwind
};

} // namespace popx
} // namespace popart
#endif // POPART_WILLOW_INCLUDE_POPART_POPX_INPUTCREATORTYPE_HPP_
