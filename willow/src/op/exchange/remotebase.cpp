// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>
#include <popart/op/exchange/remotebase.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

// RemoteBaseOp
void RemoteBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remoteBufferId);
}

} // namespace popart
