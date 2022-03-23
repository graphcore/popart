// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/op.hpp>
#include <popart/op/exchange/hostbase.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

// HostBaseOp
void HostBaseOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("hostStreamTensorId", hostStreamTensorId);
}

} // namespace popart
