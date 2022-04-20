// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <functional>
#include <ostream>
#include <string>
#include <testutil/irquery/opstestwrapper.hpp>
#include <testutil/irquery/tensortestwrapper.hpp>
#include <popart/tensor.hpp>

#include "testutil/irquery/testwrapper.hpp"
#include "testutil/irquery/testwrapper_impl.hpp"

namespace popart {
class Ir;

namespace irquery {

TensorTestWrapper::TensorTestWrapper(Ir &ir, Tensor *tensor)
    : TestWrapper<Tensor *>{ir, tensor} {}

OpsTestWrapper TensorTestWrapper::consumers() {
  Tensor *tensor = wrappedObj;
  auto ops       = tensor->consumers.getOps();
  std::stringstream ss;
  ss << "consumers of tensor '" << tensor->id << "'";
  return OpsTestWrapper{ir, ops, ss.str()};
}

} // namespace irquery
} // namespace popart
