// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <testutil/irquery/tensorindextestwrapper.hpp>

#include <testutil/irquery/tensortestwrapper.hpp>

namespace popart {
namespace irquery {

TensorIndexTestWrapper::TensorIndexTestWrapper(
    Ir &ir,
    const std::pair<int, Tensor *> &tensorIndex,
    const std::string &srcObjDescr_,
    const std::string &descrSingular_,
    const std::string &descrPlural_)
    : TestWrapper<std::pair<int, Tensor *>>{ir, tensorIndex},
      srcObjDescr{srcObjDescr_}, descrSingular{descrSingular_},
      descrPlural{descrPlural_} {}

int TensorIndexTestWrapper::index() { return wrappedObj.first; }

TensorId TensorIndexTestWrapper::id() {
  Tensor *tensor = wrappedObj.second;
  return tensor->id;
}

TensorTestWrapper TensorIndexTestWrapper::tensor() {
  Tensor *tensor = wrappedObj.second;
  return TensorTestWrapper{ir, tensor};
}

} // namespace irquery
} // namespace popart