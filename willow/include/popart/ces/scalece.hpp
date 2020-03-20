// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CONSTEXPRS_SCALECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SCALECE_HPP

#include <popart/ces/constexpr.hpp>

namespace popart {

class ConstExprScale : public ConstExprOp {
public:
  ConstExprScale(Op *);
  std::vector<char> compute() final;

private:
  // obtained from the onnx::NodeProto, the factor by which to scale the input
  float factor32;
  // the actual scaling will be done in double precision for all types
  double factor64;
};
} // namespace popart

#endif
