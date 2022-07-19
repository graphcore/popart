// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_CES_ELEMENTWISECE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_CES_ELEMENTWISECE_HPP_

#include <vector>
#include <popart/ces/constexpr.hpp>

namespace popart {
class Op;

class ConstExprAdd : public ConstExprOp {
public:
  ConstExprAdd(Op *op);
  std::vector<char> compute() final;
};

class ConstExprDiv : public ConstExprOp {
public:
  ConstExprDiv(Op *op);
  std::vector<char> compute() final;
};

class ConstExprMul : public ConstExprOp {
public:
  ConstExprMul(Op *op);
  std::vector<char> compute() final;
};

class ConstExprSub : public ConstExprOp {
public:
  ConstExprSub(Op *op);
  std::vector<char> compute() final;
};

class ConstExprFmod : public ConstExprOp {
public:
  ConstExprFmod(Op *op);
  std::vector<char> compute() final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_CES_ELEMENTWISECE_HPP_
