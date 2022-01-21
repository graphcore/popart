// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CASTTOFP8_HPP
#define GUARD_NEURALNET_CASTTOFP8_HPP

#include <popart/op.hpp>

namespace popart {

// CastToFp8Op casts input tensor from FLOAT/FLOAT16 to FLOAT8, user can specify
// the number of bits for mantissa and exponent, exponent bias, which is
// different from CastOp
class CastToFp8Op : public Op {
public:
  CastToFp8Op(const OperatorIdentifier &_opid,
              int _nBitMantissa,
              int _nBitExponent,
              int _exponentBias,
              const Op::Settings &settings);
  std::unique_ptr<Op> clone() const override;
  void setup() override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  int getNBitMantissa() { return nBitMantissa; }
  int getNBitExponent() { return nBitExponent; }
  int getExponentBias() { return exponentBias; }

private:
  int nBitMantissa;
  int nBitExponent;
  int exponentBias;
};

} // namespace popart

#endif
