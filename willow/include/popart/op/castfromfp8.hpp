// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_CASTFROMFP8_HPP
#define GUARD_NEURALNET_CASTFROMFP8_HPP

#include <popart/op.hpp>

namespace popart {

// CastFromFp8Op casts input tensor from FLOAT8 to FLOAT/FLOAT16, user can
// specify the number of bits for mantissa and exponent, exponent bias, which is
// different from CastOp

class CastFromFp8Op : public Op {
public:
  CastFromFp8Op(const OperatorIdentifier &_opid,
                DataType _to,
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
  DataType getDataType() { return to; }

private:
  DataType to;
  int nBitMantissa;
  int nBitExponent;
  int exponentBias;
};

} // namespace popart

#endif
