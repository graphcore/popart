// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnxutil.hpp>
#include <popart/graph.hpp>
#include <popart/op/casttofp8.hpp>
#include <popart/opmanager.hpp>

namespace popart {

CastToFp8Op::CastToFp8Op(const OperatorIdentifier &_opid,
                         int _nBitMantissa,
                         int _nBitExponent,
                         int _exponentBias,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), nBitMantissa(_nBitMantissa),
      nBitExponent(_nBitExponent), exponentBias(_exponentBias) {}

std::unique_ptr<Op> CastToFp8Op::clone() const {
  return std::make_unique<CastToFp8Op>(*this);
}

void CastToFp8Op::setup() {
  TensorInfo info = inInfo(getInIndex());

  info.set(DataType::FLOAT8);
  outInfo(getOutIndex()) = info;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT16, DataType::FLOAT};
static OpDefinition::DataTypes T2 = {DataType::FLOAT8};

static OpDefinition CastToFp8Def({OpDefinition::Inputs({{"input", T1}}),
                                  OpDefinition::Outputs({{"output", T2}}),
                                  OpDefinition::Attributes({
                                      {"nBitMantissa", {"INT32"}},
                                      {"nBitExponent", {"INT32"}},
                                      {"exponentBias", {"INT32"}},
                                  })});

static OpCreator<CastToFp8Op> CastToFp8Creator(
    OpDefinitions({
        {Onnx::CustomOperators::CastToFp8, CastToFp8Def},
    }),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      int nBitMantissa =
          oci.attributes.getAttribute<Attributes::Int>("nBitMantissa", 4);
      int nBitExponent =
          oci.attributes.getAttribute<Attributes::Int>("nBitExponent", 3);
      int exponentBias =
          oci.attributes.getAttribute<Attributes::Int>("exponentBias", 7);

      return std::unique_ptr<CastToFp8Op>(new CastToFp8Op(
          oci.opid, nBitMantissa, nBitExponent, exponentBias, oci.settings));
    },
    true);
} // namespace

} // namespace popart
