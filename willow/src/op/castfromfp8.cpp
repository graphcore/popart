// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <memory>
#include <onnxutil.hpp>
#include <popart/graph.hpp>
#include <popart/op/castfromfp8.hpp>
#include <popart/opmanager.hpp>

namespace popart {

CastFromFp8Op::CastFromFp8Op(const OperatorIdentifier &_opid,
                             DataType _to,
                             int _nBitMantissa,
                             int _nBitExponent,
                             int _exponentBias,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), to(_to), nBitMantissa(_nBitMantissa),
      nBitExponent(_nBitExponent), exponentBias(_exponentBias) {}

std::unique_ptr<Op> CastFromFp8Op::clone() const {
  return std::make_unique<CastFromFp8Op>(*this);
}

void CastFromFp8Op::setup() {
  TensorInfo info = inInfo(getInIndex());

  info.set(to);
  outInfo(getOutIndex()) = info;
}

namespace {

static OpDefinition::DataTypes T1 = {DataType::FLOAT8};
static OpDefinition::DataTypes T2 = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition CastFromFp8Def({OpDefinition::Inputs({{"input", T1}}),
                                    OpDefinition::Outputs({{"output", T2}}),
                                    OpDefinition::Attributes({
                                        {"to", {"FLOAT|FLOAT16"}},
                                        {"nBitMantissa", {"INT32"}},
                                        {"nBitExponent", {"INT32"}},
                                        {"exponentBias", {"INT32"}},
                                    })});

static OpCreator<CastFromFp8Op> CastFromFp8Creator(
    OpDefinitions({
        {Onnx::CustomOperators::CastFromFp8, CastFromFp8Def},
    }),
    [](const popart::OpCreatorInfo &oci) -> std::unique_ptr<popart::Op> {
      std::string type  = oci.attributes.getAttribute<Attributes::String>("to");
      DataType dataType = dataTypeFromString(type);
      int nBitMantissa =
          oci.attributes.getAttribute<Attributes::Int>("nBitMantissa", 4);
      int nBitExponent =
          oci.attributes.getAttribute<Attributes::Int>("nBitExponent", 3);
      int exponentBias =
          oci.attributes.getAttribute<Attributes::Int>("exponentBias", 7);

      return std::unique_ptr<CastFromFp8Op>(new CastFromFp8Op(oci.opid,
                                                              dataType,
                                                              nBitMantissa,
                                                              nBitExponent,
                                                              exponentBias,
                                                              oci.settings));
    },
    true);
} // namespace

} // namespace popart
