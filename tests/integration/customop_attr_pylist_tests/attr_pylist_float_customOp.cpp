// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <vector>
// #include <filereader.hpp>
// #include <testdevice.hpp>

#include "popart/datatype.hpp"
#include <popart/op.hpp>
#include <popart/opmanager.hpp>
#include <popart/popx/opx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

using namespace popart;

namespace CustomOperators {
const OperatorIdentifier CustomopAttrPylist = {"com.acme",
                                               "CustomopAttrPylist",
                                               1};
} // namespace CustomOperators

// An IdentityOp that doesn't return any grad ops.
class CustomopAttrPylistOp : public Op {
public:
  CustomopAttrPylistOp(const OperatorIdentifier &_opid,
                       const Op::Settings &settings_)
      : Op(_opid, settings_) {}

  void setup() final { outInfo(0) = inInfo(0); }

  std::unique_ptr<Op> clone() const final {
    return std::make_unique<CustomopAttrPylistOp>(*this);
  }

  float getSubgraphValue() const final { return getLowSubgraphValue(); }
};

static popart::OpDefinition CustomopAttrPylistOpDef(
    {popart::OpDefinition::Inputs({
         {"input", {{popart::DataType::FLOAT, popart::DataType::FLOAT16}}},
     }),
     popart::OpDefinition::Outputs(
         {{"output", {{popart::DataType::FLOAT, popart::DataType::FLOAT16}}}}),
     popart::OpDefinition::Attributes({{"values", {"*"}}})});

static OpCreator<CustomopAttrPylistOp> CustomopAttrPylistOpCreator(
    OpDefinitions({{CustomOperators::CustomopAttrPylist,
                    CustomopAttrPylistOpDef}}),
    [](const OpCreatorInfo &info) -> std::unique_ptr<Op> {
      const OperatorIdentifier &opid = info.opid;
      const Op::Settings &settings   = info.settings;
      const Attributes &attr         = info.attributes;
      auto values = attr.getAttribute<Attributes::Floats>("values");
      return std::unique_ptr<Op>(new CustomopAttrPylistOp(opid, settings));
    },
    true);

class CustomopAttrPylistOpx : public popx::Opx {
public:
  CustomopAttrPylistOpx(Op *op, popx::Devicex *devicex)
      : popx::Opx(op, devicex) {
    verifyOp<CustomopAttrPylistOp>(op, CustomOperators::CustomopAttrPylist);
  }

  void grow(poplar::program::Sequence &prog) const final {
    insert(outId(0), cloneNcopy(prog, getInTensor(0)));
  }
};

static popx::OpxCreator<CustomopAttrPylistOpx>
    CustomopAttrPylistOpxCreator(CustomOperators::CustomopAttrPylist);
