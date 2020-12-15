// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/op.hpp>
#include <popart/opdebuginfo.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensorindex.hpp>

namespace {
using namespace popart;
class ProfileValueOpSerialiser : public OpSerialiserBase {
public:
  ProfileValue::Map map;

  ProfileValueOpSerialiser()          = default;
  virtual ~ProfileValueOpSerialiser() = default;

  void appendAttribute(const std::string &name, float value) {
    map.insert({name, std::to_string(value)});
  }
  void appendAttribute(const std::string &name, int value) {
    map.insert({name, std::to_string(value)});
  }
  void appendAttribute(const std::string &name, int64_t value) {
    map.insert({name, std::to_string(value)});
  }
  void appendAttribute(const std::string &name, uint32_t value) {
    map.insert({name, std::to_string(value)});
  }
  void appendAttribute(const std::string &name, uint64_t value) {
    map.insert({name, std::to_string(value)});
  }
  void appendAttribute(const std::string &name, const std::string &value) {
    map.insert({name, value});
  }
  void appendAttribute(const std::string &name,
                       const std::vector<int64_t> &value) {
    std::stringstream ss;
    ss << value;
    map.insert({name, ss.str()});
  }
  void appendAttribute(const std::string &name, const Scope &value) {
    map.insert({name, value.str()});
  }
  void appendAttribute(const std::string &name, bool value) {
    map.insert({name, value ? "true" : "false"});
  }

  virtual void appendAttribute(const std::string &name,
                               nonstd::optional<int64_t> value) {
    if (value) {
      map.insert({name, std::to_string(*value)});
    }
  }
  virtual void appendAttribute(const std::string &name,
                               nonstd::optional<float> value) {
    if (value) {
      map.insert({name, std::to_string(*value)});
    }
  }
  virtual void appendAttribute(const std::string &,
                               const std::map<TensorId, uint64_t>) {}

  virtual void appendForwardOp(const Op *) {}

private:
  virtual void appendStrAttr(const std::string &name,
                             const std::string &value) {
    map.insert({name, value});
  }
};

} // namespace

namespace popart {

OpDebugInfo::OpDebugInfo(const DebugContext &debugContext, const Op &_op)
    : DebugInfo(debugContext, "popart"), op(_op) {
  setValue("category", ProfileValue{"op"});
  setValue("instanceId", std::to_string(op.id));

  std::stringstream ss;
  ss << op.opid;
  setValue("opid", ss.str());
}

OpDebugInfo::~OpDebugInfo() {
  if (updateCalled == false) {
    logging::warn("SetupComplete not called for OpDebugInfo");
    setValue("discard", 1);
  } else {
    setValue("discard", 0);
  }
}

void OpDebugInfo::update() {

  updateCalled = true;

  ProfileValue::Vector inputs;
  if (op.input) {
    for (auto t : op.input->tensorMap()) {
      if (t.second) {
        std::string n = t.second->str();
        inputs.push_back(n);
      }
    }
  }
  setValue("inputs", inputs);

  ProfileValue::Vector outputs;
  if (op.output) {
    for (auto t : op.output->tensorMap()) {
      if (t.second) {
        std::string n = t.second->id;
        outputs.push_back(n);
      }
    }
  }
  setValue("outputs", outputs);

  ProfileValueOpSerialiser attributes;
  op.appendAttributes(attributes);
  setValue("attributes", attributes.map);
  setValue("graphId", op.settings.graph.get().getGraphId());
}

} // namespace popart