// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <builderdebuginfo.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>
#include <popart/vendored/optional.hpp>

namespace {
using namespace popart;
}

namespace popart {

BuilderDebugInfo::BuilderDebugInfo(
    const DebugContext &debugContext,
    const std::string &api,
    const std::vector<TensorId> &inputs,
    const std::map<std::string, popart::any> attributes,
    const std::vector<TensorId> &outputs)
    : DebugInfo(debugContext, "popartbuilder") {

  setValue("category", ProfileValue{"api"});
  setValue("api", ProfileValue{api});

  if (inputs.size() > 0) {
    ProfileValue::Vector inputsPV;
    for (auto input : inputs) {
      inputsPV.push_back(ProfileValue{input});
    }
    setValue("inputs", inputsPV);
  }

  if (outputs.size() > 0) {
    ProfileValue::Vector outputsPV;
    for (auto output : outputs) {
      outputsPV.push_back(ProfileValue{output});
    }
    setValue("outputs", outputsPV);
  }

  ProfileValue::Map args;
  for (auto p : attributes) {

    if (p.second.type() == typeid(int64_t)) {
      int64_t v = popart::any_cast<int64_t>(p.second);
      args.insert({p.first, ProfileValue{std::to_string(v)}});
    } else if (p.second.type() == typeid(uint64_t)) {
      uint64_t v = popart::any_cast<uint64_t>(p.second);
      args.insert({p.first, ProfileValue{std::to_string(v)}});
    } else if (p.second.type() == typeid(float)) {
      float v = popart::any_cast<float>(p.second);
      args.insert({p.first, ProfileValue{std::to_string(v)}});
    } else if (p.second.type() == typeid(unsigned)) {
      unsigned v = popart::any_cast<unsigned>(p.second);
      args.insert({p.first, ProfileValue{std::to_string(v)}});
    } else if (p.second.type() == typeid(std::string)) {
      std::string v = popart::any_cast<std::string>(p.second);
      args.insert({p.first, ProfileValue{v}});
    } else if (p.second.type() == typeid(const std::vector<std::string> &)) {
      auto v = popart::any_cast<const std::vector<std::string> &>(p.second);
      std::stringstream ss;
      ss << v;
      args.insert({p.first, ProfileValue{ss.str()}});
    } else if (p.second.type() == typeid(const std::vector<int64_t> &)) {
      auto v = popart::any_cast<const std::vector<int64_t> &>(p.second);
      std::stringstream ss;
      ss << v;
      args.insert({p.first, ProfileValue{ss.str()}});
    } else if (p.second.type() == typeid(const std::vector<float> &)) {
      auto v = popart::any_cast<const std::vector<float> &>(p.second);
      std::stringstream ss;
      ss << v;
      args.insert({p.first, ProfileValue{ss.str()}});
    } else if (p.second.type() == typeid(nonstd::optional<int64_t>)) {
      auto v = popart::any_cast<nonstd::optional<int64_t>>(p.second);
      if (v) {
        std::stringstream ss;
        ss << *v;
        args.insert({p.first, ProfileValue{ss.str()}});
      }
    } else if (p.second.type() == typeid(nonstd::optional<float>)) {
      auto v = popart::any_cast<nonstd::optional<float>>(p.second);
      if (v) {
        std::stringstream ss;
        ss << *v;
        args.insert({p.first, ProfileValue{ss.str()}});
      }
    } else {
      args.insert({p.first, ProfileValue{"<UNKNOWN>"}});
    }
  }
  setValue("attributes", args);
}

void BuilderDebugInfo::setOutputs(const std::vector<TensorId> &outputs) {
  if (outputs.size() > 0) {
    ProfileValue::Vector outputsPV;
    for (auto output : outputs) {
      outputsPV.push_back(ProfileValue{output});
    }
    setValue("outputs", outputsPV);
  }
}

BuilderVarDebugInfo::BuilderVarDebugInfo(const DebugContext &debugContext,
                                         const std::string &api,
                                         const TensorId &id,
                                         const TensorInfo &ti)
    : DebugInfo(debugContext, "popartbuilder") {
  setValue("category", ProfileValue{"variable"});
  setValue("api", ProfileValue{api});

  setValue("tensorId", id);

  std::stringstream ss;
  ss << ti.shape();
  setValue("shape", ss.str());

  std::string dataType = "<UNKNOWN>";
  switch (ti.dataType()) {
  case popart::DataType::UINT8:
    dataType = "UINT8";
    break;
  case popart::DataType::INT8:
    dataType = "INT8";
    break;
  case popart::DataType::UINT16:
    dataType = "UINT16";
    break;
  case popart::DataType::INT16:
    dataType = "INT16";
    break;
  case popart::DataType::INT32:
    dataType = "INT32";
    break;
  case popart::DataType::INT64:
    dataType = "INT64";
    break;
  case popart::DataType::UINT32:
    dataType = "UINT32";
    break;
  case popart::DataType::UINT64:
    dataType = "UINT64";
    break;
  case popart::DataType::BOOL:
    dataType = "BOOL";
    break;
  case popart::DataType::FLOAT:
    dataType = "FLOAT";
    break;
  case popart::DataType::FLOAT16:
    dataType = "FLOAT16";
    break;
  case popart::DataType::BFLOAT16:
    dataType = "BFLOAT16";
    break;
  case popart::DataType::DOUBLE:
    dataType = "DOUBLE";
    break;
  case popart::DataType::COMPLEX64:
    dataType = "COMPLEX64";
    break;
  case popart::DataType::COMPLEX128:
    dataType = "COMPLEX128";
    break;
  case popart::DataType::STRING:
    dataType = "STRING";
    break;
  case popart::DataType::UNDEFINED:
    dataType = "UNDEFINED";
    break;
  }
  setValue("type", dataType);
}
BuilderVarDebugInfo::BuilderVarDebugInfo(const DebugContext &debugContext,
                                         const std::string &api,
                                         const TensorId &id)
    : DebugInfo(debugContext, "popartbuilder") {
  setValue("category", ProfileValue{"variable"});
  setValue("api", ProfileValue{api});
  setValue("tensorId", id);
}

} // namespace popart