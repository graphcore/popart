#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <poponnx/builder.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/error.hpp>
#include <poponnx/graphtransformer.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/numerics.hpp>
#include <poponnx/op/l1.hpp>
#include <poponnx/op/loss.hpp>
#include <poponnx/op/nll.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/optimizer.hpp>
#include <poponnx/optionflags.hpp>
#include <poponnx/patterns/patterns.hpp>
#include <poponnx/session.hpp>
#include <poponnx/tensordata.hpp>
#include <poponnx/tensornames.hpp>
#include <poponnx/version.hpp>

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>
#include <stdexcept>

#include <onnx/onnx_pb.h>

namespace py = pybind11;
using namespace poponnx;

std::map<std::string, DataType> initNpTypeMap() {
  std::map<std::string, DataType> M;
  // see tensorinfo.hpp for the complete list of
  // DataTypes (defined originally in ONNX)
  M["float16"] = DataType::FLOAT16;
  M["float32"] = DataType::FLOAT;
  M["uint8"]   = DataType::UINT8;
  M["uint16"]  = DataType::UINT16;
  M["uint32"]  = DataType::UINT32;
  M["uint64"]  = DataType::UINT64;
  M["int8"]    = DataType::INT8;
  M["int16"]   = DataType::INT16;
  M["int32"]   = DataType::INT32;
  M["int64"]   = DataType::INT64;
  M["bool"]    = DataType::BOOL;
  return M;
}

DataType getDataTypeFromNpType(std::string npType) {
  const static std::map<std::string, DataType> M = initNpTypeMap();
  auto found                                     = M.find(npType);
  if (found == M.end()) {
    throw error("No numpy type {} registered in map to DataType", npType);
  }
  return found->second;
}

TensorInfo getTensorInfo(py::array npArr) {
  auto dtype      = npArr.dtype();
  auto typeString = py::str(dtype);
  auto tRank      = npArr.ndim();
  std::vector<int64_t> shape;
  for (int i = 0; i < tRank; ++i) {
    shape.push_back(npArr.shape(i));
  }
  return TensorInfo(getDataTypeFromNpType(typeString), shape);
}

// The following code attempts to convert the python dictionary
// (py::dict) into a map of strings for keys and values. The default
// pybind will attempt to match types
// TODO : This is not very elegant code is there a better way to do
// this?
std::map<std::string, std::string> getDictionary(py::dict pydict) {

  std::map<std::string, std::string> dictionary;
  for (auto element : pydict) {
    std::stringstream key;
    key << element.first;

    std::stringstream value;
    value << element.second;

    dictionary.insert(std::make_pair(key.str(), value.str()));
  }
  return dictionary;
}
class PyStepIO : public IStepIO {
public:
  PyStepIO(std::map<TensorId, py::array> inputs_,
           std::map<TensorId, py::array> outputs_)
      : inputs(inputs_), outputs(outputs_) {}

  template <typename T>
  T get(TensorId id,
        const std::map<TensorId, py::array> &M,
        std::string mapName) const {
    auto found = M.find(id);
    if (found == M.end()) {
      throw error("No tensor {} provided in PyStepIO's {}", id, mapName);
    }
    py::array npArr = found->second;
    T stepData;
    stepData.data = npArr.request().ptr;
    stepData.info = getTensorInfo(npArr);
    return stepData;
  }

  ConstVoidData in(TensorId id) const final {
    return get<ConstVoidData>(id, inputs, "inputs");
  }

  MutableVoidData out(TensorId id) const final {
    return get<MutableVoidData>(id, outputs, "outputs");
  }

private:
  std::map<TensorId, py::array> inputs;
  std::map<TensorId, py::array> outputs;
};

class PyWeightsIO : public IWeightsIO {
public:
  PyWeightsIO(std::map<TensorId, py::array> weights_) : weights(weights_) {}

  template <typename T>
  T get(TensorId id,
        const std::map<TensorId, py::array> &M,
        std::string mapName) const {
    auto found = M.find(id);
    if (found == M.end()) {
      throw error("No tensor {} provided in PyWeightsIO's {}", id, mapName);
    }
    py::array npArr = found->second;
    T stepData;
    stepData.data = npArr.request().ptr;
    stepData.info = getTensorInfo(npArr);
    return stepData;
  }

  bool contains(TensorId id) const final {
    return weights.find(id) != weights.end();
  }

  MutableVoidData weight(TensorId id) const final {
    return get<MutableVoidData>(id, weights, "weights");
  }

private:
  std::map<TensorId, py::array> weights;
};

class AttributeContextManager {
  Builder &builder;
  std::string attribute;
  uint64_t index;

public:
  AttributeContextManager(Builder &_builder,
                          const std::string &_attribute,
                          int64_t _i)
      : builder(_builder), attribute(_attribute), index(_i) {}

  void enter() { builder.setAttribute(sVirtualGraphAttribute, index); }
  void exit() { builder.clearAttribute(sVirtualGraphAttribute); }
};

struct PrepareDeviceError {
  bool success = true;
  std::unique_ptr<poponnx::memory_allocation_err> exception;

  virtual ~PrepareDeviceError() {}

  virtual bool isSuccessful() const { return success; }
  std::string what() const { return exception->what(); }
  std::string getSummaryReport() const { return exception->getSummaryReport(); }
  std::string getGraphReport(bool use_cbor) const {
    return exception->getGraphReport(use_cbor);
  }
};

class NameContextManager {
  Builder &builder;
  std::string name;

public:
  NameContextManager(Builder &_builder, const std::string &_name)
      : builder(_builder), name(_name) {}

  void enter() { builder.pushNameScope(name); }
  void exit() { builder.popNameScope(); }
};

// Create a logging interface for poponnx that is similar to python logging
// module

class Logger {

  std::string name;

  Logger(const std::string &name_) : name(name_) {}

public:
  static Logger getLogger(const std::string &name = "all") {
    return Logger(name);
  }

  void setLevel(const std::string &level) {
    logging::configure({{name, level}});
  }

  void debug(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Debug, std::move(info));
  }

  void info(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Info, std::move(info));
  }

  void warn(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Warn, std::move(info));
  }

  void error(const std::string &info) {
    logging::log(logging::Module::python, logging::Level::Err, std::move(info));
  }

  void critical(const std::string &info) {
    logging::log(
        logging::Module::python, logging::Level::Critical, std::move(info));
  }
};

// The following code allow boost optional to be used in the C++ interface and
// map to python types
namespace pybind11 {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};
} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(poponnx_core, m) {
  m.doc() = "binding for C++ poponnx library";

  m.def("getTensorInfo", &getTensorInfo);

  m.def("getLogger", &Logger::getLogger, py::arg("name") = "all");

  m.def("versionString", &poponnx::core::versionString);
  m.def("packageHash", &poponnx::core::packageHash);

  py::class_<Logger>(m, "Logger")
      .def("setLevel", &Logger::setLevel)
      .def("debug", &Logger::debug)
      .def("info", &Logger::info)
      .def("warn", &Logger::warn)
      .def("error", &Logger::error)
      .def("critical", &Logger::critical);

  py::class_<OperatorIdentifier>(m, "OperatorIdentifier")
      .def(py::init<const std::string &, const std::string &, unsigned>(),
           py::arg("domain"),
           py::arg("type"),
           py::arg("version"))
      .def_readonly("domain", &OperatorIdentifier::domain)
      .def_readonly("type", &OperatorIdentifier::type)
      .def_readonly("version", &OperatorIdentifier::version);

  m.def("getSupportedOperations",
        &OpManager::getSupportedOperations,
        py::arg("includeInternal"));

  py::class_<IStepIO> stepio(m, "IStepIO");

  py::class_<IWeightsIO> weightsio(m, "IWeightsIO");

  py::enum_<AnchorReturnTypeId>(m, "AnchorReturnTypeId")
      .value("FINAL", AnchorReturnTypeId::FINAL)
      .value("EVERYN", AnchorReturnTypeId::EVERYN)
      .value("ALL", AnchorReturnTypeId::ALL);

  py::class_<PyStepIO>(m, "PyStepIO", stepio)
      .def(py::init<std::map<TensorId, py::array>,
                    std::map<TensorId, py::array>>(),
           py::arg("inputs"),
           py::arg("outputs"));

  py::class_<PyWeightsIO>(m, "PyWeightsIO", weightsio)
      .def(py::init<std::map<TensorId, py::array>>(), py::arg("weights"));

  py::class_<AnchorReturnType>(m, "AnchorReturnType")
      .def(py::init<std::string>(), py::arg("anchorReturnTypeString"))
      .def(py::init<std::string, int>(),
           py::arg("anchorReturnTypeString"),
           py::arg("returnPeriod"))
      .def("id", &AnchorReturnType::id)
      .def("rp", &AnchorReturnType::rp);

  py::class_<DataFlow>(m, "DataFlow")
      .def(py::init<int, const std::map<TensorId, AnchorReturnType> &>(),
           py::arg("batchesPerStep"),
           py::arg("anchorTensors"))
      .def("isAnchored", &DataFlow::isAnchored)
      .def("nAnchors", &DataFlow::nAnchors)
      .def("batchesPerStep", &DataFlow::batchesPerStep)
      .def("anchors", &DataFlow::anchors, pybind11::return_value_policy::copy)
      .def("art", &DataFlow::art);

  py::class_<TensorInfo>(m, "TensorInfoCore")
      .def(py::init<std::string, const std::vector<int64_t> &>(),
           py::arg("dataType"),
           py::arg("shape"))
      .def("data_type_lcase", &TensorInfo::data_type_lcase)
      .def("shape", &TensorInfo::shape);

  py::class_<numerics::NumericsReport>(m, "NumericsReport")
      .def(py::init<std::string, std::string, std::string, std::string>(),
           py::arg("A0"),
           py::arg("A1"),
           py::arg("B0"),
           py::arg("B1"))
      .def("report", &numerics::NumericsReport::report)
      .def("fullReport", &numerics::NumericsReport::fullReport)
      .def("getRelativeErrors", &numerics::NumericsReport::getRelativeErrors);

  py::class_<InputShapeInfo>(m, "InputShapeInfo")
      .def(py::init<>())
      .def("add", &InputShapeInfo::add)
      .def("get", &InputShapeInfo::get)
      .def("has", &InputShapeInfo::has);

  py::class_<Loss> loss(m, "Loss");
  loss.def("input", &Loss::input);
  loss.def("output", &Loss::output);

  py::enum_<ReductionType>(m, "ReductionType")
      .value("Sum", ReductionType::SUM)
      .value("Mean", ReductionType::MEAN);

  py::class_<NllLoss>(m, "NllLoss", loss)
      .def(py::init<TensorId, TensorId, TensorId, ReductionType>(),
           py::arg("probabilities"),
           py::arg("labels"),
           py::arg("output"),
           py::arg("reduction") = ReductionType::SUM)
      .def(py::init<TensorId, TensorId, TensorId, int, ReductionType>(),
           py::arg("probabilities"),
           py::arg("labels"),
           py::arg("output"),
           py::arg("ignore_index"),
           py::arg("reduction") = ReductionType::SUM)
      .def("probsTensorId", &NllLoss::probsTensorId)
      .def("labelTensorId", &NllLoss::labelTensorId)
      .def("virtualGraph", &NllLoss::virtualGraph);

  py::class_<L1Loss>(m, "L1Loss", loss)
      .def(py::init<TensorId, TensorId, float, ReductionType>(),
           py::arg("input"),
           py::arg("output"),
           py::arg("lambda"),
           py::arg("reduction") = ReductionType::SUM)
      .def("getInputId", &L1Loss::getInputId)
      .def("getLambda", &L1Loss::getLambda)
      .def("virtualGraph", &L1Loss::virtualGraph);

  py::class_<Optimizer> optimizer(m, "Optimizer");

  py::class_<BaseSGD> basesgd(m, "BaseSGD", optimizer);
  basesgd.def("learnRate", &BaseSGD::learnRate);
  basesgd.def("weightDecay", &BaseSGD::weightDecay);

  py::class_<SGD>(m, "SGD", basesgd)
      .def(py::init<float>(), py::arg("learning_rate"))
      .def(py::init<float, float>(),
           py::arg("learning_rate"),
           py::arg("weight_decay"));

  py::class_<ConstSGD>(m, "ConstSGD", basesgd)
      .def(py::init<float>(), py::arg("learning_rate"))
      .def(py::init<float, float>(),
           py::arg("learning_rate"),
           py::arg("weight_decay"));

  py::class_<SessionOptions>(m, "SessionOptionsCore")
      .def(py::init<>())
      .def_readwrite("logDir", &SessionOptions::logDir)
      .def_readwrite("exportPoplarComputationGraph",
                     &SessionOptions::exportPoplarComputationGraph)
      .def_readwrite("exportPoplarVertexGraph",
                     &SessionOptions::exportPoplarVertexGraph)
      .def_readwrite("ignoreData", &SessionOptions::ignoreData)
      .def_readwrite("enableConvolutionGraphCaching",
                     &SessionOptions::enableConvolutionGraphCaching)
      .def_readwrite("enableOutlining", &SessionOptions::enableOutlining)
      .def_readwrite("enableOutliningCopyCostPruning",
                     &SessionOptions::enableOutliningCopyCostPruning)
      .def_readwrite("outlineThreshold", &SessionOptions::outlineThreshold)
      .def_readwrite("accumulationFactor", &SessionOptions::accumulationFactor)
      .def_readwrite("enableGradientAccumulation",
                     &SessionOptions::enableGradientAccumulation)
      .def_readwrite("enableNonStableSoftmax",
                     &SessionOptions::enableNonStableSoftmax)
      .def_readwrite("enablePipelining", &SessionOptions::enablePipelining)
      .def_readwrite("autoRecomputation", &SessionOptions::autoRecomputation)
      .def_readwrite("mergeVarUpdate", &SessionOptions::mergeVarUpdate)
      .def_readwrite("mergeVarUpdateMemThreshold",
                     &SessionOptions::mergeVarUpdateMemThreshold)
      .def_readwrite("rearrangeAnchorsOnHost",
                     &SessionOptions::rearrangeAnchorsOnHost)
      .def_readwrite("enableVirtualGraphs",
                     &SessionOptions::enableVirtualGraphs)
      .def_readwrite("autoVirtualGraph", &SessionOptions::autoVirtualGraph)
      .def_readwrite("enableReplicatedGraphs",
                     &SessionOptions::enableReplicatedGraphs)
      .def_readwrite("replicatedGraphCount",
                     &SessionOptions::replicatedGraphCount)
      .def_readwrite("compileEngine", &SessionOptions::compileEngine)
      .def_readwrite("engineOptions", &SessionOptions::engineOptions)
      .def_readwrite("convolutionOptions", &SessionOptions::convolutionOptions)
      .def_readwrite("reportOptions", &SessionOptions::reportOptions)
      .def_readwrite("dotOpNames", &SessionOptions::dotOpNames)
      .def_readwrite("separateCallOpPdfs", &SessionOptions::separateCallOpPdfs)
      .def_readwrite("finalDotOp", &SessionOptions::finalDotOp)
      .def_readwrite("firstDotOp", &SessionOptions::firstDotOp)
      .def_readwrite("constantWeights", &SessionOptions::constantWeights)
      .def_readwrite("cachePath", &SessionOptions::cachePath)
      .def_readwrite("enableEngineCaching",
                     &SessionOptions::enableEngineCaching)
      .def_readwrite("enableFloatingPointChecks",
                     &SessionOptions::enableFloatingPointChecks)
      .def_readwrite("enableStochasticRounding",
                     &SessionOptions::enableStochasticRounding)
      // set in python use the python set constructor, so something like
      // mySessionOptions.dotChecks = {poponnx.DotCheck.FINAL}
      .def_readwrite("dotChecks", &SessionOptions::dotChecks);

  py::enum_<PatternsLevel>(m, "PatternsLevel")
      .value("ALL", PatternsLevel::ALL)
      .value("DEFAULT", PatternsLevel::DEFAULT)
      .value("NONE", PatternsLevel::NONE);

  py::enum_<DotCheck>(m, "DotCheck")
      .value("FWD0", DotCheck::FWD0)
      .value("FWD1", DotCheck::FWD1)
      .value("BWD0", DotCheck::BWD0)
      .value("PREALIAS", DotCheck::PREALIAS)
      .value("FINAL", DotCheck::FINAL);

  py::enum_<RecomputationType>(m, "RecomputationType")
      .value("None", RecomputationType::None)
      .value("Standard", RecomputationType::Standard)
      .value("NormOnly", RecomputationType::NormOnly);

  py::enum_<MergeVarUpdateType>(m, "MergeVarUpdateType")
      .value("Off", MergeVarUpdateType::None)
      .value("All", MergeVarUpdateType::All)
      .value("AutoTight", MergeVarUpdateType::AutoTight)
      .value("AutoLoose", MergeVarUpdateType::AutoLoose);

  py::enum_<PreAliasPatternType>(m, "PreAliasPatternType")
      .value("PREUNIREPL", PreAliasPatternType::PREUNIREPL)
      .value("POSTNREPL", PreAliasPatternType::POSTNREPL)
      .value("SOFTMAXGRADDIRECT", PreAliasPatternType::SOFTMAXGRADDIRECT)
      .value("NLLLWITHSOFTMAXGRADDIRECT",
             PreAliasPatternType::NLLLWITHSOFTMAXGRADDIRECT)
      .value("SPLITCONVBIAS", PreAliasPatternType::SPLITCONVBIAS)
      .value("OPTOIDENTITY", PreAliasPatternType::OPTOIDENTITY)
      .value("SUBTRACTARG1GRADOP", PreAliasPatternType::SUBTRACTARG1GRADOP)
      .value("MULARGGRADOP", PreAliasPatternType::MULARGGRADOP)
      .value("RECIPROCALGRADOP", PreAliasPatternType::RECIPROCALGRADOP)
      .value("SINGRADOP", PreAliasPatternType::SINGRADOP)
      .value("COSGRADOP", PreAliasPatternType::COSGRADOP)
      .value("TANTOSINOVERCOS", PreAliasPatternType::TANTOSINOVERCOS)
      .value("DIVARG0GRADOP", PreAliasPatternType::DIVARG0GRADOP)
      .value("DIVARG1GRADOP", PreAliasPatternType::DIVARG1GRADOP)
      .value("POWARG0GRADOP", PreAliasPatternType::POWARG0GRADOP)
      .value("POWARG1GRADOP", PreAliasPatternType::POWARG1GRADOP)
      .value("SQRTGRADOP", PreAliasPatternType::SQRTGRADOP)
      .value("EXPGRADOP", PreAliasPatternType::EXPGRADOP)
      .value("GEMMDECOMPOSITION", PreAliasPatternType::GEMMDECOMPOSITION)
      .value("NEGATIVEONESCALE", PreAliasPatternType::NEGATIVEONESCALE);

  py::class_<Patterns>(m, "Patterns")
      .def(py::init<>())
      .def(py::init<PatternsLevel>())
      .def(py::init<std::vector<PreAliasPatternType>>())
      .def(py::init(
          [](std::vector<std::string> l) { return Patterns::create(l); }))
      .def_property("PreUniRepl",
                    &Patterns::isPreUniReplEnabled,
                    &Patterns::enablePreUniRepl)
      .def_property("PostNRepl",
                    &Patterns::isPostNReplEnabled,
                    &Patterns::enablePostNRepl)
      .def_property("SoftMaxGradDirect",
                    &Patterns::isSoftMaxGradDirectEnabled,
                    &Patterns::enableSoftMaxGradDirect)
      .def_property("NlllWithSoftMaxGradDirect",
                    &Patterns::isNlllWithSoftMaxGradDirectEnabled,
                    &Patterns::enableNlllWithSoftMaxGradDirect)
      .def_property("SplitConvBias",
                    &Patterns::isSplitConvBiasEnabled,
                    &Patterns::enableSplitConvBias)
      .def_property("OpToIdentity",
                    &Patterns::isOpToIdentityEnabled,
                    &Patterns::enableOpToIdentity)
      .def_property("SubtractArg1GradOp",
                    &Patterns::isSubtractArg1GradOpEnabled,
                    &Patterns::enableSubtractArg1GradOp)
      .def_property("MulArgGradOp",
                    &Patterns::isMulArgGradOpEnabled,
                    &Patterns::enableMulArgGradOp)
      .def_property(
          "InPlace", &Patterns::isInPlaceEnabled, &Patterns::enableInPlace)
      .def("__repr__", [](const Patterns &p) {
        std::stringstream ss;
        ss << p;
        return ss.str();
      });

  py::class_<PrepareDeviceError>(m, "PrepareDeviceError")
      .def(py::init<>())
      .def("__repr__", &PrepareDeviceError::what)
      .def("isSuccessful", &PrepareDeviceError::isSuccessful)
      .def("getSummaryReport", &PrepareDeviceError::getSummaryReport)
      .def(
          "getGraphReport",
          [](const PrepareDeviceError &error, bool use_cbor) {
            auto report = error.getGraphReport(use_cbor);
            return py::bytes(report);
          },
          py::arg("use_cbor") = false);

  py::class_<InferenceSession>(m, "InferenceSessionCore")
      .def(py::init(&InferenceSession::createFromOnnxModel),
           py::arg("model"),
           py::arg("dataFlow").none(),
           py::arg("deviceInfo"),
           py::arg("losses"),
           py::arg("inputShapeInfo"),
           py::arg("userOptions"),
           py::arg("patterns"))
      .def(
          "prepareDevice",
          [](InferenceSession &session, PrepareDeviceError *status) {
            try {
              session.prepareDevice();
            } catch (const poponnx::memory_allocation_err &e) {
              if (status != nullptr) {
                status->exception = e.clone();
                status->success   = false;
              } else {
                // rethrow the exception
                throw;
              }
            }
          },
          py::arg("err").none())
      .def("weightsFromHost", &InferenceSession::weightsFromHost)
      .def("writeWeights", &TrainingSession::writeWeights)
      .def("run", &InferenceSession::run)
      .def("modelToHost", &InferenceSession::modelToHost)
      .def("getInfo", &InferenceSession::getInfo)
      .def("getSummaryReport", &InferenceSession::getSummaryReport)
      .def(
          "getGraphReport",
          [](const InferenceSession &session, bool use_cbor) {
            auto report = session.getGraphReport(use_cbor);
            return py::bytes(report);
          },
          py::arg("use_cbor") = false)
      .def(
          "getExecutionReport",
          [](const InferenceSession &session, bool use_cbor) {
            auto report = session.getExecutionReport(use_cbor);
            return py::bytes(report);
          },
          py::arg("use_cbor") = false)
      .def("getSerializedGraph",
           [](const InferenceSession &session) {
             auto report = session.getSerializedGraph();
             return py::bytes(report);
           })
      .def("getTensorTileMap", &InferenceSession::getTensorTileMap)
      .def("resetHostWeights", &InferenceSession::resetHostWeights);

  py::class_<TrainingSession>(m, "TrainingSessionCore")
      .def(py::init(&TrainingSession::createFromOnnxModel),
           py::arg("model"),
           py::arg("dataFlow").none(),
           py::arg("losses"),
           py::arg("optimizer"),
           py::arg("deviceInfo"),
           py::arg("inputShapeInfo"),
           py::arg("userOptions"),
           py::arg("patterns"))
      .def("updateOptimizer", &TrainingSession::updateOptimizer)
      .def(
          "prepareDevice",
          [](TrainingSession &session, PrepareDeviceError *status) {
            try {
              session.prepareDevice();
            } catch (const poponnx::memory_allocation_err &e) {
              if (status != nullptr) {
                status->exception = e.clone();
                status->success   = false;
              } else {
                // rethrow the exception
                throw;
              }
            }
          },
          py::arg("err").none())
      .def("weightsToHost", &TrainingSession::weightsToHost)
      .def("weightsFromHost", &TrainingSession::weightsFromHost)
      .def("readWeights", &TrainingSession::readWeights)
      .def("writeWeights", &TrainingSession::writeWeights)
      .def("optimizerFromHost", &TrainingSession::optimizerFromHost)
      .def("run", &TrainingSession::run)
      .def("modelToHost", &TrainingSession::modelToHost)
      .def("getInfo", &TrainingSession::getInfo)
      .def("getSummaryReport", &TrainingSession::getSummaryReport)
      .def(
          "getGraphReport",
          [](const TrainingSession &session, bool use_cbor) {
            auto report = session.getGraphReport(use_cbor);
            return py::bytes(report);
          },
          py::arg("use_cbor") = false)
      .def(
          "getExecutionReport",
          [](const TrainingSession &session, bool use_cbor) {
            auto report = session.getExecutionReport(use_cbor);
            return py::bytes(report);
          },
          py::arg("use_cbor") = false)
      .def("getSerializedGraph",
           [](const TrainingSession &session) {
             auto report = session.getSerializedGraph();
             return py::bytes(report);
           })
      .def("getTensorTileMap", &TrainingSession::getTensorTileMap)
      .def("resetHostWeights", &TrainingSession::resetHostWeights);

  py::class_<GraphTransformer>(m, "GraphTransformer")
      .def(py::init<const std::string &>(), py::arg("modelProtoOrFilename"))
      .def("getModelProto",
           [](const GraphTransformer &graphtransformer) {
             return py::bytes(graphtransformer.getModelProto());
           })

      .def("removeUnusedInputs", &GraphTransformer::removeUnusedInputs)
      .def("prepareNodesForTraining",
           &GraphTransformer::prepareNodesForTraining)
      .def("convertFloatsToHalfs", &GraphTransformer::convertFloatsToHalfs)
      .def("convertInitializersToConstants",
           &GraphTransformer::convertInitializersToConstants,
           py::arg("ids"))
      .def("convertAllFixedPointInitializersToConstants",
           &GraphTransformer::convertAllFixedPointInitializersToConstants);

// Include the generated poponx.cpp code
#include "poponnx.cpp.gen"

  py::class_<AiGraphcoreOpset1>(m, "AiGraphcoreOpset1")
      .def("groupnormalization",
           &AiGraphcoreOpset1::groupnormalization,
           py::arg("args"),
           py::arg("num_groups"),
           py::arg("epsilon")     = 1e-05f,
           py::arg("debugPrefix") = std::string())
      .def("printtensor",
           &AiGraphcoreOpset1::printtensor,
           py::arg("args"),
           py::arg("print_gradient") = 1,
           py::arg("debugPrefix")    = std::string())
      .def("subsample",
           &AiGraphcoreOpset1::subsample,
           py::arg("args"),
           py::arg("strides"),
           py::arg("debugPrefix") = std::string());

  py::class_<Builder>(m, "BuilderCore")
      .def(py::init(&Builder::create))
      .def(py::init(&Builder::createFromOnnxModel),
           py::arg("modelProtoOrFilename"))
      .def("addInputTensor",
           &Builder::addInputTensor,
           py::arg("tensorInfo"),
           py::arg("debugPrefix") = std::string())
      .def("addInputTensorFromParentGraph",
           &Builder::addInputTensorFromHigherScope,
           py::arg("tensorId"))
      .def(
          "addInitializedInputTensor",
          [](Builder &builder, py::array array, std::string &debugPrefix) {
            ConstVoidData initData;
            initData.data = array.request().ptr;
            initData.info = getTensorInfo(array);
            return builder.addInitializedInputTensor(initData, debugPrefix);
          },
          py::arg("initVal"),
          py::arg("debugPrefix") = std::string())
      .def("addOutputTensor", &Builder::addOutputTensor, py::arg("outputName"))

      // Accessors for the ai.onnx domain builder interfac
      .def_property_readonly("aiOnnxOpset6", &Builder::aiOnnxOpset6)
      .def_property_readonly("aiOnnxOpset7", &Builder::aiOnnxOpset7)
      .def_property_readonly("aiOnnxOpset8", &Builder::aiOnnxOpset8)
      .def_property_readonly("aiOnnxOpset9", &Builder::aiOnnxOpset9)
      .def_property_readonly("aiOnnxOpset10", &Builder::aiOnnxOpset10)

      // Accessors for the ai.graphcore domain builder interface
      .def_property_readonly("aiGraphcoreOpset1", &Builder::aiGraphcoreOpset1)

      .def("addNodeAttribute",
           static_cast<void (Builder::*)(const std::string &,
                                         const int64_t &,
                                         const std::set<TensorId> &)>(
               &Builder::addNodeAttribute),
           py::arg("attributeName"),
           py::arg("attributeValue"),
           py::arg("nodeOutputNames"))
      .def("addNodeAttribute",
           static_cast<void (Builder::*)(const std::string &,
                                         const std::vector<int64_t> &,
                                         const std::set<TensorId> &)>(
               &Builder::addNodeAttribute),
           py::arg("attributeName"),
           py::arg("attributeValue"),
           py::arg("nodeOutputNames"))
      .def("addNodeAttribute",
           static_cast<void (Builder::*)(
               const std::string &, const float &, const std::set<TensorId> &)>(
               &Builder::addNodeAttribute),
           py::arg("attributeName"),
           py::arg("attributeValue"),
           py::arg("nodeOutputNames"))
      .def("addNodeAttribute",
           static_cast<void (Builder::*)(const std::string &,
                                         const std::vector<float> &,
                                         const std::set<TensorId> &)>(
               &Builder::addNodeAttribute),
           py::arg("attributeName"),
           py::arg("attributeValue"),
           py::arg("nodeOutputNames"))
      .def("addNodeAttribute",
           static_cast<void (Builder::*)(const std::string &,
                                         const std::string &,
                                         const std::set<TensorId> &)>(
               &Builder::addNodeAttribute),
           py::arg("attributeName"),
           py::arg("attributeValue"),
           py::arg("nodeOutputNames"))
      .def("addNodeAttribute",
           static_cast<void (Builder::*)(const std::string &,
                                         const std::vector<std::string> &,
                                         const std::set<TensorId> &)>(
               &Builder::addNodeAttribute),
           py::arg("attributeName"),
           py::arg("attributeValue"),
           py::arg("nodeOutputNames"))
      .def("nodeHasAttribute",
           &Builder::nodeHasAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getInt64NodeAttribute",
           &Builder::getInt64NodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getInt64VectorNodeAttribute",
           &Builder::getInt64VectorNodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getFloatNodeAttribute",
           &Builder::getFloatNodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getFloatVectorNodeAttribute",
           &Builder::getFloatVectorNodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getStringNodeAttribute",
           &Builder::getStringNodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getStringVectorNodeAttribute",
           &Builder::getStringVectorNodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("removeNodeAttribute",
           &Builder::removeNodeAttribute,
           py::arg("attributeName"),
           py::arg("nodeOutputNames"))
      .def("getAllNodeAttributeNames",
           &Builder::getAllNodeAttributeNames,
           py::arg("nodeOutputNames"))
      .def("getModelProto",
           [](const Builder &builder) {
             return py::bytes(builder.getModelProto());
           })
      .def("getInputTensorIds", &Builder::getInputTensorIds)
      .def("getOutputTensorIds", &Builder::getOutputTensorIds)
      .def("getValueTensorIds", &Builder::getValueTensorIds)
      .def("getTensorShape", &Builder::getTensorShape, py::arg("id"))
      .def("virtualGraph",
           static_cast<void (Builder::*)(const TensorId &, int64_t value)>(
               &Builder::virtualGraph),
           py::arg("nodeOutputNames"),
           py::arg("value") = 0)
      .def(
          "virtualGraph",
          [](Builder &self, int64_t index) -> AttributeContextManager {
            AttributeContextManager acm(self, sVirtualGraphAttribute, index);
            return acm;
          },
          py::arg("value"))
      .def(
          "nameScope",
          [](Builder &self, const std::string &name) -> NameContextManager {
            NameContextManager ncm(self, name);
            return ncm;
          },
          py::arg("name"))
      .def("getVirtualGraph",
           static_cast<int64_t (Builder::*)(const TensorId &)>(
               &Builder::getVirtualGraph),
           py::arg("nodeOutputNames"))

      .def("recomputeOutputInBackwardPass",
           static_cast<void (Builder::*)(const TensorId &, bool value)>(
               &Builder::recomputeOutputInBackwardPass),
           py::arg("nodeOutputName"),
           py::arg("value") = true)
      .def("recomputeOutputInBackwardPass",
           static_cast<void (Builder::*)(const std::set<TensorId> &,
                                         bool value)>(
               &Builder::recomputeOutputInBackwardPass),
           py::arg("nodeOutputNames"),
           py::arg("value") = true)

      .def("getRecomputeOutputInBackwardPass",
           static_cast<bool (Builder::*)(const TensorId &)>(
               &Builder::getRecomputeOutputInBackwardPass),
           py::arg("nodeOutputName"))

      .def("getRecomputeOutputInBackwardPass",
           static_cast<bool (Builder::*)(const std::set<TensorId> &)>(
               &Builder::getRecomputeOutputInBackwardPass),
           py::arg("nodeOutputNames"))

      .def("setInplacePreferences",
           static_cast<void (Builder::*)(const TensorId &,
                                         const std::map<OpType, float> &)>(
               &Builder::setInplacePreferences),
           py::arg("nodeOutputName"),
           py::arg("prefs"));

  py::class_<AttributeContextManager>(m, "AttributeContextManager")
      .def("__enter__", &AttributeContextManager::enter)
      .def("__exit__",
           [](AttributeContextManager &self, void *, void *, void *) {
             self.exit();
           });

  py::class_<NameContextManager>(m, "NameContextManager")
      .def("__enter__", &NameContextManager::enter)
      .def("__exit__", [](NameContextManager &self, void *, void *, void *) {
        self.exit();
      });

  // PyBinding to a singleton
  py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>>(
      m, "DeviceManager")
      .def(py::init([]() {
        return std::unique_ptr<DeviceManager, py::nodelete>(
            &DeviceManager::createDeviceManager());
      }))
      .def(
          "acquireAvailableDevice",
          static_cast<std::shared_ptr<DeviceInfo> (DeviceManager::*)(int, int)>(
              &DeviceManager::acquireAvailableDevice),
          py::arg("numIpus")     = 1,
          py::arg("tilesPerIpu") = 0)
      .def(
          "acquireDeviceById", &DeviceManager::acquireDeviceById, py::arg("id"))
      .def("createCpuDevice", &DeviceManager::createCpuDevice)
      .def("createIpuModelDevice",
           [](DeviceManager &dm, py::dict e) {
             std::map<std::string, std::string> options = getDictionary(e);
             return dm.createIpuModelDevice(options);
           })
      .def("createSimDevice",
           [](DeviceManager &dm, py::dict e) {
             std::map<std::string, std::string> options = getDictionary(e);
             return dm.createSimDevice(options);
           })
      .def("enumerateDevices", &DeviceManager::enumerateDevices);

  py::enum_<DeviceType>(m, "DeviceType")
      .value("IpuModel", DeviceType::IpuModel)
      .value("Cpu", DeviceType::Cpu)
      .value("Ipu", DeviceType::Ipu)
      .value("Sim", DeviceType::Sim);

  py::class_<DeviceInfo, std::shared_ptr<DeviceInfo>>(m, "DeviceInfo")
      .def("attach", &DeviceInfo::attach)
      .def("detach", &DeviceInfo::detach)
      .def_property_readonly("type", &DeviceInfo::getType)
      .def_property_readonly("version", &DeviceInfo::getVersion)
      .def_property_readonly("id", &DeviceInfo::getId)
      .def_property_readonly("numIpus", &DeviceInfo::getNumIpus)
      .def_property_readonly("tilesPerIpu", &DeviceInfo::getTilesPerIpu)
      .def_property_readonly("driverIds", &DeviceInfo::getDriverIds)

      .def_property_readonly("numWorkerContexts",
                             &DeviceInfo::getNumWorkerContexts)
      .def("__repr__", [](const DeviceInfo &di) {
        std::stringstream ss;
        ss << di;
        return ss.str();
      });

  m.def("reservedGradientPrefix", &reservedGradientPrefix);
  m.def("reservedUpdatedVarPrefix", &reservedUpdatedVarPrefix);
  m.def("reservedAccumulationPrefix", &reservedAccumulationPrefix);
  m.def("reservedAccumulationOutPrefix", &reservedAccumulationOutPrefix);
  m.def("reservedAccumulationResetPrefix", &reservedAccumulationResetPrefix);
  m.def("reservedStashedPrefix", &reservedStashedPrefix);
  m.def("reservedRestoredPrefix", &reservedRestoredPrefix);

  // Exceptions are processed explicitly to allow the main dynamic library
  // to do the type inference.  This prevents some inter dynamic library type
  // inference issues on OS/X
  static py::exception<poponnx::error> ePoponnx(m, "poponnx_exception");
  static py::exception<poplar::poplar_error> ePoplar(m, "poplar_exception");
  static py::exception<poputil::poplibs_error> ePoplibs(m, "poplibs_exception");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      std::rethrow_exception(p);
    } catch (std::exception &e) {
      switch (poponnx::getErrorSource(e)) {
      case poponnx::ErrorSource::poponnx:
        ePoponnx(e.what());
        return;
      case poponnx::ErrorSource::poplar:
        ePoplar(e.what());
        return;
      case poponnx::ErrorSource::poplibs:
        ePoplibs(e.what());
        return;
      case poponnx::ErrorSource::unknown:
        throw;
      }
    }
  });
}
