#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <poponnx/builder.hpp>
#include <poponnx/device.hpp>
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

#include <poplar/exceptions.hpp>
#include <poputil/exceptions.hpp>
#include <stdexcept>

namespace py = pybind11;
using namespace poponnx;

std::map<std::string, DataType> initNpTypeMap() {
  std::map<std::string, DataType> M;
  // see tensorinfo.hpp for the complete list of
  // DataTypes (defined originally in ONNX)
  M["float16"] = DataType::FLOAT16;
  M["float32"] = DataType::FLOAT;
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

// The follow code attempts to convert the python dictionary
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

PYBIND11_MODULE(poponnx_core, m) {
  m.doc() = "binding for C++ poponnx library";

  m.def("getTensorInfo", &getTensorInfo);

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

  py::enum_<AnchorReturnTypeId>(m, "AnchorReturnTypeId")
      .value("FINAL", AnchorReturnTypeId::FINAL)
      .value("EVERYN", AnchorReturnTypeId::EVERYN)
      .value("ALL", AnchorReturnTypeId::ALL);

  py::class_<PyStepIO>(m, "PyStepIO", stepio)
      .def(py::init<std::map<TensorId, py::array>,
                    std::map<TensorId, py::array>>(),
           py::arg("inputs"),
           py::arg("outputs"));

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

  py::class_<TensorInfo>(m, "TensorInfo")
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

  py::class_<NllLoss>(m, "NllLoss", loss)
      .def(py::init<TensorId, TensorId, TensorId>(),
           py::arg("probabilities"),
           py::arg("labels"),
           py::arg("output"))
      .def("probsTensorId", &NllLoss::probsTensorId)
      .def("labelTensorId", &NllLoss::labelTensorId)
      .def("setVirtualGraph", &NllLoss::setVirtualGraphId);

  py::class_<L1Loss>(m, "L1Loss", loss)
      .def(py::init<TensorId, TensorId, float>(),
           py::arg("input"),
           py::arg("output"),
           py::arg("lambda"))
      .def("getInputId", &L1Loss::getInputId)
      .def("getLambda", &L1Loss::getLambda)
      .def("setVirtualGraph", &L1Loss::setVirtualGraphId);

  py::class_<Optimizer> optimizer(m, "Optimizer");

  py::class_<BaseSGD> basesgd(m, "BaseSGD", optimizer);
  basesgd.def("learnRate", &BaseSGD::learnRate);

  py::class_<SGD>(m, "SGD", basesgd)
      .def(py::init<float>(), py::arg("learning_rate"));

  py::class_<ConstSGD>(m, "ConstSGD", basesgd)
      .def(py::init<float>(), py::arg("learning_rate"));

  py::class_<SessionOptions>(m, "SessionOptionsCore")
      .def(py::init<>())
      .def_readwrite("logDir", &SessionOptions::logDir)
      .def_readwrite("exportDot", &SessionOptions::exportDot)
      .def_readwrite("ignoreData", &SessionOptions::ignoreData)
      .def_readwrite("enableConvolutionGraphCaching",
                     &SessionOptions::enableConvolutionGraphCaching)
      .def_readwrite("enableRecomputation",
                     &SessionOptions::enableRecomputation)
      .def_readwrite("enableVirtualGraphs",
                     &SessionOptions::enableVirtualGraphs)
      .def_readwrite("engineOptions", &SessionOptions::engineOptions)
      .def_readwrite("convolutionOptions", &SessionOptions::convolutionOptions)
      .def_readwrite("reportOptions", &SessionOptions::reportOptions)
      .def_readwrite("logging", &SessionOptions::loggingOptions);

  py::enum_<PatternsLevel>(m, "PatternsLevel")
      .value("ALL", PatternsLevel::ALL)
      .value("DEFAULT", PatternsLevel::DEFAULT)
      .value("NONE", PatternsLevel::NONE);

  py::enum_<PatternType>(m, "PatternType")
      .value("PREUNIREPL", PatternType::PREUNIREPL)
      .value("POSTNREPL", PatternType::POSTNREPL)
      .value("SOFTMAXGRADDIRECT", PatternType::SOFTMAXGRADDIRECT)
      .value("SPLITCONVBIAS", PatternType::SPLITCONVBIAS)
      .value("OPTOIDENTITY", PatternType::OPTOIDENTITY)
      .value("SUBTRACTARG1GRADOP", PatternType::SUBTRACTARG1GRADOP)
      .value("MULARGGRADOP", PatternType::MULARGGRADOP)
      .value("RECIPROCALGRADOP", PatternType::RECIPROCALGRADOP)
      .value("SINGRADOP", PatternType::SINGRADOP)
      .value("COSGRADOP", PatternType::COSGRADOP)
      .value("TANTOSINOVERCOS", PatternType::TANTOSINOVERCOS)
      .value("INPLACE0", PatternType::INPLACE0)
      .value("DIVARG0GRADOP", PatternType::DIVARG0GRADOP)
      .value("DIVARG1GRADOP", PatternType::DIVARG1GRADOP)
      .value("SQRTGRADOP", PatternType::SQRTGRADOP)
      .value("EXPGRADOP", PatternType::EXPGRADOP)
      .value("GEMMDECOMPOSITION", PatternType::GEMMDECOMPOSITION);

  py::class_<Patterns>(m, "Patterns")
      .def(py::init<>())
      .def(py::init<PatternsLevel>())
      .def(py::init<std::vector<PatternType>>())
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
          "InPlace0", &Patterns::isInPlace0Enabled, &Patterns::enableInPlace0)
      .def_property("InPlaceAll",
                    &Patterns::isInPlaceAllEnabled,
                    &Patterns::enableInPlaceAll)
      .def("__repr__", [](const Patterns &p) {
        std::stringstream ss;
        ss << p;
        return ss.str();
      });

  py::class_<Session>(m, "SessionCore")
      .def(py::init(&Session::createFromOnnxModel),
           py::arg("model"),
           py::arg("dataFlow").none(),
           py::arg("inputShapeInfo"),
           py::arg("losses"),
           py::arg("optimizer").none(),
           py::arg("userOptions"),
           py::arg("patterns"))
      .def("updateOptimizer", &Session::updateOptimizer)
      .def("setDevice", &Session::setDevice)
      .def("prepareDevice", &Session::prepareDevice)
      .def("weightsFromHost", &Session::weightsFromHost)
      .def("optimizerFromHost", &Session::optimizerFromHost)
      .def("train", &Session::train)
      .def("evaluate", &Session::evaluate)
      .def("infer", &Session::infer)
      .def("modelToHost", &Session::modelToHost)
      .def("getInfo", &Session::getInfo)
      .def("getSummaryReport", &Session::getSummaryReport)
      .def("getGraphReport", &Session::getGraphReport)
      .def("getExecutionReport", &Session::getExecutionReport)
      .def("getTensorTileMap", &Session::getTensorTileMap)
      .def("resetHostWeights", &Session::resetHostWeights);

  py::class_<Builder::BatchNormalizationTrainingOutputs>(
      m, "BatchNormalizationTrainingOutputs")
      .def_readonly("y", &Builder::BatchNormalizationTrainingOutputs::y)
      .def_readonly("mean", &Builder::BatchNormalizationTrainingOutputs::mean)
      .def_readonly("var", &Builder::BatchNormalizationTrainingOutputs::var)
      .def_readonly("savedMean",
                    &Builder::BatchNormalizationTrainingOutputs::savedMean)
      .def_readonly("savedvar",
                    &Builder::BatchNormalizationTrainingOutputs::savedVar);

  py::enum_<Builder::ExecutionMode>(m, "ExecutionMode")
      .value("INFERENCE", Builder::ExecutionMode::INFERENCE)
      .value("TRAINING", Builder::ExecutionMode::TRAINING);

  py::class_<GraphTransformer>(m, "GraphTransformer")
      .def(py::init<const py::bytes &>(), py::arg("modelProtoOrFilename"))
      .def("getModelProto",
           [](const GraphTransformer &graphtransformer) {
             return py::bytes(graphtransformer.getModelProto());
           })
      .def("convertFloatsToHalfs", &GraphTransformer::convertFloatsToHalfs)
      .def("convertInitializersToConstants",
           &GraphTransformer::convertInitializersToConstants,
           py::arg("ids"))
      .def("convertAllFixedPointInitializersToConstants",
           &GraphTransformer::convertAllFixedPointInitializersToConstants);

  py::class_<Builder>(m, "BuilderCore")
      .def(py::init(&Builder::create))
      .def(py::init(&Builder::createFromOnnxModel),
           py::arg("modelProtoOrFilename"))
      .def("addInputTensor", &Builder::addInputTensor, py::arg("tensorInfo"))
      .def("addInitializedInputTensor",
           [](Builder &builder, py::array array) {
             ConstVoidData initData;
             initData.data = array.request().ptr;
             initData.info = getTensorInfo(array);
             return builder.addInitializedInputTensor(initData);
           },
           py::arg("initVal"))
      .def("addOutputTensor", &Builder::addOutputTensor, py::arg("outputName"))
      .def("constant",
           [](Builder &builder, py::array array, const std::string &name) {
             ConstVoidData initData;
             initData.data = array.request().ptr;
             initData.info = getTensorInfo(array);
             return builder.constant(initData, name);
           },
           py::arg("initVal"),
           py::arg("debugPrefix") = std::string())
      .def("abs",
           &Builder::abs,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("acos",
           &Builder::acos,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("acosh",
           &Builder::acosh,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("add",
           &Builder::add,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("logical_and",
           &Builder::logical_and,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("asin",
           &Builder::asin,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("asinh",
           &Builder::asinh,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("atan",
           &Builder::atan,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("atanh",
           &Builder::atanh,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("ceil",
           &Builder::ceil,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("cos",
           &Builder::cos,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("cosh",
           &Builder::cosh,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("dropout",
           &Builder::dropout,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("div",
           &Builder::div,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("elu",
           &Builder::elu,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("equal",
           &Builder::equal,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("exp",
           &Builder::exp,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("floor",
           &Builder::floor,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("gather",
           &Builder::gather,
           py::arg("args"),
           py::arg("axis"),
           py::arg("debugPrefix") = std::string())
      .def("greater",
           &Builder::greater,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("identity",
           &Builder::identity,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("less",
           &Builder::less,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("log",
           &Builder::log,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("logsoftmax",
           &Builder::logsoftmax,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("max",
           &Builder::max,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("mean",
           &Builder::mean,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("min",
           &Builder::min,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("mul",
           &Builder::mul,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("neg",
           &Builder::neg,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("logical_not",
           &Builder::logical_not,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("logical_or",
           &Builder::logical_or,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("pow",
           &Builder::pow,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("reciprocal",
           &Builder::reciprocal,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("relu",
           &Builder::relu,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("scatter",
           &Builder::scatter,
           py::arg("args"),
           py::arg("axis"),
           py::arg("debugPrefix") = std::string())
      .def("sigmoid",
           &Builder::sigmoid,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("sin",
           &Builder::sin,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("sinh",
           &Builder::sinh,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("softmax",
           &Builder::softmax,
           py::arg("args"),
           py::arg("axis")        = 1,
           py::arg("debugPrefix") = std::string())
      .def("softsign",
           &Builder::softsign,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("sqrt",
           &Builder::sqrt,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("squeeze",
           &Builder::squeeze,
           py::arg("args"),
           py::arg("axes"),
           py::arg("debugPrefix") = std::string())
      .def("unsqueeze",
           &Builder::unsqueeze,
           py::arg("args"),
           py::arg("axes"),
           py::arg("debugPrefix") = std::string())
      .def("sub",
           &Builder::sub,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("sum",
           &Builder::sum,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("tan",
           &Builder::tan,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("tanh",
           &Builder::tanh,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("logical_xor",
           &Builder::logical_xor,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("convolution",
           &Builder::convolution,
           py::arg("args"),
           py::arg("strides"),
           py::arg("padding"),
           py::arg("dilation"),
           py::arg("groups")         = 1,
           py::arg("cacheOperation") = true,
           py::arg("debugPrefix")    = std::string())
      .def("averagepool",
           &Builder::averagepool,
           py::arg("args"),
           py::arg("kernel_shape"),
           py::arg("strides"),
           py::arg("padding"),
           py::arg("debugPrefix") = std::string())
      .def("maxpool",
           &Builder::maxpool,
           py::arg("args"),
           py::arg("kernel_shape"),
           py::arg("strides"),
           py::arg("padding"),
           py::arg("debugPrefix") = std::string())
      .def("lstm",
           &Builder::lstm,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("subsample",
           &Builder::subsample,
           py::arg("args"),
           py::arg("strides"),
           py::arg("debugPrefix") = std::string())
      .def("pad",
           &Builder::pad,
           py::arg("args"),
           py::arg("mode"),
           py::arg("pads"),
           py::arg("value"),
           py::arg("debugPrefix") = std::string())
      .def("onehot",
           &Builder::onehot,
           py::arg("args"),
           py::arg("axis"),
           py::arg("debugPrefix") = std::string())
      .def("gemm",
           &Builder::gemm,
           py::arg("args"),
           py::arg("alpha"),
           py::arg("beta"),
           py::arg("transA"),
           py::arg("transB"),
           py::arg("debugPrefix") = std::string())
      .def("shape",
           &Builder::shape,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("slice",
           &Builder::slice,
           py::arg("args"),
           py::arg("axes"),
           py::arg("starts"),
           py::arg("ends"),
           py::arg("debugPrefix") = std::string())
      .def("transpose",
           &Builder::transpose,
           py::arg("args"),
           py::arg("perm"),
           py::arg("debugPrefix") = std::string())
      .def("reshape",
           &Builder::reshape,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("reshape_const",
           &Builder::reshape_const,
           py::arg("args"),
           py::arg("shape"),
           py::arg("debugPrefix") = std::string())
      .def("matmul",
           &Builder::matmul,
           py::arg("args"),
           py::arg("debugPrefix") = std::string())
      .def("batchnormalizationTraining",
           &Builder::batchnormalizationTraining,
           py::arg("x"),
           py::arg("scale"),
           py::arg("b"),
           py::arg("mean"),
           py::arg("var"),
           py::arg("epsilon")     = 1e-5,
           py::arg("momentum")    = 0.9,
           py::arg("spatial")     = 1,
           py::arg("debugPrefix") = std::string())
      .def("batchnormalizationTesting",
           &Builder::batchnormalizationTesting,
           py::arg("x"),
           py::arg("scale"),
           py::arg("b"),
           py::arg("mean"),
           py::arg("var"),
           py::arg("epsilon")     = 1e-5,
           py::arg("momentum")    = 0.9,
           py::arg("spatial")     = 1,
           py::arg("debugPrefix") = std::string())
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
      .def("virtualGraph",
           [](Builder &self, int64_t index) -> AttributeContextManager {
             AttributeContextManager acm(self, sVirtualGraphAttribute, index);
             return acm;
           },
           py::arg("value"))

      .def("getVirtualGraph",
           static_cast<int64_t (Builder::*)(const TensorId &)>(
               &Builder::getVirtualGraph),
           py::arg("nodeOutputNames"))
      .def("recomputeOutputInBackwardPass",
           static_cast<void (Builder::*)(const TensorId &, bool value)>(
               &Builder::recomputeOutputInBackwardPass),
           py::arg("nodeOutputNames"),
           py::arg("value") = true)
      .def("getRecomputeOutputInBackwardPass",
           static_cast<bool (Builder::*)(const TensorId &)>(
               &Builder::getRecomputeOutputInBackwardPass),
           py::arg("nodeOutputNames"))
      .def("listConstExprNodes", &Builder::listConstExprNodes, py::arg("mode"))
      .def("listNonConstExprNodes",
           &Builder::listNonConstExprNodes,
           py::arg("mode"));

  py::class_<AttributeContextManager>(m, "AttributeContextManager")
      .def("__enter__", &AttributeContextManager::enter)
      .def("__exit__",
           [](AttributeContextManager &self, void *, void *, void *) {
             self.exit();
           });

  // PyBinding to a singleton
  py::class_<DeviceManager, std::unique_ptr<DeviceManager, py::nodelete>>(
      m, "DeviceManager")
      .def(py::init([]() {
        return std::unique_ptr<DeviceManager, py::nodelete>(
            &DeviceManager::createDeviceManager());
      }))
      .def("acquireAvailableDevice",
           static_cast<std::unique_ptr<DeviceInfo> (DeviceManager::*)()>(
               &DeviceManager::acquireAvailableDevice))
      .def(
          "acquireAvailableDevice",
          static_cast<std::unique_ptr<DeviceInfo> (DeviceManager::*)(int, int)>(
              &DeviceManager::acquireAvailableDevice),
          py::arg("numIpus"),
          py::arg("tilesPerIpu"))
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

  py::class_<DeviceInfo>(m, "DeviceInfo")
      .def("attach", &DeviceInfo::attach)
      .def("detach", &DeviceInfo::detach)
      .def_property_readonly("type", &DeviceInfo::getType)
      .def_property_readonly("version", &DeviceInfo::getVersion)
      .def_property_readonly("id", &DeviceInfo::getId)
      .def_property_readonly("numIpus", &DeviceInfo::getNumIpus)
      .def_property_readonly("tilesPerIpu", &DeviceInfo::getTilesPerIpu)
      .def_property_readonly("numWorkerContexts",
                             &DeviceInfo::getNumWorkerContexts)
      .def("__repr__", [](const DeviceInfo &di) {
        std::stringstream ss;
        ss << di;
        return ss.str();
      });

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
