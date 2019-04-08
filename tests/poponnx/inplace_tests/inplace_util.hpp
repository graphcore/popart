#ifndef GUARD_INPLACE_UTIL_HPP
#define GUARD_INPLACE_UTIL_HPP

#include <memory>
#include <vector>
#include <poponnx/builder.hpp>
#include <poponnx/dataflow.hpp>
#include <poponnx/devicemanager.hpp>
#include <poponnx/filereader.hpp>
#include <poponnx/names.hpp>
#include <poponnx/ndarraywrapper.hpp>

// Hack to see the internals of Session
#define private public
#define protected public
#include <poponnx/session.hpp>
#undef private
#undef protected

using namespace poponnx;

class TestTensor {
public:
  TestTensor()                   = delete;
  TestTensor(const TestTensor &) = delete;
  TestTensor(TestTensor &&)      = default;

  template <typename T>
  static TestTensor create(const TensorId &id,
                           const std::vector<T> &data,
                           const std::vector<int64_t> &shape);
  template <typename T>
  static TestTensor create(const TensorId &id,
                           const std::vector<int64_t> &shape);
  IArray &getIArray() { return *dataWrapper.get(); }
  template <typename T> void setData(const std::vector<T> &data_);
  template <typename T> const T *getDataPtr();
  template <typename T> std::vector<T> getDataCopy();

  const TensorId id;

private:
  TestTensor(const TensorId &id_) : id(id_) {}

  std::vector<char> data;
  std::unique_ptr<IArray> dataWrapper;
};

class TestRunner {
public:
  template <typename ModelBuilder> void buildModel(ModelBuilder &modelBuilder) {
    // Build an onnx model
    auto builder = Builder::create();
    outId        = modelBuilder(*builder);
    proto        = builder->getModelProto();
  }

  template <typename IrChecker> void checkIr(IrChecker &irChecker) {
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow  = DataFlow(1, {{outId, AnchorReturnType("ALL")}});
    auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

    session = InferenceSession::createFromOnnxModel(
        proto,
        dataFlow,
        cpuDevice,
        {},
        {},
        {},
        Patterns(PatternsLevel::NONE).enableInPlace(enableInPlace));

    irChecker(session->ir);
  }

  template <typename ResultsChecker>
  void checkResult(ResultsChecker &resultsChecker,
                   std::vector<TestTensor> &inputs,
                   std::vector<TestTensor> &outputs) {
    std::map<TensorId, IArray &> output_map;
    for (auto &output : outputs) {
      output_map.insert({output.id, output.getIArray()});
    }

    std::map<TensorId, IArray &> input_map;
    for (auto &input : inputs) {
      input_map.insert({input.id, input.getIArray()});
    }

    session->prepareDevice();
    StepIO stepio(input_map, output_map);
    session->run(stepio);

    resultsChecker(outputs.front());
  }

  bool enableInPlace = true;

private:
  std::string proto;
  TensorId outId;
  std::unique_ptr<InferenceSession> session;
};

template <typename T>
TensorInfo createTensorInfo(const std::vector<T> &data,
                            const std::vector<int64_t> &shape);

template <>
TensorInfo createTensorInfo(const std::vector<float> &data,
                            const std::vector<int64_t> &shape) {
  return TensorInfo{DataType::FLOAT, shape};
}

template <typename T>
TestTensor TestTensor::create(const TensorId &id,
                              const std::vector<T> &data,
                              const std::vector<int64_t> &shape) {
  TestTensor input(id);

  input.data.resize(data.size() * sizeof(T));
  input.setData(data);

  auto tinfo    = createTensorInfo(data, shape);
  auto raw_data = reinterpret_cast<T *>(input.data.data());
  input.dataWrapper.reset(new NDArrayWrapper<T>(raw_data, tinfo));

  return input;
}

template <typename T>
TestTensor TestTensor::create(const TensorId &id,
                              const std::vector<int64_t> &shape) {
  TestTensor input(id);

  auto tinfo = createTensorInfo(std::vector<T>{}, shape);
  input.data.resize(tinfo.nbytes());

  auto raw_data = reinterpret_cast<T *>(input.data.data());
  input.dataWrapper.reset(new NDArrayWrapper<T>(raw_data, tinfo));

  return input;
}

template <typename T> void TestTensor::setData(const std::vector<T> &data_) {
  assert(data.size() == data_.size() * sizeof(T));

  const char *char_data = reinterpret_cast<const char *>(data_.data());
  for (int i = 0; i < data_.size() * sizeof(T); i++) {
    data[i] = char_data[i];
  }
}

template <typename T> const T *TestTensor::getDataPtr() {
  return reinterpret_cast<const T *>(data.data());
}

template <typename T> std::vector<T> TestTensor::getDataCopy() {
  std::vector<T> outData;
  auto rawData = getDataPtr<T>();
  for (int i = 0; i < dataWrapper->nelms(); i++) {
    outData.push_back(rawData[i]);
  }
  return outData;
}

#endif
