#include <iostream>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

namespace popart {

char *getPopartEnvVar(std::string env_var) {
  auto result = std::getenv(logging::format("POPART_{}", env_var).c_str());
  if (result) {
    return result;
  }

  result = std::getenv(logging::format("POPONNX_{}", env_var).c_str());
  if (result != nullptr) {
    std::cerr << logging::format(
        "You are using a deprecated environment variable "
        "'POPONNX_{}', please change it to 'POPART_{}'\n",
        env_var,
        env_var);
  }

  return result;
}

std::ostream &operator<<(std::ostream &ss, const std::vector<std::size_t> &v) {
  appendSequence(ss, v);
  return ss;
}

void OpSearchHelper::pushConsumers(Tensor *t) {
  for (auto consumer : t->consumers.getOps()) {
    push(consumer);
  }
}

void OpSearchHelper::pushOutputConsumers(Op *op) {
  for (auto output : op->output->tensors()) {
    pushConsumers(output);
  }
}

// convert a float to the DataType `dtype`
std::vector<char> convertFloatToDataType(DataType dtype, float data) {
  if (dtype == DataType::FLOAT) {
    return convertFloatTo<float>(data);
  }

  else if (dtype == DataType::FLOAT16) {
    return convertFloatTo<Half>(data);
  }

  else if (dtype == DataType::INT32) {
    return convertIntTo<int>(static_cast<int>(data));
  }

  throw error("Can't convert float to DataType {}",
              getDataTypeInfoMap().at(dtype).name());
}

// convert a float to type T
template <typename T> std::vector<char> convertFloatTo(float data) {
  std::vector<char> data_out;
  T converted_data{data};
  data_out.resize(sizeof(T));
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// convert an int to type T
template <typename T> std::vector<char> convertIntTo(int data) {
  std::vector<char> data_out;
  T converted_data{data};
  data_out.resize(sizeof(T));
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

} // namespace popart
