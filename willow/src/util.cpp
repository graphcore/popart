// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <iostream>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/util.hpp>

#include <boost/lexical_cast.hpp>

namespace popart {

char *getPopartEnvVar(std::string env_var) {
  return std::getenv(logging::format("POPART_{}", env_var).c_str());
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

int roundToInt(float d) { return static_cast<int>(std::roundf(d)); }

unsigned roundToUnsigned(float d) {
  return static_cast<unsigned>(std::roundf(d));
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
    return convertIntTo<int>(roundToInt(data));
  }

  else if (dtype == DataType::UINT32) {
    return convertUnsignedIntTo<uint32_t>(roundToUnsigned(data));
  }

  else if (dtype == DataType::INT8) {
    return convertIntTo<int8_t>(roundToInt(data));
  }

  else if (dtype == DataType::UINT8) {
    return convertUnsignedIntTo<uint8_t>(roundToUnsigned(data));
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
  data_out.resize(sizeof(T));
  T converted_data{static_cast<T>(data)};
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// convert an unsigned int to type T
template <typename T> std::vector<char> convertUnsignedIntTo(uint32_t data) {
  std::vector<char> data_out;
  T converted_data{static_cast<T>(data)};
  data_out.resize(sizeof(T));
  *reinterpret_cast<T *>(data_out.data()) = converted_data;
  return data_out;
}

// map negative indices to positive indices, and cast to uint64_t.
std::vector<uint64_t> getAxes_u64(const std::vector<int64_t> &axes,
                                  uint64_t outRank) {

  std::vector<uint64_t> axes_u64;
  for (auto d : axes) {
    if (d < 0) {
      d += outRank;
    }
    if (d < 0) {
      std::ostringstream oss;
      oss << "Invalid axis in getAxes_u64(axes=";
      poprithms::util::append(oss, axes);
      oss << ", outRank=" << outRank << "). ";
      throw error(oss.str());
    }
    d = d % outRank;
    axes_u64.push_back(d);
  }
  return axes_u64;
}

int64_t getReduceAxis(int64_t axis_, int64_t inShapeSize) {
  // Onnx 11 supports negative axis indexing for reduce.
  if (axis_ >= int64_t(0)) {
    return axis_;
  } else {
    return inShapeSize + axis_;
  }
}

void normalizeReduceAxes(std::vector<int64_t> &axes, int64_t inShapeSize) {
  for (int64_t i = 0; i < axes.size(); i++) {
    axes[i] = getReduceAxis(axes[i], inShapeSize);
  }
}

void validateReduceAxis(int64_t axis_,
                        int64_t inShapeSize,
                        const std::string &message) {

  if (inShapeSize == 0) {
    throw error("Reduce input rank must be greater than 0, invalid "
                "Reduce {}.",
                message);
  }

  if (inShapeSize <= axis_) {
    throw error("Cannot compute Reduce on axis {} when input rank is {}, "
                "invalid Reduce {}.",
                axis_,
                inShapeSize,
                message);
  }

  // From the onnx spec:
  // Accepted range is [-r, r-1] where r = rank(data).
  if (axis_ > static_cast<int64_t>(inShapeSize) - 1 ||
      axis_ < -static_cast<int64_t>(inShapeSize)) {
    throw error("Axis {} is out of acceptable range [{}, {}]",
                axis_,
                -static_cast<int64_t>(inShapeSize),
                inShapeSize - 1);
  }
}

void validateReduceAxes(const std::vector<int64_t> &axes,
                        int64_t inShapeSize,
                        const std::string &message) {
  for (size_t i = 0; i < axes.size(); i++) {
    validateReduceAxis(axes[i], inShapeSize, message);
  }
}

namespace {
template <typename S, typename D>
void cast(const void *src, void *dst, size_t nelms) {
  for (size_t i = 0; i < nelms; ++i) {
    *(reinterpret_cast<D *>(dst) + i) =
        boost::numeric_cast<D, S>(*(reinterpret_cast<const S *>(src) + i));
  }
}
} // namespace

std::vector<char>
cast(DataType src, DataType dst, const void *data, size_t nbytes) {
  const DataTypeInfo *srcDataTypeInfo = &getDataTypeInfoMap().at(src);
  const DataTypeInfo *dstDataTypeInfo = &getDataTypeInfoMap().at(dst);

  size_t nelms     = nbytes / srcDataTypeInfo->nbytes();
  size_t dstnbytes = nelms * dstDataTypeInfo->nbytes();

  if (dstnbytes < nbytes) {
    logging::info("[cast] Narrowing cast from {} to {}", src, dst);
  }

  std::vector<char> outData(dstnbytes);

  const void *srcData = data;
  void *dstData       = static_cast<void *>(outData.data());

  auto err = [&src, &dst]() {
    throw error("[cast] Unsupported cast data types {} -> {}", src, dst);
  };

  try {
    switch (src) {
    case DataType::INT32:
      switch (dst) {
      case DataType::INT64:
        cast<int32_t, int64_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    case DataType::UINT32:
      switch (dst) {
      case DataType::UINT64:
        cast<uint32_t, uint64_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    case DataType::INT64:
      switch (dst) {
      case DataType::INT32:
        cast<int64_t, int32_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    case DataType::UINT64:
      switch (dst) {
      case DataType::UINT32:
        cast<uint64_t, uint32_t>(srcData, dstData, nelms);
        break;
      default:
        err();
      }
      break;
    default:
      err();
    }
  } catch (boost::bad_numeric_cast &e) {
    throw error("[cast] Cast {} -> {} failed: {}", src, dst, e.what());
  }

  return outData;
}

std::vector<char>
cast(DataType src, DataType dst, const std::vector<char> &data) {
  return cast(src, dst, static_cast<const void *>(data.data()), data.size());
}

TensorId getBaseTensorId(const TensorId &t) {
  int64_t i = t.size() - 1;
  if (!isdigit(t.at(i))) {
    return t;
  }
  while (i >= 0 && isdigit(t.at(i))) {
    i--;
  }
  if (i < 3 || t.at(i) != 't') {
    return t;
  }
  if (t.at(i - 1) != '_' || t.at(i - 2) != '_') {
    return t;
  }
  return t.substr(0, i - 2);
}

} // namespace popart
