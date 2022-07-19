// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_STEPIO_GENERIC_HPP_
#define POPART_WILLOW_INCLUDE_POPART_STEPIO_GENERIC_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <popart/error.hpp>
#include <popart/istepio.hpp>
#include <popart/stepio_size_assertion.hpp>

#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/voiddata.hpp"

namespace popart {
namespace popx {
class Executablex;
}

template <typename ARRAY_TYPE, typename ACCESSOR_TYPE, typename ArrayInfoT>
class StepIOGeneric : public IStepIO {

  struct ArrayInfo {
    ArrayInfoT array;
    int64_t offset;
  };

public:
  void assertNumElements(const popx::Executablex &exe) const final {
    auto g = [](const ArrayInfo &info) {
      return ACCESSOR_TYPE::getArraySize(info.array);
    };
    iosizecheck::assertInCorrect(exe, inputsInfo, g);
    iosizecheck::assertOutCorrect(exe, outputsInfo, g);
  }

  TensorInfo getTensorInfo(ARRAY_TYPE &array) const {
    auto dtype = ACCESSOR_TYPE::getArrayDataType(array);
    auto tRank = ACCESSOR_TYPE::getArrayRank(array);
    std::vector<int64_t> shape;
    for (size_t i = 0; i < tRank; ++i) {
      shape.push_back(ACCESSOR_TYPE::getArrayDim(array, i));
    }
    return TensorInfo(dtype, shape);
  }
  template <typename T>
  T get(TensorId id,
        std::map<TensorId, ArrayInfo> &M,
        int64_t numElements,
        bool advance_,
        std::string mapName) {

    auto found = M.find(id);
    if (found == M.end()) {
      throw runtime_error(
          "No tensor {} provided in PyStepIO's {}", id, mapName);
    }

    ArrayInfo &arrayInfo = found->second;
    int64_t offset       = arrayInfo.offset;

    T stepData;
    stepData.info = getTensorInfo(arrayInfo.array);

    int64_t arraySize = stepData.info.nbytes();

    // Set the data using the offset
    stepData.data =
        static_cast<uint8_t *>(ACCESSOR_TYPE::getDataPointer(arrayInfo.array)) +
        offset;

    if (advance_) {

      int64_t numBytes =
          static_cast<int64_t>(stepData.info.getDataTypeInfo()->nbytes()) *
          numElements;

      // Wrap around if we read all the data
      if (offset + numBytes == arraySize) {
        arrayInfo.offset = 0;
      } else {
        arrayInfo.offset = offset + numBytes;
      }
    }

    return stepData;
  }

  template <typename T>
  void advance(TensorId id,
               std::map<TensorId, ArrayInfo> &M,
               int64_t numElements,
               std::string mapName) {

    auto found = M.find(id);
    if (found == M.end()) {
      throw runtime_error(
          "No tensor {} provided in PyStepIO's {}", id, mapName);
    }

    ArrayInfo &arrayInfo = found->second;
    int64_t offset       = arrayInfo.offset;

    T stepData;
    stepData.info = getTensorInfo(arrayInfo.array);

    int64_t arraySize = stepData.info.nbytes();

    // Set the data using the offset
    int64_t numBytes =
        static_cast<int64_t>(stepData.info.getDataTypeInfo()->nbytes()) *
        numElements;

    // Wrap around if we read all the data
    if (offset + numBytes == arraySize) {
      arrayInfo.offset = 0;
    } else {
      arrayInfo.offset = offset + numBytes;
    }
  }

  ConstVoidData in(TensorId id, int64_t numElements, bool)final {
    return get<ConstVoidData>(id, inputsInfo, numElements, true, "inputs");
  }

  void inComplete(TensorId id, int64_t numElements) final { return; }

  MutableVoidData out(TensorId id, int64_t numElements) final {
    return get<MutableVoidData>(id, outputsInfo, numElements, true, "outputs");
  }

protected:
  StepIOGeneric() {}
  std::map<TensorId, ArrayInfo> outputsInfo;
  std::map<TensorId, ArrayInfo> inputsInfo;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_STEPIO_GENERIC_HPP_
