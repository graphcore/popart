// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORDATA_HPP
#define GUARD_NEURALNET_TENSORDATA_HPP

#include <functional>
#include <numeric>
#include <ostream>
#include <popart/error.hpp>
#include <popart/iarray.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

// A class to hold data, used
// within the popart::Tensor class.
class TensorData {
public:
  // create by copying from src to data_,
  // the size of the copy determined by TensorInfo
  TensorData(const TensorInfo &, const void *src);

  // create by copying to data_ from ONNX_NAMESPACE::TensorProto
  TensorData(const ONNX_NAMESPACE::TensorProto &);

  void *data();
  const void *data() const;

  // reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  void resetData(const TensorInfo &, const void *src);

  // reset the data in the TensorData by copying from
  // ONNX_NAMESPACE::TensorProto. Input data must be the same size as the
  // existing data_
  void resetData(const ONNX_NAMESPACE::TensorProto &);

  // reset the data in the TensorData by copying from src.
  // Input data does not have to be the same size as the
  // existing data_
  void resetDataWithNonMatchingSize(const TensorInfo &info,
                                    const std::vector<char> from);

  template <typename RESULT_TYPE>
  std::vector<RESULT_TYPE> copyDataAs(int expectedResultSize) const {
    if (data_.size() != expectedResultSize * sizeof(RESULT_TYPE)) {
      throw error("Size of data does not match expected result size. Expected "
                  "data of {} bytes, but data is {} bytes in size.",
                  expectedResultSize * sizeof(RESULT_TYPE),
                  data_.size());
    }

    std::vector<RESULT_TYPE> result;
    const RESULT_TYPE *x = reinterpret_cast<const RESULT_TYPE *>(data());
    for (int i = 0; i < expectedResultSize; i++) {
      result.push_back(x[i]);
    }
    return result;
  }

private:
  std::vector<char> data_;
};

} // namespace popart

#endif
