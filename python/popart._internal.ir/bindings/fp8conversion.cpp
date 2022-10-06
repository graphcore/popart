// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#include <../popart/shared_cpp/np_utils.hpp>
#include <bindings/fp8conversion.hpp>
#include <initializer_list>
#include <pybind11/cast.h>
#include <pybind11/functional.h> // IWYU pragma: keep
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // IWYU pragma: keep
#include <poplar/Quarter.hpp>
#include <poplar/Type.hpp>
#include <poplar/TypeConversion.hpp>
#include <popart/docs/pydocs_popart_core.hpp>
#include <popart/popx/irlowering.hpp>

namespace py = pybind11;

namespace popart {
namespace _internal {
namespace ir {

namespace {

/** Convert from host buffer of type float to device data of type
 *  uint8 (a proxy for float8 , since float8 is not available in numpy). Choose
 * to saturate or produce NaNs where values exceed the numeric range of the
 * destination type selected. PopART equivalent of the same poplar function.
 *
 *  \param type        Data type on device which must be DataType::FLOAT8_143 or
 *                     DataType::FLOAT8_152.
 *  \param src         Host data buffer to read.
 *  \param notANumberOnOverflow If True produce NaN when the input values
 *                     exceed the numeric range of the destination type
 *                     selected.
 *                     If False saturate the results.
 * \returns            The converted numpy array in uint8 format.
 */

/** Convert from host buffer of type float to device data of type
 *  uint8 (a proxy for float8 , since float8 is not available in numpy). Choose
 * to saturate or produce NaNs where values exceed the numeric range of the
 * destination type selected. PopART equivalent of the same poplar function.
 *
 * @tparam T Type to convert from float or double.
 * \param type Data type on device which must be DataType::FLOAT8_143 or
 * DataType::FLOAT8_152.
 * \param src Host data buffer to read.
 * \param scaleBias number to be raised to the power of 2 to scale the input.
 * \param notANumberOnOverflow If True produce NaN when the input values exceed
 * the numeric range of the destination type selected. If False saturate the
 * results.
 * \returns py::array_t<uint8_t> The converted numpy array in uint8
 * format.
 */
template <typename T>
py::array_t<uint8_t> convertToFloat8AsUInt8(const DataType destType,
                                            py::array_t<T> &src,
                                            int8_t scaleBias          = 0,
                                            bool notANumberOnOverflow = true) {

  poplar::QuarterMetadata metadata;

  if (destType == DataType::FLOAT8_143) {
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F143,
                                       scaleBias);
  } else if (destType == DataType::FLOAT8_152) {
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F152,
                                       scaleBias);
  } else {
    throw error("Unsupported data type {} for conversion to FP8", destType);
  }

  src               = makeContiguous(src);
  auto arr_obj_prop = src.request();
  auto vals         = static_cast<T *>(arr_obj_prop.ptr);

  std::vector<T> vect_arr;
  vect_arr.reserve(src.size());

  for (unsigned int i = 0; i < src.size(); i++) {
    vect_arr.push_back(vals[i]);
  }
  gccs::ArrayRef<T> ins{vect_arr};

  std::vector<uint8_t> out_vec(src.size(), 0);
  gccs::ArrayRef<uint8_t> dest{out_vec};

  poplar::convertToDeviceType(
      poplar::QUARTER, metadata, ins, dest.data(), notANumberOnOverflow);

  py::array outs = py::cast(out_vec);

  return outs;
}

/** Convert from uint8 (a proxy for float8 , since float8 is not available in
 * numpy) to host type T (float or double). Choose to saturate or produce NaNs
 * where values exceed the numeric range of the destination type selected.
 * PopART equivalent of the same poplar function.
 *
 * @tparam T Type to convert from float or double.
 * \param type Data type on device which must be DataType::FLOAT8_143 or
 * DataType::FLOAT8_152.
 * \param src Host data buffer to read.
 * \param scaleBias number to be raised to the power of 2 to scale the input.
 * \returns py::array_t<T> The converted numpy array in type T format.
 */
template <typename T>
py::array_t<T> convertFromFloat8(const DataType srcType,
                                 py::array_t<uint8_t> &src,
                                 int8_t scaleBias = 0) {

  poplar::QuarterMetadata metadata;

  if (srcType == DataType::FLOAT8_143) {
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F143,
                                       scaleBias);
  } else if (srcType == DataType::FLOAT8_152) {
    metadata = poplar::QuarterMetadata(poplar::QuarterMetadata::Format::F152,
                                       scaleBias);
  } else {
    throw error("Unsupported data type {} for conversion from FP8", srcType);
  }

  src                 = makeContiguous(src);
  auto arr_obj_prop   = src.request();
  const uint8_t *vals = static_cast<const uint8_t *>(arr_obj_prop.ptr);

  std::vector<uint8_t> vect_arr;
  vect_arr.reserve(src.size());

  for (unsigned int i = 0; i < src.size(); i++) {
    vect_arr.push_back(static_cast<uint8_t>(vals[i]));
  }
  gccs::ArrayRef<uint8_t> ins{static_cast<unsigned char *>(src.request().ptr),
                              static_cast<size_t>(src.size())};

  std::vector<T> out_vec(src.size(), 0);
  gccs::ArrayRef<T> dest{out_vec};

  poplar::convertFromDeviceType(poplar::QUARTER, metadata, ins.data(), dest);

  py::array outs = py::cast(out_vec);

  return outs;
}

} // namespace

void bindFp8conversion(py::module &m) {
  {
    m.def("convertToFloat8AsUInt8", &convertToFloat8AsUInt8<float>);
    m.def("convertToFloat8AsUInt8", &convertToFloat8AsUInt8<double>);
    m.def("convertFromFloat8AsUInt8ToFloat32", &convertFromFloat8<float>);
    m.def("convertFromFloat8AsUInt8ToFloat64", &convertFromFloat8<double>);
  }
}

} // namespace ir
} // namespace _internal
} // namespace popart
