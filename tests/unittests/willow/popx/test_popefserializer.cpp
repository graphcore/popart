// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE PopEFSerialisationUnittest

#include <sstream>
#include <utility>
#include <vector>

#include <boost/test/data/test_case.hpp>
#include <boost/test/unit_test.hpp>

#include <popef/Reader.hpp>
#include <popef/Types.hpp>
#include <popef/Writer.hpp>

#include "popart/tensordata.hpp"
#include <popart/datatype.hpp>
#include <popart/ir.hpp>
#include <popart/popx/popefserializer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

using namespace popart::popx::serialization;

void comparePopartAndPopefTensorInfo(const popart::TensorInfo &popartTI,
                                     const popef::TensorInfo &popefTI) {
  BOOST_CHECK(popefTI.dataType() == toPopefDataType(popartTI.dataType()));
  BOOST_REQUIRE_EQUAL_COLLECTIONS(popefTI.shape().begin(),
                                  popefTI.shape().end(),
                                  popartTI.shape().begin(),
                                  popartTI.shape().end());
  BOOST_REQUIRE_EQUAL(popefTI.sizeInBytes(), popartTI.nbytes());
}

BOOST_AUTO_TEST_CASE(castingPopartDataTypeToPopefDataType) {
  const std::vector<std::pair<popart::DataType, popef::DataType>> dts{
      {popart::DataType::BOOL, popef::DataType::BOOL},
      {popart::DataType::UINT8, popef::DataType::U8},
      {popart::DataType::INT8, popef::DataType::S8},
      {popart::DataType::UINT16, popef::DataType::U16},
      {popart::DataType::INT16, popef::DataType::S16},
      {popart::DataType::INT32, popef::DataType::S32},
      {popart::DataType::UINT32, popef::DataType::U32},
      {popart::DataType::INT64, popef::DataType::S64},
      {popart::DataType::UINT64, popef::DataType::U64},
      {popart::DataType::FLOAT16, popef::DataType::F16},
      {popart::DataType::FLOAT, popef::DataType::F32},
      {popart::DataType::DOUBLE, popef::DataType::F64}};

  for (const auto &dt : dts) {
    BOOST_CHECK(toPopefDataType(dt.first) == dt.second);
  }
}

BOOST_AUTO_TEST_CASE(castingPopartTensorInfoToPopefTensorInfo) {
  auto test = [](const popart::TensorInfo &popartTI) {
    const popef::TensorInfo popefTI = createTensorInfo(popartTI);
    comparePopartAndPopefTensorInfo(popartTI, popefTI);
  };

  test(popart::TensorInfo("BOOL", "()"));
  test(popart::TensorInfo("UINT64", popart::Shape{64, 8, 8}));
  test(popart::TensorInfo(popart::DataType::DOUBLE, {16, 4, 256, 256}));
}

BOOST_AUTO_TEST_CASE(creatingPopefTensorInfoFromDataTypeAndShape) {
  auto test = [](const popart::DataType dt, const popart::Shape &shape) {
    const popef::DataType popefDT   = toPopefDataType(dt);
    const popef::TensorInfo popefTI = createTensorInfo(popefDT, shape);
    const auto &dataTypeInfo        = popart::getDataTypeInfoMap().at(dt);
    const int64_t nbytes =
        std::accumulate(shape.begin(),
                        shape.end(),
                        static_cast<int64_t>(dataTypeInfo.nbytes()),
                        std::multiplies<int64_t>());

    BOOST_CHECK(popefTI.dataType() == popefDT);
    BOOST_REQUIRE_EQUAL_COLLECTIONS(popefTI.shape().begin(),
                                    popefTI.shape().end(),
                                    shape.begin(),
                                    shape.end());
    BOOST_REQUIRE_EQUAL(popefTI.sizeInBytes(), nbytes);
  };

  test(popart::DataType::FLOAT16, {16384});
  test(popart::DataType::DOUBLE, {16, 4});
}

BOOST_AUTO_TEST_CASE(serializeAndDeserializePopefTensor) {
  auto test = [](const popart::Tensor &tensor, const popef::TensorInfo &ti) {
    auto ss_ptr = std::make_shared<std::stringstream>();

    popef::Writer writer(*ss_ptr);
    serializePopefTensor(tensor, ti, writer);
    writer.close();

    popef::Reader reader;
    reader.parseStream(ss_ptr);
    BOOST_CHECK_EQUAL(reader.tensors().size(), 1);
    popef::TensorReader tensorReader = reader.tensors()[0];

    const size_t bufferSize = tensorReader.info.tensorInfo().sizeInBytes();
    BOOST_CHECK_EQUAL(bufferSize, tensor.tensorData()->size());
    std::vector<char> tensorBuffer(bufferSize);
    tensorReader.data.read(tensorBuffer.data(), bufferSize);

    const char *dataPtr =
        reinterpret_cast<const char *>(tensor.tensorData()->data());

    comparePopartAndPopefTensorInfo(tensor.info,
                                    tensorReader.info.tensorInfo());
    BOOST_REQUIRE_EQUAL(tensor.id, tensorReader.info.name());
    BOOST_CHECK(std::memcmp(tensorBuffer.data(), dataPtr, bufferSize) == 0);
  };

  popart::Ir ir{};
  popart::Graph &graph = ir.getMainGraph();
  const popart::VariableSettings vs;

  popart::Tensor tensor("tensor", vs, graph);
  tensor.info = popart::TensorInfo(popart::DataType::UINT16, {4, 128, 16, 16});
  const popef::TensorInfo ti = createTensorInfo(tensor.info);
  BOOST_REQUIRE_THROW(test(tensor, ti), std::exception);

  std::vector<uint16_t> data(tensor.info.nelms());
  std::iota(data.begin(), data.end(), 0);
  tensor.setTensorData(data.data(), tensor.info.nbytes());

  BOOST_REQUIRE_THROW(test(tensor, popef::TensorInfo()), std::exception);

  test(tensor, ti);
}
