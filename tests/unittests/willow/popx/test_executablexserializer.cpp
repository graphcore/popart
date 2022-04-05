// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExecutableXSerialisationUnittest

#include <sstream>

#include <boost/test/unit_test.hpp>

#include <capnp/message.h>
#include <capnp/serialize.h>

#include <kj/std/iostream.h>

#include <popx/executablexserializer.hpp>
#include <popart/ir.hpp>
#include <popart/tensor.hpp>

using namespace popart;
using namespace popart::popx::serialization;

BOOST_AUTO_TEST_CASE(serialiseAndDeserialiseTensorType) {

  // Check tensor type is unchanged following serialisation/deserialisation.

  auto test = [](TensorType orgTensorType) {
    auto capnpTensorType = toCapnpTensorType(orgTensorType);
    auto resTensorType   = toPopartTensorType(capnpTensorType);
    BOOST_REQUIRE_EQUAL(resTensorType, orgTensorType);
  };

  test(TensorType::ActGrad);
  test(TensorType::Const);
  test(TensorType::Stream);
  test(TensorType::Unknown);
  test(TensorType::Variable);
}

BOOST_AUTO_TEST_CASE(serialiseAndDeserialiseDataType) {

  // Check data type is unchanged following serialisation/deserialisation.

  auto test = [](DataType orgDataType) {
    auto capnpDataType = toCapnpDataType(orgDataType);
    auto resDataType   = toPopartDataType(capnpDataType);
    BOOST_REQUIRE_EQUAL(resDataType, orgDataType);
  };

  test(DataType::UINT8);
  test(DataType::INT8);
  test(DataType::UINT16);
  test(DataType::INT16);
  test(DataType::INT32);
  test(DataType::INT64);
  test(DataType::UINT32);
  test(DataType::UINT64);
  test(DataType::BOOL);
  test(DataType::FLOAT);
  test(DataType::FLOAT16);
  test(DataType::BFLOAT16);
  test(DataType::DOUBLE);
  test(DataType::COMPLEX64);
  test(DataType::COMPLEX128);
  test(DataType::STRING);
  test(DataType::UNDEFINED);
}

BOOST_AUTO_TEST_CASE(serialiseAndDeserialiseCommGroupType) {

  // Check CommGroupType is unchanged following serialisation/deserialisation.

  auto test = [](CommGroupType orgCommGroupType) {
    auto capnpCommGroupType = toCapnpCommGroupType(orgCommGroupType);
    auto resDataType        = toPopartCommGroupType(capnpCommGroupType);
    BOOST_REQUIRE_EQUAL(resDataType, orgCommGroupType);
  };

  test(CommGroupType::All);
  test(CommGroupType::Orthogonal);
  test(CommGroupType::Consecutive);
  test(CommGroupType::None);
}

BOOST_AUTO_TEST_CASE(serialiseAndDeserialiseVariableRetrievalMode) {

  // Check VariableRetrievalMode is unchanged following
  // serialisation/deserialisation.

  auto test = [](VariableRetrievalMode orgVariableRetrievalMode) {
    auto capnpVariableRetrievalMode =
        toCapnpVariableRetrievalMode(orgVariableRetrievalMode);
    auto resVariableRetrievalMode =
        toPopartVariableRetrievalMode(capnpVariableRetrievalMode);
    BOOST_REQUIRE_EQUAL(resVariableRetrievalMode, orgVariableRetrievalMode);
  };

  test(VariableRetrievalMode::OnePerGroup);
  test(VariableRetrievalMode::AllReduceReplicas);
  test(VariableRetrievalMode::AllReplicas);
}

BOOST_AUTO_TEST_CASE(serialiseAndDeserialiseTensor) {

  // Check Tensor is unchanged following serialisation/deserialisation.

  auto test = [](TensorId id,
                 TensorType type,
                 TensorInfo info,
                 VariableSettings varSet,
                 nonstd::optional<std::vector<char>> data) {
    Ir ir{};
    Graph &graph = ir.getMainGraph();

    // Create a tensor to test with.
    Tensor orgTensor(id, varSet, graph);
    orgTensor.info = info;

    if (data) {
      // Set data if we have any.
      orgTensor.setTensorData(reinterpret_cast<void *>(data->data()),
                              data->size());
    }

    // Serialise via capnp to stringstream.
    std::stringstream ss;
    ::capnp::MallocMessageBuilder msgBuilder;
    auto tensorBuilder = msgBuilder.initRoot<cap::Tensor>();
    serializeTensor(&orgTensor, tensorBuilder, orgTensor.hasTensorData());
    kj::std::StdOutputStream sos(ss);
    capnp::writeMessage(sos, msgBuilder);

    // Deserialise from stringstream.
    kj::std::StdInputStream sis(ss);
    capnp::ReaderOptions opts;
    capnp::InputStreamMessageReader msg(sis, opts);
    auto tensorReader = msg.getRoot<cap::Tensor>();
    auto resTensor =
        deserializeTensor(ir, tensorReader, orgTensor.hasTensorData());

    // Check TensorId.
    BOOST_REQUIRE_EQUAL(orgTensor.id, resTensor->id);

    // Check TensorType.
    BOOST_REQUIRE_EQUAL(orgTensor.tensorType(), resTensor->tensorType());

    // Check VariableSettings.
    BOOST_REQUIRE_EQUAL(orgTensor.getVariableSettings().getRetrievalMode(),
                        resTensor->getVariableSettings().getRetrievalMode());
    BOOST_REQUIRE_EQUAL(
        orgTensor.getVariableSettings().getSharedVariableDomain().type,
        resTensor->getVariableSettings().getSharedVariableDomain().type);
    BOOST_REQUIRE_EQUAL(orgTensor.getVariableSettings()
                            .getSharedVariableDomain()
                            .replicaGroupSize,
                        resTensor->getVariableSettings()
                            .getSharedVariableDomain()
                            .replicaGroupSize);

    // Check TensorData
    BOOST_REQUIRE_EQUAL(orgTensor.hasTensorData(), resTensor->hasTensorData());
    if (orgTensor.hasTensorData()) {
      BOOST_REQUIRE_EQUAL(orgTensor.tensorData()->size(),
                          resTensor->tensorData()->size());
      for (size_t i = 0; i < orgTensor.tensorData()->size(); ++i) {
        BOOST_REQUIRE_EQUAL(
            reinterpret_cast<char *>(orgTensor.tensorData()->data())[i],
            reinterpret_cast<char *>(resTensor->tensorData()->data())[i]);
      }
    }
  };

  test(TensorId("tensor-A"),
       TensorType::Variable,
       TensorInfo(DataType::INT8, {4}),
       VariableSettings(CommGroup(CommGroupType::Consecutive, 2),
                        VariableRetrievalMode::OnePerGroup),
       std::vector<char>({'a', 'b', 'c', 'd'}));

  test(TensorId("tensor-B"),
       TensorType::Const,
       TensorInfo(DataType::INT32, {166}),
       VariableSettings(CommGroup(CommGroupType::Orthogonal, 5),
                        VariableRetrievalMode::AllReduceReplicas),
       nonstd::nullopt);
}
