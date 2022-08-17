// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ExecutableXSerialisationUnittest

#include <boost/test/unit_test.hpp>
#include <capnp/message.h>
#include <capnp/serialize.h>
#include <cstddef>
#include <kj/std/iostream.h>
#include <memory>
#include <sstream>
#include <vector>

#include <popef/Reader.hpp>
#include <popef/Writer.hpp>

#include <popx/executablexserializer.hpp>
#include <popart/ir.hpp>
#include <popart/popx/popefserializer.hpp>
#include <popart/tensor.hpp>

#include "popart/capnp/Ir.capnp.h"
#include "popart/commgroup.hpp"
#include "popart/datatype.hpp"
#include "popart/tensordata.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/variablesettings.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class Graph;
} // namespace popart

using namespace popart;
using namespace popart::popx::serialization;

void writeDataToStream(std::shared_ptr<std::stringstream> ss_ptr,
                       popart::Tensor &tensor) {
  popef::Writer writer(*ss_ptr);
  std::shared_ptr<popef::BlobWriter> opaque =
      writer.createOpaqueBlob("tensor", "null");
  ::capnp::MallocMessageBuilder msgBuilder;

  // Popart metadata for tensor is written to opaque blob.
  auto tensorBuilder = msgBuilder.initRoot<cap::Tensor>();
  serializeTensor(&tensor, tensorBuilder);
  kj::std::StdOutputStream sos(opaque->stream);
  capnp::writeMessage(sos, msgBuilder);
  opaque->close();

  // Binary tensor data is written to tensor data blob.
  if (tensor.hasTensorData()) {
    popef::TensorInfo ti = createTensorInfo(tensor.info);
    serializePopefTensor(tensor, ti, writer);
  }

  writer.close();
}

std::unique_ptr<popart::Tensor>
readDataFromStream(std::shared_ptr<std::stringstream> ss_ptr, popart::Ir &ir) {
  popef::Reader reader;
  reader.parseStream(ss_ptr);
  // Check if opaque is present in the popef stream.
  BOOST_REQUIRE_EQUAL(reader.opaqueBlobs().size(), 1);
  popef::OpaqueReader opaque = reader.opaqueBlobs()[0];
  kj::std::StdInputStream sis(opaque.data);
  capnp::ReaderOptions opts;
  capnp::InputStreamMessageReader msg(sis, opts);
  auto tensorReader = msg.getRoot<cap::Tensor>();

  const popef::TensorReader *tensorData = nullptr;
  if (reader.tensors().size() == 1) {
    // Get tensor data blob reader if it is present.
    tensorData = &reader.tensors()[0];
  }

  return deserializeTensor(ir, tensorReader, tensorData);
}

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

    if (data.has_value()) {
      // Set data if we have any.
      orgTensor.setTensorData(reinterpret_cast<void *>(data->data()),
                              data->size());
    }

    auto ss_ptr = std::make_shared<std::stringstream>();
    writeDataToStream(ss_ptr, orgTensor);
    auto resTensor = readDataFromStream(ss_ptr, ir);

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
