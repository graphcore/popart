// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE GetTensorIdsTest

#include <fstream>

#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>

#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/session.hpp>
#include <popart/testdevice.hpp>
#include <popart/voiddata.hpp>

struct TmpDir {
  TmpDir() {
    auto uniqueDir =
        boost::filesystem::unique_path("popart-tests-%%%%_%%%%_%%%%");
    tmpDir = boost::filesystem::temp_directory_path() / uniqueDir;
    if (boost::filesystem::exists(tmpDir)) {
      BOOST_REQUIRE(boost::filesystem::remove_all(tmpDir));
    }
    BOOST_REQUIRE(boost::filesystem::create_directories(tmpDir));
  }

  ~TmpDir() { BOOST_REQUIRE(boost::filesystem::remove_all(tmpDir)); }

  std::string path(const std::string &file) {
    return (tmpDir / file.c_str()).string();
  }

  boost::filesystem::path tmpDir;
};

bool isSubset(const std::vector<std::string> &vec,
              const std::set<std::string> &set) {
  for (const auto &str : vec) {
    if (std::find(set.begin(), set.end(), str) == set.end()) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<popart::Builder> getSimpleModel() {
  auto builder = popart::Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  int dataSize = 32;
  popart::TensorInfo info{popart::DataType::FLOAT,
                          std::vector<int64_t>{dataSize, dataSize}};

  const popart::TensorInfo exponent_tensor_info =
      popart::TensorInfo(popart::DataType::FLOAT, {});
  const int exponent_val = 2;
  const popart::ConstVoidData exponent_data(&exponent_val,
                                            exponent_tensor_info);
  auto exponent = aiOnnx.constant(exponent_data, "exponent");

  auto t1 = builder->addInputTensor(info, "matmul1_in1");
  auto t2 = builder->addInputTensor(info, "matmul1_in2");
  auto t3 = builder->addInputTensor(info, "matmul2_in2");

  auto m1 = aiOnnx.matmul({t1, t2}, "matmul1");
  auto m2 = aiOnnx.matmul({m1, t3}, "matmul2");

  auto pow = aiOnnx.pow({m1, exponent});

  auto add = aiOnnx.add({pow, m2}, "add_out");

  builder->addOutputTensor(add);

  return builder;
}

BOOST_AUTO_TEST_CASE(GetAllTensorIdsTest) {
  auto builder = getSimpleModel();

  auto inIds  = builder->getInputTensorIds();
  auto outIds = builder->getOutputTensorIds();
  auto valIds = builder->getValueTensorIds();

  BOOST_CHECK(builder->getTrainableTensorIds().empty());

  const popart::TensorId out = builder->getOutputTensorIds()[0];
  auto dataFlow = popart::DataFlow(1, {{out, popart::AnchorReturnType("All")}});
  auto device   = popart::createTestDevice(popart::TEST_TARGET, 1);

  auto session = popart::InferenceSession::createFromOnnxModel(
      builder->getModelProto(), dataFlow, device);
  session->prepareDevice();

  auto tensorIds = session->getAllTensorIds();

  BOOST_CHECK(isSubset(inIds, tensorIds));
  BOOST_CHECK(isSubset(outIds, tensorIds));
  BOOST_CHECK(isSubset(valIds, tensorIds));
}

BOOST_AUTO_TEST_CASE(GetAllTensorIdsFromDeserializedExecutableTest) {
  TmpDir testDir;

  auto builder = getSimpleModel();

  auto inIds  = builder->getInputTensorIds();
  auto outIds = builder->getOutputTensorIds();
  auto valIds = builder->getValueTensorIds();

  const popart::TensorId out = builder->getOutputTensorIds()[0];
  auto dataFlow = popart::DataFlow(1, {{out, popart::AnchorReturnType("All")}});
  auto device   = popart::createTestDevice(popart::TestDeviceType::OfflineIpu,
                                         1,
                                         0,
                                         popart::SyncPattern::Full,
                                         {{"ipuVersion", "ipu2"}});

  auto opts                = popart::SessionOptions();
  opts.enableEngineCaching = true;
  opts.cachePath           = testDir.tmpDir.string();

  {
    auto session =
        popart::InferenceSession::createFromOnnxModel(builder->getModelProto(),
                                                      dataFlow,
                                                      device,
                                                      popart::InputShapeInfo(),
                                                      opts);
    session->prepareDevice(false);

    auto tensorIds = session->getAllTensorIds();

    BOOST_CHECK(isSubset(inIds, tensorIds));
    BOOST_CHECK(isSubset(outIds, tensorIds));
    BOOST_CHECK(isSubset(valIds, tensorIds));
  }

  {
    auto session =
        popart::InferenceSession::createFromOnnxModel(builder->getModelProto(),
                                                      dataFlow,
                                                      device,
                                                      popart::InputShapeInfo(),
                                                      opts);
    session->prepareDevice(false);

    auto tensorIds = session->getAllTensorIds();

    BOOST_CHECK(isSubset(inIds, tensorIds));
    BOOST_CHECK(isSubset(outIds, tensorIds));
    BOOST_CHECK(isSubset(valIds, tensorIds) == false);
  }
}