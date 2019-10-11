#define BOOST_TEST_MODULE IrHashTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>

using namespace popart;

/*
 Things the Ir hash needs to take into account:
  - op schedule ss
  - dataflow
  - deviceInfo
  - userOptions

 This test verifies that this is the case
*/

onnx::ModelProto getProto() {
  // Build a basic onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();
  TensorInfo info{"FLOAT", std::vector<int64_t>{1}};
  std::vector<float> vals(1, -10.0f);
  ConstVoidData data = {vals.data(), info};
  auto inId0         = builder->addInitializedInputTensor(data);
  auto inId1         = builder->addInputTensor(info);
  auto outId         = aiOnnx.add({inId0, inId1});
  builder->addOutputTensor(outId);

  auto proto = builder->getModelProto();
  return io::getModelFromString(proto);
}

BOOST_AUTO_TEST_CASE(test0) {
  auto proto = getProto();
  auto isi   = InputShapeInfo();
  auto inId  = proto.graph().input()[0].name();
  auto outId = proto.graph().output()[0].name();
  auto df0   = DataFlow(1, {{outId, AnchorReturnType("ALL")}});
  std::vector<Loss *> losses{new L1Loss(outId, "l1", 0.1, ReductionType::SUM)};
  auto opt0      = ConstSGD(0.01);
  auto deviceMgr = DeviceManager::createDeviceManager();
  std::map<std::string, std::string> deviceOpts0{{"tilesPerIPU", "20"}};
  auto device0 = deviceMgr.createIpuModelDevice(deviceOpts0);

  // Now create the Ir instances and their hashes, for:
  // ir0 : the reference
  // ir1 : different op schedule
  // ir2 : different anchor return type
  // ir3 : different batches-per-step
  // ir4 : different archor id
  // ir5 : different deviceInfo
  // ir6 : different user options (1)
  // ir7 : different user options (2)

  // 0. The reference Ir
  Ir ir0;
  ir0.prepare({proto, isi, df0, losses, &opt0, *device0, {}, Patterns()});
  std::size_t irHash0_0 = std::hash<Ir>{}(ir0);
  std::size_t irHash0_1 = std::hash<Ir>{}(ir0);

  // 1. Same as ref, except different optimizer - Op schedule no longer the same
  Ir ir1;
  auto opt1 = SGD({{"defaultLearningRate", {0.01, false}}});
  ir1.prepare({proto, isi, df0, losses, &opt1, *device0, {}, Patterns()});
  std::size_t irHash1 = std::hash<Ir>{}(ir1);

  // 2. Same as reference, except different anchor return type
  Ir ir2;
  auto df1 = DataFlow(1, {{outId, AnchorReturnType("FINAL")}});
  ir2.prepare({proto, isi, df1, losses, &opt0, *device0, {}, Patterns()});
  std::size_t irHash2 = std::hash<Ir>{}(ir2);

  // 3. Same as reference, except different batches-per-step
  Ir ir3;
  auto df2 = DataFlow(2, {{outId, AnchorReturnType("ALL")}});
  ir3.prepare({proto, isi, df2, losses, &opt0, *device0, {}, Patterns()});
  std::size_t irHash3 = std::hash<Ir>{}(ir3);

  // 4. Same as reference, except different anchor id
  Ir ir4;
  auto df3 = DataFlow(2, {{inId, AnchorReturnType("ALL")}});
  ir4.prepare({proto, isi, df3, losses, &opt0, *device0, {}, Patterns()});
  std::size_t irHash4 = std::hash<Ir>{}(ir4);

  // 5. Same as reference, except different device options
  Ir ir5;
  std::map<std::string, std::string> deviceOpts1{{"tilesPerIPU", "40"}};
  auto device1 = deviceMgr.createIpuModelDevice(deviceOpts1);
  ir5.prepare({proto, isi, df0, losses, &opt0, *device1, {}, Patterns()});
  std::size_t irHash5 = std::hash<Ir>{}(ir5);

  // 6. Same as reference, except different session (user) options, (1)
  Ir ir6;
  SessionOptions uopt0;
  uopt0.ignoreData = true;
  ir6.prepare({proto, isi, df0, losses, &opt0, *device0, uopt0, Patterns()});
  std::size_t irHash6 = std::hash<Ir>{}(ir6);

  // 7. Same as reference, except different session (user) options, (2)
  Ir ir7;
  SessionOptions uopt1;
  uopt1.rearrangeAnchorsOnHost = false;
  ir7.prepare({proto, isi, df0, losses, &opt0, *device0, uopt1, Patterns()});
  std::size_t irHash7 = std::hash<Ir>{}(ir7);

  // The checks
  BOOST_CHECK(irHash0_0 == irHash0_1); // Ir hashing is deterministic
  BOOST_CHECK(irHash0_0 != irHash1);   // Different optimizer, different hash
  BOOST_CHECK(irHash0_0 != irHash2);   // Different a.r.t, different hash
  BOOST_CHECK(irHash0_0 != irHash3);   // Different b.p.s, different hash
  BOOST_CHECK(irHash0_0 != irHash4);   // Different anchor id, different hash
  BOOST_CHECK(irHash0_0 != irHash5);   // Different device opts, different hash
  BOOST_CHECK(irHash0_0 != irHash6); // Different user opts, different hash (1)
  BOOST_CHECK(irHash0_0 != irHash6); // Different user opts, different hash (2)
}
