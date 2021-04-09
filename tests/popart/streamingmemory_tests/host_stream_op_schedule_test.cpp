// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE ScheduleHostOpTest

#include <../random_util.hpp>
#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/call.hpp>
#include <popart/op/dynamic/dynamicslice.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/init.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/remote.hpp>
#include <popart/opmanager.hpp>
#include <popart/optimizer.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/region.hpp>
#include <popart/session.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

#include <algorithm>
#include <map>
#include <tuple>
#include <vector>

using namespace popart;

template <class T> std::vector<T *> getOpsOfType(Ir &ir) {
  std::vector<T *> ops;
  for (auto &id_op : ir.getMainGraphOps()) {
    auto op = id_op.second.get();
    if (op->isConvertibleTo<T>()) {
      ops.push_back(dynamic_cast<T *>(op));
    }
  }
  return ops;
}

void createAndRun(bool hostIO = false) {

  // Try to create a simple host copy
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  int bps   = 3;
  int steps = 10;

  // we will generate random initializations
  int seed = 1337;
  DefaultRandomEngine eng(seed);
  UniformRealDistribution<float> fdis(-1.f, 1.f);

  TensorInfo tinfo{"FLOAT", std::vector<int64_t>{2, 2}};
  std::vector<float> input(tinfo.nelms() * bps * steps);
  for (auto &val : input) {
    val = fdis(eng);
  }
  TensorInfo linfo{"INT32", std::vector<int64_t>{2}};

  std::vector<float> w_init(tinfo.nelms());
  for (auto &val : w_init) {
    val = fdis(eng);
  }

  auto i = builder->addInputTensor(tinfo, "A_input");
  auto l = builder->addInputTensor(linfo, "label");
  auto c = builder->addInitializedInputTensor({w_init.data(), tinfo}, "weight");

  auto a0 = aiOnnx.matmul({i, c});

  auto sm  = aiOnnx.softmax({a0}, 1);
  auto nll = builder->aiGraphcoreOpset1().nllloss({sm, l});

  auto out_id = nll;

  builder->addOutputTensor(out_id);

  SessionOptions opts;

  opts.useOverlappedIO = hostIO;

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);
  auto art        = AnchorReturnType("All");
  auto dataFlow   = DataFlow(bps, {{out_id, art}, {i, art}});
  auto device     = createTestDevice(TEST_TARGET);

  // inputs:
  popart::NDArrayWrapper<float> A_wrapper(
      input.data(), TensorInfo("FLOAT", std::vector<int64_t>{bps, 2, 2}));

  auto carriedWrapper = A_wrapper;

  // prepare the anchors
  int32_t rawLabel[bps * 2];
  int32_t j = 1;
  for (auto &val : rawLabel) {
    val = j;
    j += 1;
  }
  popart::NDArrayWrapper<int32_t> lWrapper(rawLabel, {bps, 2});

  std::map<popart::TensorId, popart::IArray &> inputs = {{i, A_wrapper},
                                                         {l, lWrapper}};

  std::vector<float> raw_out(bps);
  popart::NDArrayWrapper<float> B_wrapper(raw_out.data(),
                                          std::vector<int64_t>{bps});

  std::map<popart::TensorId, popart::IArray &> anchors = {{out_id, B_wrapper},
                                                          {i, A_wrapper}};
  auto optimizer = popart::ConstSGD(0.01f);

  if (device != nullptr) {
    auto session = popart::TrainingSession::createFromOnnxModel(
        proto,
        dataFlow,
        out_id,
        optimizer,
        device,
        popart::InputShapeInfo(),
        opts,
        popart::Patterns(PatternsLevel::Default));
    session->prepareDevice();
    popart::StepIO stepio(inputs, anchors);

    std::vector<Op *> schedule =
        session->getIr().getOpSchedule({}, RequireOptimalSchedule::Yes);
    int initOpscount = 0;
    int hlOpscount   = 0;

    std::string schedule_str = "";
    for (auto op : schedule) {
      std::cout << op->debugName() << std::endl;
      if (op->opid == Onnx::CustomOperators::Init_1) {
        initOpscount += 1;
        schedule_str += "i";
      } else if (op->opid == Onnx::CustomOperators::HostLoad) {
        hlOpscount += 1;
        schedule_str += "l";
      } else {
        schedule_str += "_";
      }
    }
    if (hostIO) {
      BOOST_CHECK(hlOpscount == 2);
      BOOST_CHECK(initOpscount == 2);

      std::cout << schedule_str << std::endl;
      // Find occurences of il in schedule_str
      int count = 0;
      auto pos  = schedule_str.find("il", 0);
      while (pos != std::string::npos) {
        count++;
        pos = schedule_str.find("il", pos + 2);
      }
      BOOST_CHECK(count == 2);
    } else {
      BOOST_CHECK(hlOpscount == 0);
      BOOST_CHECK(initOpscount == 0);
    }
  }
}

BOOST_AUTO_TEST_CASE(ScheduleHostOpTest0) { createAndRun(true); }
// BOOST_AUTO_TEST_CASE(ScheduleHostOpTest1) { createAndRun(false); }