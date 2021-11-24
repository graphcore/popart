// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE NestedDotTest

#include <../random_util.hpp>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <filereader.hpp>
#include <onnxutil.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Dot_nested0) {

  auto getDotFiles = [](bool separateCallOpPdfs) {
    // Build an onnx model
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();

    auto opts               = SessionOptions();
    opts.enableOutlining    = true;
    opts.outlineThreshold   = 0.00001;
    opts.dotOpNames         = true;
    opts.separateCallOpPdfs = separateCallOpPdfs;
    opts.dotChecks.insert(DotCheck::Final);

    opts.logDir = "./nestedDotTest" + randomString(14);
    boost::filesystem::create_directories(opts.logDir);

    TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
    auto in0 = builder->addInputTensor(shape);
    auto out = aiOnnx.relu({in0});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.identity({out});
    out      = aiOnnx.relu({out});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.sigmoid({out});
    out      = aiOnnx.relu({out});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.sigmoid({out});
    out      = aiOnnx.relu({out});
    out      = aiOnnx.relu({out});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.relu({out});
    out      = aiOnnx.relu({out});
    out      = aiOnnx.exp({out});
    out      = aiOnnx.exp({out});

    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    out           = modelProto.graph().output(0).name();
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("All")}});
    auto device   = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                {},      // in inference mode, so no losses,
                nullptr, // and no optimizer
                *device,
                opts,
                Patterns().enableInPlace(true)});

    return io::getMatchFns(io::getCanonicalDirName(opts.logDir), ".dot");
  };

  auto dotFileNames = getDotFiles(false);
  BOOST_CHECK(dotFileNames.size() == 1);

  dotFileNames = getDotFiles(true);
  BOOST_CHECK(dotFileNames.size() > 1);
}
