// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// A tool for visualising the Ir (PopART's Intermediate representation).
// as a .dot file which can be compiled into a .pdf file.
#define BOOST_TEST_MODULE dot_inference_graph_test

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <cstdlib>
#include <filereader.hpp>
#include <iostream>
#include <memory>
#include <onnx/onnx_pb.h>
#include <onnxutil.hpp>
#include <set>
#include <string>
#include <vector>
#include <popart/builder.hpp>
#include <popart/devicemanager.hpp>
#include <popart/ir.hpp>
#include <popart/sessionoptions.hpp>
#include <popart/tensorinfo.hpp>

#include "../random_util.hpp"
#include "popart/builder.gen.hpp"
#include "popart/dataflow.hpp"
#include "popart/inputshapeinfo.hpp"
#include "popart/names.hpp"
#include "popart/patterns/patterns.hpp"

using namespace popart;

std::string create_model() {
  // Generate an ONNX inference model
  auto builder = popart::Builder::create();

  auto aiOnnx = builder->aiOnnxOpset9();

  // Add input tensors
  popart::TensorInfo inputInfo{"FLOAT", std::vector<int64_t>{2}};
  auto a = builder->addInputTensor(inputInfo);
  auto b = builder->addInputTensor(inputInfo);

  // Add operation
  auto o = aiOnnx.add({a, b});

  // Add output tensor
  builder->addOutputTensor(o);

  std::string modelPath = "./model" + randomString(14) + ".onnx";
  builder->saveModelProto(modelPath);

  return modelPath;
}

int runTest(std::string modelPath,
            const std::string &outputDir,
            bool convertPdf) {

  SessionOptions sessionOpts;
  sessionOpts.finalDotOp = 1;

  auto modelProto = onnxutil::getModelProto(modelPath);

  // Only the FINAL .dot file. Append others here as required.
  sessionOpts.dotChecks.insert("Final");

  sessionOpts.logDir = "./dotTestTmp" + randomString(14);
  boost::filesystem::create_directory(sessionOpts.logDir);

  BOOST_CHECK(modelProto.graph().output_size() > 0);

  auto out       = modelProto.graph().output(0).name();
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("All")}});
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},      // in inference mode, so no losses,
              nullptr, // and no optimizer.
              *cpuDevice,
              sessionOpts,
              Patterns(PatternsLevel::Minimal).enableInPlace(true)});

  // verify that the dot files have been created
  auto dotFileNames =
      io::getMatchFns(io::getCanonicalDirName(sessionOpts.logDir), ".dot");
  BOOST_CHECK(dotFileNames.size() > 0);

  // dot -Tpdf -o x.pdf x.dot
  // Add this if you want to convert the dot file to pdf. GraphViz cmd line tool
  // isn't installed on the buildbot so disabled here.
  if (convertPdf) {
    for (auto check : sessionOpts.dotChecks) {
      auto dot_string = check;
      std::stringstream command_ss;
      command_ss << "dot "
                 << " -Tpdf "
                 << " -o "
                 << io::appendDirFn(sessionOpts.logDir, dot_string + ".pdf")
                 << " "
                 << io::appendDirFn(sessionOpts.logDir, dot_string + ".dot");
      std::string command = command_ss.str();
      int ran             = std::system(command.c_str());

      BOOST_CHECK(ran == 0);
    }
  }
  return 0;
}

BOOST_AUTO_TEST_CASE(dot_inference_graph_test) {
  std::string fn = create_model();

  runTest(fn, "/tmp", false);
}
