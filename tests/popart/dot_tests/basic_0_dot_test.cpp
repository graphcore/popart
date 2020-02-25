#define BOOST_TEST_MODULE BasicDotTest

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/test/unit_test.hpp>
#include <random>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

std::string random_string(size_t length) {

  std::default_random_engine eng((std::random_device())());
  std::uniform_int_distribution<uint64_t> idis(
      0, std::numeric_limits<uint64_t>::max());

  auto randchar = [&idis, &eng]() -> char {
    const char charset[] = "0123456789"
                           "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[idis(eng) % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

BOOST_AUTO_TEST_CASE(Dot_basic0) {

  // Consider the series of Ops:
  //
  // (in0) -> [Relu] -> (h0)
  //       -> [Exp]  -> (preId)
  //       -> [Identity] -> (out),

  // Build an onnx model
  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  auto opts = SessionOptions();
  opts.dotChecks.insert(DotCheck::FWD0);
  opts.dotChecks.insert(DotCheck::FWD1);
  opts.dotChecks.insert(DotCheck::FINAL);

  opts.logDir = "./dotTestTmp" + random_string(14);
  boost::filesystem::create_directory(opts.logDir);

  TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
  auto in0   = builder->addInputTensor(shape);
  auto h0    = aiOnnx.relu({in0});
  auto preId = aiOnnx.exp({h0});
  auto out   = aiOnnx.identity({preId});
  builder->addOutputTensor(out);

  auto proto      = builder->getModelProto();
  auto modelProto = io::getModelFromString(proto);

  out           = modelProto.graph().output(0).name();
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto device   = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},      // in inference mode, so no losses,
              nullptr, // and no optimizer
              *device,
              opts,
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  // verify that there are 3 newly created dot_files
  auto dotFileNames =
      io::getMatchFns(io::getCanonicalDirName(opts.logDir), ".dot");
  BOOST_CHECK(dotFileNames.size() == 3);
}

// check that dotOpNames field functions correctly
BOOST_AUTO_TEST_CASE(Dot_dotOpNames0) {

  // For the simple model,
  // (in0) -> [Exp]  -> (preId)
  //       -> [Identity] -> (out),
  //
  // we give Exp the name,
  std::string expName = "sdgoimsdgpoisndglskdtjlsgilnsrkgnl";
  // and then test that setting .dotOpNames true (false)
  // does (does not) export the name to the .dot file

  auto getFullDotString = [expName](bool dotOpNames) {
    auto builder = Builder::create();
    auto aiOnnx  = builder->aiOnnxOpset9();
    auto opts    = SessionOptions();
    // just the one .dot file will be written
    opts.dotChecks.insert(DotCheck::BWD0);
    opts.dotOpNames = dotOpNames;
    opts.logDir     = "./dotTestTmp" + random_string(14);
    boost::filesystem::create_directory(opts.logDir);
    TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};
    auto in0   = builder->addInputTensor(shape);
    auto preId = aiOnnx.exp({in0}, expName);
    auto out   = aiOnnx.identity({preId});
    builder->addOutputTensor(out);
    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);
    out             = modelProto.graph().output(0).name();
    auto dataFlow   = DataFlow(1, {{out, AnchorReturnType("ALL")}});
    Ir ir;
    auto device = createTestDevice(TEST_TARGET);

    // note that we are not in training mode,
    // but BWD0 is still a valid checkpoint
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                {},      // in inference mode, so no losses,
                nullptr, // and no optimizer
                *device,
                opts,
                Patterns(PatternsLevel::NONE)});

    // verify that there is 1 newly created dot_file
    auto dotFileNames =
        io::getMatchFns(io::getCanonicalDirName(opts.logDir), ".dot");
    BOOST_CHECK(dotFileNames.size() == 1);

    // we have just verified that there is just the 1 .dot file,
    if (dotFileNames.size() == 1) {
      auto fn = dotFileNames.back();
      io::confirmRegularFile(fn);
      std::ifstream ifs(fn);
      return std::string((std::istreambuf_iterator<char>(ifs)),
                         (std::istreambuf_iterator<char>()));

    } else {
      throw error("dotFileNames not of size 1");
    }
  };

  auto fullDot = getFullDotString(true);
  BOOST_CHECK(fullDot.find(expName) != std::string::npos);

  fullDot = getFullDotString(false);
  BOOST_CHECK(fullDot.find(expName) == std::string::npos);
}

// check that fields firstDotOp and finalDotOp function correctly
BOOST_AUTO_TEST_CASE(Dot_dotStartEnd) {

  // The model:
  // (in0) -> [Identity]  -> (o0)
  //       -> [Identity]  -> (o1)
  //       -> [Identity]  -> (o2)
  //       -> [Identity]  -> (o3)
  //       -> [Identity]  -> (o4)
  //       -> [Identity]  -> (out),

  std::string name0    = "name0";
  std::string name1    = "name1";
  std::string name2    = "name2";
  std::string name3    = "name3";
  std::string name4    = "name4";
  std::string name_out = "name_out";

  auto getFullDotString =
      [name0, name1, name2, name3, name4, name_out](int start, int end) {
        auto builder = Builder::create();
        auto aiOnnx  = builder->aiOnnxOpset9();
        auto opts    = SessionOptions();

        // just the one .dot file will be written
        opts.dotChecks.insert(DotCheck::BWD0);
        opts.dotOpNames = true;
        opts.firstDotOp = start;
        opts.finalDotOp = end;
        opts.logDir     = "./dotTestTmp" + random_string(14);
        boost::filesystem::create_directory(opts.logDir);
        TensorInfo shape{"FLOAT", std::vector<int64_t>{1}};

        auto in0 = builder->addInputTensor(shape);
        auto o0  = aiOnnx.identity({in0}, name0);
        auto o1  = aiOnnx.identity({o0}, name1);
        auto o2  = aiOnnx.identity({o1}, name2);
        auto o3  = aiOnnx.identity({o2}, name3);
        auto o4  = aiOnnx.identity({o3}, name4);
        auto out = aiOnnx.identity({o4}, name_out);
        builder->addOutputTensor(out);

        auto proto      = builder->getModelProto();
        auto modelProto = io::getModelFromString(proto);
        out             = modelProto.graph().output(0).name();
        auto dataFlow   = DataFlow(1, {{out, AnchorReturnType("ALL")}});
        auto device     = createTestDevice(TEST_TARGET);
        Ir ir;

        // note that we are not in training mode,
        // but BWD0 is still a valid checkpoint
        ir.prepare({modelProto,
                    InputShapeInfo(),
                    dataFlow,
                    {},      // in inference mode, so no losses,
                    nullptr, // and no optimizer
                    *device,
                    opts,
                    Patterns(PatternsLevel::NONE)});

        // verify that there is 1 newly created dot_file
        auto dotFileNames =
            io::getMatchFns(io::getCanonicalDirName(opts.logDir), ".dot");
        BOOST_CHECK(dotFileNames.size() == 1);

        // we have just verified that there is just the 1 .dot file,
        if (dotFileNames.size() == 1) {
          auto fn = dotFileNames.back();
          io::confirmRegularFile(fn);
          std::ifstream ifs(fn);
          return std::string((std::istreambuf_iterator<char>(ifs)),
                             (std::istreambuf_iterator<char>()));

        } else {
          throw error("dotFileNames not of size 1");
        }
      };

  // everything is exported with these first and last
  auto fullDot = getFullDotString(-100, 100);
  BOOST_CHECK(fullDot.find(name0) != std::string::npos);
  BOOST_CHECK(fullDot.find(name_out) != std::string::npos);

  // just 1 is exported with these first and last
  fullDot = getFullDotString(1, 2);
  BOOST_CHECK(fullDot.find(name0) == std::string::npos);
  BOOST_CHECK(fullDot.find(name1) != std::string::npos);
  BOOST_CHECK(fullDot.find(name2) == std::string::npos);
}
