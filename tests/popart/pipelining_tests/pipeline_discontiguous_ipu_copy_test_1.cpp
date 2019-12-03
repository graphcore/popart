#define BOOST_TEST_MODULE PipelineTrainingTest0

#include <algorithm>
#include <boost/test/unit_test.hpp>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/devicemanager.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/session.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

using namespace popart;

// A check that the ContiguateIpuCopyIndicesPattern operates as expected
// in this instance:
//
// The model
// ---------
//            input0    input1
//              |         |
// ipu0/ps0    Add -------|
//              |         |
// ipu1/ps1  Identity     |
//              |         |
// ipu2/ps2    Add -------|
//              |         |
// ipu3/ps3  Identity     |
//              |         |
// ipu4/ps4    Add -------
//              |
//            output0
//
//
// The IpuCopy transforms
// ----------------------
// We expect the IPU copy transforms/patterns to modify the graph
// in the following order:
//
// 1. Copy to input1 to each IPU it is consumed on:
//
//            input0                  input1
//              |                       |
// ipu0/ps0    Add ---------------------|
//              |                       |
//             copy                     |
//              |                       |
// ipu1/ps1  Identity                   |
//              |                       |
//             copy                     |
//              |                       |
// ipu2/ps2    Add -- input1_c2 - copy -|
//              |                       |
//             copy                     |
//              |                       |
// ipu3/ps3  Identity                   |
//              |                       |
//             copy                     |
//              |                       |
// ipu4/ps4    Add -- input1_c4 - copy -|
//              |
//            output0
//
// 2. Order the IPU copies such that each IPU input1 is consumed from
//    copies from the closest, smaller IPU number:
//
//            input0                           input1
//              |                                |
// ipu0/ps0    Add ------------------------------|
//              |                                |
//             copy                              |
//              |                                |
// ipu1/ps1  Identity                            |
//              |                                |
//             copy                              |
//              |                                |
// ipu2/ps2    Add ----------- input1_c2 - copy -|
//              |                  |
//             copy                |
//              |                  |
// ipu3/ps3  Identity              |
//              |                  |
//             copy                |
//              |                  |
// ipu4/ps4    Add -- input1_c4 - copy
//              |
//            output0
//
// 3. Contiguate IPU copies:
//
//            input0     input1
//              |          |
// ipu0/ps0    Add --------|
//              |          |
//             copy       copy
//              |          |
// ipu1/ps1  Identity  input1_c1
//              |          |
//             copy       copy
//              |          |
// ipu2/ps2    Add --- input1_c2
//              |          |
//             copy       copy
//              |          |
// ipu3/ps3  Identity  input1_c3
//              |          |
//             copy       copy
//              |          |
// ipu4/ps4    Add --- input1_c4
//              |
//            output

std::vector<IpuCopyOp *> getIpuCopies(const Ir &ir) {
  std::vector<IpuCopyOp *> copies;
  for (const auto &x : ir.getMainGraphOps()) {
    auto ipuCopy = dynamic_cast<IpuCopyOp *>(x.second.get());
    if (ipuCopy) {
      copies.push_back(ipuCopy);
    }
  }
  return copies;
}

BOOST_AUTO_TEST_CASE(DiscontiguousIpuCopyTest1) {

  std::vector<int64_t> batchShape{5};
  TensorInfo batchInfo{"FLOAT", batchShape};

  // Construct graph
  auto builder     = Builder::create();
  auto aiOnnx      = builder->aiOnnxOpset9();
  auto aiGraphcore = builder->aiGraphcoreOpset1();
  auto input0      = builder->addInputTensor(batchInfo, "input0");
  auto input1      = builder->addInputTensor(batchInfo, "input1");

  auto add0 = aiOnnx.add({input0, input1}, "add0");
  auto id1  = aiOnnx.identity({add0}, "id1");
  auto add2 = aiOnnx.add({id1, input1}, "add2");
  auto id3  = aiOnnx.identity({add2}, "id3");
  auto add4 = aiOnnx.add({id3, input1}, "add4");
  builder->addOutputTensor(add4);

  // Annotate with vgraph and pipelining info
  builder->virtualGraph(add0, 0);
  builder->virtualGraph(id1, 1);
  builder->virtualGraph(add2, 2);
  builder->virtualGraph(id3, 3);
  builder->virtualGraph(add4, 4);

  builder->pipelineStage(add0, 0);
  builder->pipelineStage(id1, 1);
  builder->pipelineStage(add2, 2);
  builder->pipelineStage(id3, 3);
  builder->pipelineStage(add4, 4);

  std::map<std::string, std::string> deviceOpts{{"numIPUs", "5"}};
  auto device =
      DeviceManager::createDeviceManager().createIpuModelDevice(deviceOpts);

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;
  opts.enablePipelining = true;

  Ir ir;
  ir.prepare({io::getModelFromString(builder->getModelProto()),
              InputShapeInfo(),
              DataFlow(12, {{add4, AnchorReturnType("ALL")}}),
              {},      // no loss
              nullptr, // no optimizer
              *device,
              opts,
              Patterns(PatternsLevel::DEFAULT).enablePostNRepl(false)});

  auto copies = getIpuCopies(ir);
  std::vector<std::pair<VGraphId, VGraphId>> input0SrcDsts;
  for (auto copy : copies) {
    auto srcTenId = copy->getSourceIpus().begin()->first;
    if (srcTenId.find("input1") != std::string::npos) {
      input0SrcDsts.push_back({copy->getSourceIpu(), copy->getDestIpu()});
    }
  }
  std::sort(input0SrcDsts.begin(), input0SrcDsts.end());

  // Check copies are contiguous
  std::cout << "input1 srcDsts: " << std::endl;
  for (VGraphId vgId = 0; vgId < input0SrcDsts.size(); vgId++) {
    auto srcDst = input0SrcDsts.at(vgId);
    auto src    = srcDst.first;
    auto dst    = srcDst.second;
    std::cout << "[ " << src << " ] --> [ " << dst << " ]" << std::endl;

    BOOST_CHECK(src == vgId);
    BOOST_CHECK(dst == vgId + 1);
  }

  BOOST_CHECK(copies.size() == 8);        // There are 8 IPU copies in total
  BOOST_CHECK(input0SrcDsts.size() == 4); // ... 4 of which copy input1
}
