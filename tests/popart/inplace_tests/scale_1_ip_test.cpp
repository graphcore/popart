// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Scale1InplaceTest

#include <boost/test/unit_test.hpp>
#include <vector>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/op/l1.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>
#include <popart/topocons.hpp>

using namespace popart;

BOOST_AUTO_TEST_CASE(Inplace_Scale1) {

  //                 |-- slice [0,3 ], [0,3] - scale 1.1 -|
  //                 |-- slice [3,6 ], [1,5] - scale 2.2 -|
  //                 |-- slice [6,9 ], [0,5] - scale 3.3 -| - concat [3,15] -|
  //                 |-- slice [7,10], [1,4] - scale 4.4 -|                 /
  //                 |                                                     /
  // in0 (10 x 10) --|                                                    /
  //                 |                                                   |- MM
  //                 |                                                  /   |
  //                 |-- slice [0,10], [1,4 ] - scale 7.7 -| ----------/   /
  //                                                                      /
  //                                                                     /
  //                                                                    /
  //                                                            out [10, 3]
  //
  //
  // we make slices have the highest priorities, so they are inplaced.
  // we make concat have negative priority, so it is never inplaced.
  // As for the scales,
  //
  // in both tests, prioriry = 10 + scale factor, with scale factor being
  // positive or negative.
  //
  // test 1: priority = scale,
  //         in which case only the 7.7 is inplaced
  //
  // test 2: priority = 10.0 -scale,
  //         in which case 1.1, 2.2, 3.3 are inplaced

  auto test = [](bool branch77High) {
    auto builder     = Builder::create();
    auto aiOnnx      = builder->aiOnnxOpset9();
    auto aiGraphcore = builder->aiGraphcoreOpset1();
    TensorInfo shape0{"FLOAT", std::vector<int64_t>{10, 10}};
    auto in0 = builder->addInputTensor(shape0);

    auto sl11 = aiOnnx.slice({in0}, {3, 3}, {0, 0}, {0, 1});
    builder->setInplacePreferences(sl11, {{"SliceInplace", 1000.0}});
    float factor11 = 1.1 * (2 * branch77High - 1);
    auto sc11      = aiGraphcore.scale({sl11}, factor11);
    builder->setInplacePreferences(sc11, {{"ScaleInplace", 10.0 + factor11}});

    auto sl22 = aiOnnx.slice({in0}, {6, 5}, {3, 1}, {0, 1});
    builder->setInplacePreferences(sl22, {{"SliceInplace", 1000.}});
    float factor22 = 2.2 * (2 * branch77High - 1);
    auto sc22      = aiGraphcore.scale({sl22}, factor22);
    builder->setInplacePreferences(sc22, {{"ScaleInplace", 10.0 + factor22}});

    auto sl33 = aiOnnx.slice({in0}, {9, 5}, {6, 0}, {0, 1});
    builder->setInplacePreferences(sl33, {{"SliceInplace", 1000.0}});
    float factor33 = 3.3 * (2 * branch77High - 1);
    auto sc33      = aiGraphcore.scale({sl33}, factor33);
    builder->setInplacePreferences(sc33, {{"ScaleInplace", 10.0 + factor33}});

    auto sl44 = aiOnnx.slice({in0}, {10, 4}, {7, 1}, {0, 1});
    builder->setInplacePreferences(sl44, {{"SliceInplace", 1000.0}});
    float factor44 = 4.4 * (2 * branch77High - 1);
    auto sc44      = aiGraphcore.scale({sl44}, factor44);
    builder->setInplacePreferences(sc44, {{"ScaleInplace", 10.0 + factor44}});

    auto topCon = aiOnnx.concat({sc11, sc22, sc33, sc44}, 1);
    builder->setInplacePreferences(topCon, {{"ConcatInplace", -10.0}});

    auto sl77 = aiOnnx.slice({in0}, {10, 4}, {0, 1}, {0, 1});
    builder->setInplacePreferences(sl77, {{"SliceInplace", 1000.0}});
    float factor77 = 7.7 * (2 * branch77High - 1);
    auto sc77      = aiGraphcore.scale({sl77}, factor77);
    builder->setInplacePreferences(sc77, {{"ScaleInplace", 10.0 + factor77}});

    auto out = aiOnnx.matmul({sc77, topCon});
    builder->addOutputTensor(out);

    auto proto      = builder->getModelProto();
    auto modelProto = io::getModelFromString(proto);

    // Create the IR
    auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

    auto device = createTestDevice(TEST_TARGET);

    Ir ir;
    ir.prepare({modelProto,
                InputShapeInfo(),
                dataFlow,
                {},
                nullptr,
                *device,
                {},
                Patterns(PatternsLevel::NONE).enableInPlace(true)});

    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Concat).size() == 1);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ConcatInplace).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::AiOnnx::OpSet9::Slice).size() == 0);
    BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::SliceInplace).size() == 5);

    if (branch77High) {
      BOOST_CHECK(ir.opsOfType(Onnx::AiGraphcore::OpSet1::Scale).size() == 4);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ScaleInplace).size() ==
                  1);
    } else {
      BOOST_CHECK(ir.opsOfType(Onnx::AiGraphcore::OpSet1::Scale).size() == 2);
      BOOST_CHECK(ir.opsOfType(Onnx::CustomOperators::ScaleInplace).size() ==
                  3);
    }
  };

  test(true);
  test(false);
}
