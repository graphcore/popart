// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// A tool for visualising the Ir (PopART's Intermediate representation)
// as a .dot file which can be compiled into a .pdf file.
// This tool supports training and as well as inference.

#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/graphtransformer.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/nll.hpp>
#include <popart/optimizer.hpp>
#include <popart/sessionoptions.hpp>

using namespace popart;
namespace po = boost::program_options;

// TODO: improved visualization (T5144)

class Options {
public:
  Options(int argc, char **argv);

  std::string modelPath() const { return vm["model-path"].as<std::string>(); }
  std::string outputDir() const { return vm["output"].as<std::string>(); }
  bool compileDotsToPDF() const { return vm.count("compile-pdf"); }
  bool trainWithNll() const { return vm.count("nll"); }
  int startOp() const { return vm["start-op"].as<int>(); }
  int endOp() const { return vm["end-op"].as<int>(); }

private:
  void printHelp(const po::options_description &desc) const;

  po::variables_map vm;
};

int main(int argc, char **argv) {
  const auto opts = Options(argc, argv);

  logging::ir::info("modelPath set to {}", opts.modelPath());
  logging::ir::info("outputDir set to {}", opts.outputDir());
  logging::ir::info("compileDotsToPDF set to {}", opts.compileDotsToPDF());
  logging::ir::info("trainWithNll set to {}", opts.trainWithNll());

  SessionOptions session_opts;
  session_opts.firstDotOp = opts.startOp();
  session_opts.finalDotOp = opts.endOp();

  GraphTransformer gt(opts.modelPath());
  gt.convertAllFixedPointInitializersToConstants();

  // The ONNX Model might have been exported with inference in mind.
  // In general this makes no difference, but for BatchNormalization
  // (and potentially other Operator types) the number of outputs
  // is 5 for training and 1 for inference. This call appends dummy
  // output names to BatchNormalization (and possibly makes other adjustments)
  if (opts.trainWithNll()) {
    gt.prepareNodesForTraining();
  }
  auto modelProtoString = gt.getModelProto();
  auto modelProto       = io::getModelFromString(modelProtoString);

  std::vector<std::string> dotStrings;

  // Currently only the FINAL .dot file. Append others here as required.
  session_opts.dotChecks.insert(DotCheck::FINAL);

  session_opts.logDir = opts.outputDir();

  if (modelProto.graph().output_size() == 0) {
    throw error("Cannot generate dot for ONNX graph with no output");
  }

  auto out      = modelProto.graph().output(0).name();
  auto dataFlow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  std::unique_ptr<Optimizer> optimizer;
  std::vector<std::unique_ptr<Loss>> up_losses;

  TensorId label = "label";
  InputShapeInfo isi;

  if (opts.trainWithNll()) {
    optimizer.reset(new ConstSGD(0.01f));

    if (modelProto.graph().output_size() != 1) {
      // requires extension for training with more than 1 output
      throw error("Cannot call dot_graph TRAIN, model only has one output");
    }

    // choosing an arbitrary shape for label, can't run shape inference now
    isi.add(label, {"INT32", std::vector<int64_t>{7}});

    up_losses.emplace_back(std::unique_ptr<Loss>(
        new NllLoss(out, label, "nllLossVal", ReductionType::SUM)));
  }

  std::vector<Loss *> losses;
  for (auto &x : up_losses) {
    losses.push_back(x.get());
  }
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              isi,
              dataFlow,
              losses,
              optimizer.get(),
              *cpuDevice,
              session_opts,
              Patterns(PatternsLevel::DEFAULT).enableInPlace(true)});

  // verify that the dot files have been created
  auto dotFileNames =
      io::getMatchFns(io::getCanonicalDirName(session_opts.logDir), ".dot");
  if (dotFileNames.size() == 0) {
    throw error("Error in dot_graph, no .dot files in {}", session_opts.logDir);
  }

  // dot -Tpdf -o x.pdf x.dot
  // (for all x.dot  in session_opts.logDir)
  if (opts.compileDotsToPDF() == true) {
    for (auto dotFn : popart::io::getMatchFns(session_opts.logDir, ".dot")) {
      std::string pdfFn = dotFn.substr(0, dotFn.size() - 4) + ".pdf";
      std::stringstream command_ss;
      command_ss << "dot "
                 << " -Tpdf "
                 << " -o " << pdfFn << " " << dotFn;
      std::string command = command_ss.str();
      int ran             = std::system(command.c_str());
      std::cout << command << " returned with status " << ran << std::endl;
    }
  }
  return 0;
}

Options::Options(int argc, char **argv) {
  po::options_description desc("dot_graph");
  // clang-format off
    desc.add_options()
      ("help", "print this message")
      ("output,o",
       po::value<std::string>()->default_value("."),
       "path of the output directory")
      ("compile-pdf", "compile dot->pdf(s)")
      ("nll", "train with NLL on output")
      ("start-op,s",
       po::value<int>()->default_value(0),
       "the index of the first op in the schedule to export")
      ("end-op,e",
       po::value<int>()->default_value(10000),
       "the index of the final op in the schedule to export")
    ;
  // clang-format on

  po::positional_options_description p;
  p.add("model-path", 1);

  po::options_description hidden("");
  hidden.add_options()(
      "model-path", po::value<std::string>(), "path to ONNX model file");

  po::options_description cmdline_options;
  cmdline_options.add(desc).add(hidden);

  try {

    po::store(po::command_line_parser(argc, argv)
                  .options(cmdline_options)
                  .positional(p)
                  .run(),
              vm);
    po::notify(vm);

  } catch (std::exception &e) {
    std::cerr << "dot_graph: " << e.what() << ". See 'dot_graph --help'\n";
    exit(1);
  }

  if (vm.count("help")) {
    printHelp(desc);
    exit(0);
  }

  if (!vm.count("model-path")) {
    std::cout << "dot_graph error: no model\n";
    printHelp(desc);
    exit(0);
  }
}

void Options::printHelp(const po::options_description &desc) const {
  std::stringstream ss;
  ss << desc
     << "\n  Example usage : ./dot_graph /path/to/model.onnx -o . --nll "
        "--compile-pdf -s 10 -e 100 "
        "\n    will generate a pdf for model.onnx in the current directory, "
        "all Ops 10->100.";
  std::cout << ss.str() << std::endl;
}
