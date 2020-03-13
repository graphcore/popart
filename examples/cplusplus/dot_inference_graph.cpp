// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// A tool for visualising the Ir (PopART's Intermediate representation).
// as a .dot file which can be compiled into a .pdf file.

#include <boost/program_options.hpp>
#include <iostream>
#include <vector>
#include <popart/builder.hpp>
#include <popart/filereader.hpp>
#include <popart/ir.hpp>
#include <popart/onnxutil.hpp>
#include <popart/sessionoptions.hpp>

using namespace popart;
namespace po = boost::program_options;


class Options {
public:
  Options(int argc, char **argv);

  std::string modelPath() const { return vm["model-path"].as<std::string>(); }
  std::string outputDir() const { return vm["output"].as<std::string>(); }
  bool compileDotsToPDF() const { return vm.count("compile-pdf"); }
  int opCount() const { return vm["op-count"].as<int>(); }

private:
  void printHelp(const po::options_description &desc) const;

  po::variables_map vm;
};

int main(int argc, char **argv) {
  const auto opts = Options(argc, argv);

  logging::ir::info("{} user command-line parameters received", argc - 1);

  SessionOptions sessionOpts;
  sessionOpts.finalDotOp = opts.opCount();

  auto modelProto = onnxutil::getModelProto(opts.modelPath());
  std::vector<std::string> dotStrings;

  // Only the FINAL .dot file. Append others here as required.
  sessionOpts.dotChecks.insert(DotCheck::FINAL);

  sessionOpts.logDir = opts.outputDir();

  if (modelProto.graph().output_size() == 0) {
    throw error("Cannot generate dot for ONNX graph with no output");
  }

  auto out       = modelProto.graph().output(0).name();
  auto dataFlow  = DataFlow(1, {{out, AnchorReturnType("ALL")}});
  auto cpuDevice = DeviceManager::createDeviceManager().createCpuDevice();

  Ir ir;
  ir.prepare({modelProto,
              InputShapeInfo(),
              dataFlow,
              {},      // in inference mode, so no losses,
              nullptr, // and no optimizer.
              *cpuDevice,
              sessionOpts,
              Patterns(PatternsLevel::NONE).enableInPlace(true)});

  // verify that the dot files have been created
  auto dotFileNames =
      io::getMatchFns(io::getCanonicalDirName(sessionOpts.logDir), ".dot");
  if (dotFileNames.size() == 0) {
    throw error("Error in dot_inference_graph, no .dot files in {}",
                sessionOpts.logDir);
  }

  // dot -Tpdf -o x.pdf x.dot
  if (opts.compileDotsToPDF() == true) {
    for (auto check : sessionOpts.dotChecks) {
      auto dot_string = getDotCheckString(check);
      std::stringstream command_ss;
      command_ss << "dot "
                 << " -Tpdf "
                 << " -o "
                 << io::appendDirFn(sessionOpts.logDir, dot_string + ".pdf")
                 << " "
                 << io::appendDirFn(sessionOpts.logDir, dot_string + ".dot");
      std::string command = command_ss.str();
      int ran             = std::system(command.c_str());
      std::cout << command << " returned with status " << ran << std::endl;
    }
  }
  return 0;
}

Options::Options(int argc, char **argv) {
  po::options_description desc("dot_inference_graph");
  // clang-format off
    desc.add_options()
      ("help", "print this message")
      ("output,o",
       po::value<std::string>()->default_value("."),
       "path of the output directory")
      ("compile-pdf", "compile dot->pdf(s)")
      ("op-count,c",
       po::value<int>()->default_value(10000),
       "the number of Ops to write to the .dot file")
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
    std::cerr << "dot_inference_graph: " << e.what()
              << ". See 'dot_inference_graph --help'\n";
    exit(1);
  }

  if (vm.count("help")) {
    printHelp(desc);
    exit(0);
  }

  if (!vm.count("model-path")) {
    std::cout << "dot_inference_graph error: no model\n";
    printHelp(desc);
    exit(0);
  }
}

void Options::printHelp(const po::options_description &desc) const {
  std::stringstream ss;
  ss << desc
     << "\n  Example usage : ./dot_inference_graph /path/to/model.onnx -o . "
        "--compile-pdf -c 121"
        "\n    will generate a pdf for model.onnx in the current directory.";
  std::cout << ss.str() << std::endl;
}
