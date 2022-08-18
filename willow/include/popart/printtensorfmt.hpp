// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PRINTTENSORFMT_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PRINTTENSORFMT_HPP_

namespace popart {

class PrintTensorFmtImpl;

/**
 * PrintTensorFmt specifies how the print output of PrintTensor should be
 * formatted.
 */
class PrintTensorFmt {
public:
  /**
   * Floating point format to use when printing a tensor
   */
  enum class FloatFormat {
    Auto       = 0, /* Automatically determine the format through analysis */
    Fixed      = 1, /* Use fixed point e.g. -100.00 */
    Scientific = 2, /* Use scientific notation e.g. -1.123e+10 */
    None       = 3, /* Do not display all elements with the same format */
  };

  const unsigned summariseThreshold;
  const unsigned edgeItems;
  const unsigned maxLineWidth;
  const unsigned digits;
  const FloatFormat floatFormat = FloatFormat::Auto;
  const char separator;
  const char openBracket;
  const char closeBracket;

  /**
   * PrintTensorFmt specifies how the print output of PrintTensor should be
   * formatted.
   *
   * The default output format will split large lines, print all elements in the
   * same format, pad elements so that they align and summarise large tensors.
   *
   * You can use the `disableFormatting` constructor to disable all types of
   * formatting.
   *
   * \param summariseThreshold (default 1000) If the number of elements of the
   * tensor exceeds this threshold the output will be summarised. Only the edge
   * elements will be displayed with an ellipsis indicating skipped elements.
   * A value of 0 will disable summarisation.
   *
   * \param edgeItems (default 3) number of edge elements to include at the
   * beginning and end when summarisation is enabled
   *
   * \param maxLineWidth (default 75) lines longer than this limit will be split
   * across multiple lines. A value of 0 will disable line splitting.
   *
   * \param digits (default 8) number of digits to display. For integers this
   * limit can be exceeded if any number is large enough. For floating points
   * this does not include the exponent. The number of digits is used in
   * conjunction analysis of the tensor to determine the width of each element
   * to align all elements when printed. A value of 0 disables this analysis
   * and each elements will be printed in an unaligned format.
   *
   * \param floatFormat (default Auto) determines the floating point format to
   * use. Automatic mode determines the appropriate format based on the data.
   * If `digits==0` this option is disregarded and the floatFormat is set to
   * `None`.
   *
   * \param separator (default space) character used to delininate values.
   *
   * \param openBracket (default square bracket) character used to open a
   * tensor.
   *
   * \param closeBracket (default square bracket) character used to close a
   * tensor.
   *
   */
  PrintTensorFmt(unsigned summariseThreshold = 1000,
                 unsigned edgeItems          = 3,
                 unsigned maxLineWidth       = 75,
                 unsigned digits             = 8,
                 FloatFormat floatFormat     = FloatFormat::Auto,
                 char separator              = ' ',
                 char openBracket            = '[',
                 char closeBracket           = ']')
      : summariseThreshold(summariseThreshold), edgeItems(edgeItems),
        maxLineWidth(maxLineWidth), digits(digits), floatFormat(floatFormat),
        separator(separator), openBracket(openBracket),
        closeBracket(closeBracket) {}
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PRINTTENSORFMT_HPP_
