// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_DOCS_PYDOCS_POPART_CUSTOM_HPP_
#define POPART_WILLOW_INCLUDE_POPART_DOCS_PYDOCS_POPART_CUSTOM_HPP_
/*
Define custom documentation strings here. These are accessed like the other
strings, except with DOC(custom, ...), see below for an example.
 */

#define __EXPAND(x) x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...) COUNT
#define __VA_SIZE(...) __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b) a##b
#define __CAT2(a, b) __CAT1(a, b)
#define __DOC1(n1) n1
#define __DOC2(n1, n2) n1##_##n2
#define __DOC3(n1, n2, n3) n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4) n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5) n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6) n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)                                     \
  n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                                               \
  __CAT2(                                                                      \
      __doc_,                                                                  \
      __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__)))
#define SINGLE_LINE_DOC(...)                                                   \
  __CAT2(                                                                      \
      __singlelinedoc_,                                                        \
      __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__)))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

// clang-format off
static const char *__doc_custom_Session_exampleFunction =
    R"doc(This is an example of a custom doc string for a function.
    It would be accessed with DOC(custom, Session, exampleFunction)
    in the last argument of the pybind11 binding.)doc";

static const char *__doc_custom_PyStepIO_class =
    R"doc(This class is an implementation of the `IStepIO interface <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_
    backed by user-provided dictionaries for both input and output.
    These dictionaries map TensorId values to numpy arrays for PopART
    to read from and write to, respectively.)doc";

static const char *__doc_custom_PyStepIO_init =
    R"doc(
        Construct a new PyStepIO instance.

        Args:
            inputs:
                A dictionary with an entry for every input tensor,
                comprising a TensorId for the `key` and
                a numpy array for a `value` for PopART to read from. The
                numpy arrays are assumed to be size-compatible
                with a tensor of shape
                [`replicationFactor`, `accumulationFactor`, `batchesPerStep`,
                `<tensor shape>`].

            outputs:
                A dictionary with an entry for every output tensor,
                comprising a TensorId for the `key` and a
                numpy array `value` to which PopART will write the
                associated data. The expected shape of this numpy array
                is explained in the
                `C++ API documentation for popart::AnchorReturnTypeId <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-flow>`_.
                The convenience method Session.initAnchorArrays()
                is typically used to create a dictionary with suitable
                arrays.
    )doc";

static const char *__doc_custom_PyStepIO_enableRuntimeAsserts =
    R"doc(
        Enable (or disable) run-time checks that check the sizes of the provided numpy arrays.

        Args:
            arg0:
                Flag to enable/disable checks
    )doc";

static const char *__doc_custom_PyStepIOCallback_class =
    R"doc(This class is an implementation of the `IStepIO interface <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_
    backed by user-provided callback functions. This class inherits from IStepIO
    and implements those member functions by delegating the logic to the
    callback functions passed in the constructor. This gives the user full
    control as to how data buffers are provisioned.")doc";

static const char *__doc_custom_PyStepIOCallback_init =
    R"doc(Construct a new PyStepIOCallback instance.

Args:
    input_callback:
        Callable object that the PyStepIOCallback instance will use when ``IStepIO::in()`` is called.
        See `IStepIO <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_ for details on how to implement this method.
    input_complete_callback:
        Callable object that the PyStepIOCallback instance will use when ``IStepIO::inComplete()`` is called.
        See `IStepIO <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_ for details on how to implement this method.
    output_callback:
        Callable object that the PyStepIOCallback instance will use when ``IStepIO::out()`` is called.
        See `IStepIO <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_ for details on how to implement this method.
    output_complete_callback:
        Callable object that the PyStepIOCallback instance will use when ``IStepIO::outComplete()`` is called.
        See `IStepIO <https://docs.graphcore.ai/projects/popart-cpp-api/en/latest/api-cpp.html#data-input-and-output-istepio>`_ for details on how to implement this method.
    )doc";

// clang-format on

#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

#endif // POPART_WILLOW_INCLUDE_POPART_DOCS_PYDOCS_POPART_CUSTOM_HPP_
