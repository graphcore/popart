# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
## Strict warning level
if (MSVC)
    # Use the highest warning level for visual studio.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /w")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /w")

else()
    foreach(COMPILER C CXX)
        set(CMAKE_COMPILER_WARNINGS)
        list(APPEND CMAKE_COMPILER_WARNINGS

            # The -Werror, -Wall, -pedantic and -Wextra flags make it so a lot of bad coding practices
            # get flagged up as a compilation error (and hence will be picked up by continuous integration.
            #
            # NOTE: If you find you have a warning that you cannot readily avoid, please think twice before
            # changing these flags. Instead, consider turning off promotion to error for that specific warning
            # (-Wno-error=foo), or turning off that warning altogether (-Wno-foo) instead.
            -Werror
            -Wall
            -pedantic
            -Wextra
            # Additional warnings.
            -Werror=disabled-optimization
            -Werror=endif-labels
            -Werror=format=2
            -Werror=unreachable-code
            # Warnings we don't want to promote to errors (but keep as warnings).
            -Wno-error=undef
            -Wno-error=shadow
            -Wno-error=deprecated-declarations
            # Warnings we want to silence altogether.
            -Wno-double-promotion
            -Wno-missing-noreturn
            -Wno-sign-compare
            -Wno-unused-parameter
        )

        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "Clang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                # Adding these to clang only as g++ doesn't have these warnings.
                -Wextra-semi

                # Enable spurious semicolon errors to match g++'s pedantic -Wpedantic behaviour.
                -Wno-nested-anon-types
                -Wno-shorten-64-to-32

                # Not sure we actually need these now -Weverything is off.
                -Wno-exit-time-destructors
                -Wno-c++98-compat
                -Wno-c++98-compat-pedantic
                -Wno-padded
                -Wno-weak-vtables
                -Wno-sign-conversion
                -Wno-covered-switch-default
                -Wno-global-constructors
            )
        elseif (CMAKE_${COMPILER}_COMPILER_ID MATCHES "GNU")
            list(APPEND CMAKE_COMPILER_WARNINGS
                # Avoid g++ erroring over ASCII graphs in comments.
                -Wno-comment

                # Turn off problematic warnings.
                -Wno-strict-aliasing
                -Wno-stringop-overflow
                -Wno-maybe-uninitialized

                # Turn off more warnings that are exclusive to g++.
                -Wno-missing-field-initializers
                -Wno-sign-conversion
            )
        endif()

        if (CMAKE_${COMPILER}_COMPILER_ID MATCHES "AppleClang")
            list(APPEND CMAKE_COMPILER_WARNINGS
                # Turn off more warnings for apple clang.
                # Seems like best practise is to always return by value,
                # c11-rvalues-and-move-semantics-confusion-return-statement
                -Wno-return-std-move-in-c++11
              )
        endif()

        add_compile_options(${CMAKE_COMPILER_WARNINGS})
    endforeach()
endif ()

