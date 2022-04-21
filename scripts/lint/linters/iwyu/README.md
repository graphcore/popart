# Include what you use

`include-what-you-use` (`IWYU`) is a program used to remove superfluous includes, add missing includes and suggest forward declarations where appropriate in C++ files. It is set up as a `pre-commit` hook for PopART, and can be directly called in one of the following ways from the home directory:

```sh
# Runs on all of PopART
pre-commit run iwyu --all-files

# Runs on specific files
pre-commit run iwyu --files <file1> <file2> ...
```

## Setup

`IWYU` needs to know where the build directory is, so that it can access the include directories of the view. This information is taken from the `POPLAR_VIEW_BUILD_DIR` environment variable which needs to be set by the user. No additional setup should be necessary.

Using `IWYU` with `popsdk` is not supported at the moment; you can instead skip the hook by setting the environment variable `SKIP=iwyu`.

## Fixing suggestions

As `IWYU` has no way to know for sure which headers belong where, it doesn't automatically fix it's suggestions, instead requiring them to be applied manually. The user can then decide whether the changes are acceptable, or whether they should update the mappings to disable the suggestion (as explained below)

In most cases there won't be many changes needed. `IWYU` provides line numbers for removal suggestions, and reasons for including files in a very nice human readable format, hence making changes is relatively simple.

If you have a lot of changes suggested and want to include all of them, you can use the `fix_includes.py` tool that is part of the `include-what-you-use` repository. It's options are clearly listed in the [`IWYU` documentation](https://github.com/include-what-you-use/include-what-you-use#applying-fixes).

## Dealing with undesirable suggestions

If `IWYU` makes undesirable suggestions for a particular file you have a couple options to deal with them.

### Exclude file from being linted

If you think that `IWYU` should not run on the file (for example if the file is just a collection of header includes for legacy reasons), you can exclude it in `.pre-commit-config.yaml` by adding it to the `excludes` list under the `iwyu` hook.

### Use mappings

You can also add a new mapping in one of the `*.imp` files in `scripts/lint/linters/iwyu/` to specify where certain symbols should / could be included from. The following mapping rules are available to use:

* `{ include: ["a.hpp", "public", "b.hpp", "public"] }` specifies that if `b.hpp` is present, `a.hpp` should not be included. But `a.hpp` can be included when `b.hpp` is not included.
* `{ include: ["a.hpp", "private", "b.hpp", "public"] }` specifies that `a.hpp` should always be included through `b.hpp`.
* `{ symbol: ["some::symbol", "private", "b.hpp", "public"] }` specifies that `some::symbol` should always be included through `b.hpp`
* `{ ref: "c.imp" }` allows us to include rules from `c.imp`

Additionally regex patterns are supported for the include case when the string starts with an `@` symbol:

```text
# Matches both angle brackets and quotes
{ include: ["@[\"<]popart/onnxoperators.gen.hpp[\">]", "private", "\"popart/operators.hpp\"", "public"]}
# Matches any header that starts with '<boost/icl/'
{ include: ["@<boost/icl/.*>", "public", "<boost/icl/interval.hpp>", "public"] },
```

You can find plenty of examples in the `*.imp` files in `scripts/lint/linters/iwyu`.

### Use IWYU pragmas

You can also use `IWYU` pragmas as described in the [`IWYU` documentation](https://github.com/include-what-you-use/include-what-you-use/blob/master/docs/IWYUPragmas.md).
