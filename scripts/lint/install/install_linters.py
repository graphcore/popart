#!/usr/bin/env python3
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import os
import sys
import subprocess
import argparse
import site
from pathlib import Path
from scripts.lint.install.check_versions import VersionChecker


def call_command(cmd: str) -> None:
    """Call a command line command and print the output continuously.

    Args:
        cmd (str): The command to run

    Raises:
        SystemExit: The error if the command returned a non-zero integer
    """
    process = subprocess.Popen(cmd.split(),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               cwd=Path(__file__).parent)
    for c in iter(lambda: process.stdout.read(1), b''):
        sys.stdout.buffer.write(c)
    return_code = None
    while return_code is None:
        return_code = process.poll()
    if return_code != 0:
        err_lines = process.stderr.readlines()
        for err_line in err_lines:
            print(err_line.decode("utf-8"))
        raise RuntimeError(
            f"{cmd} failed with return code {return_code}. See error message above for details"
        )


class Installer:
    """Class used to install linter binaries.
    """

    def __init__(self, install_dir: str) -> None:
        """Setup the class and run the version checker.

        Args:
            install_dir (str): The directory where `bin`, `lib` and `include` will be installed to
        """
        # Resolve the path
        self.install_dir = str(Path(install_dir).resolve())

        # As we currently install llvm with a different version than clang-format, we need to take
        # care that our PATH and LD_LIBRARY_PATH is resolving the paths correctly

        # Get the location of the bin and lib of the packages (this differs if one uses anaconda or pip)
        package_root = Path(site.getusersitepackages()).parents[2]
        package_bin = package_root.joinpath("bin")
        package_lib = package_root.joinpath("lib")

        # Note: The python paths must come before the rest due to clang-format being installed by pip
        self.path = f"{package_bin}:{self.install_dir}/bin"
        self.ld_library_path = f"{package_lib}:{self.install_dir}/lib"

        # Set the install_dir as a environmental variable
        os.environ["PATH"] = f"{self.path}:{os.environ['PATH']}"
        os.environ[
            "LD_LIBRARY_PATH"] = f"{self.ld_library_path}:{os.environ['PATH']}"

        self.version_checker = VersionChecker()
        self.linters_to_install = self.version_checker.linter_with_errors

    def install_uninstalled(self) -> None:
        """Install all linters marked for install.
        """
        if len(self.linters_to_install) == 0:
            print(
                "All linters appears to be installed with the correct version."
            )

        # llvm needs to be installed first due to dependencies
        self._install_uninstalled_llvm()
        self._install_uninstalled_iwyu()
        self._install_uninstalled_oclint()
        self._install_uninstalled_pip_packages()
        self._install_uninstalled_others()

        self._print_success()

    def _install_uninstalled_llvm(self) -> None:
        """Install the llvm suite.
        """
        llvm_linters = ("clang", "clang-tidy")
        llvm_dependent_linters = (*llvm_linters, "include-what-you-use",
                                  "oclint")
        if any(clang_linter in self.linters_to_install
               for clang_linter in llvm_dependent_linters):
            call_command(
                f"bash install_llvm.sh {self.install_dir} {self.version_checker.version_list['clang']}"
            )
            for llvm_linter in llvm_linters:
                if llvm_linter in self.linters_to_install:
                    self.linters_to_install.remove(llvm_linter)

    def _install_uninstalled_iwyu(self) -> None:
        """Install include-what-you-use.
        """
        if "include-what-you-use" in self.linters_to_install:
            # NOTE: The clang version needs to be passed, not the IWYU version
            call_command(
                f"bash install_iwyu.sh {self.install_dir} {self.install_dir} {self.version_checker.version_list['clang'].split('.')[0]}"
            )
            self.linters_to_install.remove("include-what-you-use")

    def _install_uninstalled_oclint(self):
        """Install oclint.
        """
        if "oclint" in self.linters_to_install:
            call_command(
                f"bash install_oclint.sh {self.install_dir} {self.install_dir} {self.version_checker.version_list['oclint']}"
            )
            self.linters_to_install.remove("oclint")

    def _install_uninstalled_pip_packages(self):
        """Install pip packages.
        """
        pip_packages = ("yapf", "clang-format", "pylint")
        for pip_package in pip_packages:
            if pip_package in self.linters_to_install:
                command = f"pip3 install {pip_package}=={self.version_checker.version_list[pip_package]}"
                call_command(command)
                self.linters_to_install.remove(pip_package)

    def _install_uninstalled_others(self):
        """Install linters with a generic install script signature.
        """
        for linter in self.linters_to_install.copy():
            call_command(
                f"bash install_{linter}.sh {self.install_dir} {self.version_checker.version_list[linter]}"
            )
            self.linters_to_install.remove(linter)

    def _print_success(self):
        """Print the success message.
        """
        print("\n\x1b[6;30;42mSuccess!\x1b[0m")
        print("Ensure that the linters are available by adding for example")
        print(f"export PATH=\"{self.path}:$PATH\"")
        print(
            f"export LD_LIBRARY_PATH=\"{self.ld_library_path}:$LD_LIBRARY_PATH\""
        )
        print("To $HOME/.bashrc or the like")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install linters")
    parser.add_argument("install_dir",
                        help="Directory to install bin/ lib/ and include/ to")
    args = parser.parse_args()
    installer = Installer(args.install_dir)
    installer.install_uninstalled()


if __name__ == "__main__":
    main()
