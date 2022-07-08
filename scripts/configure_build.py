# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
"""A script that generates a cmake command based on user input.
To use, run the script with python 3 (3.8+, may work with 3.7 but not tested).
Use the up/down arrows to select options and the l/r arrows or enter to change
the value. Press Q to exit, P to print the command and R to run it.

IMPORTANT: This should be ran from the build directory.
"""
import json
from typing import List, Union
import os
import curses
import pytest

# This is the file where the last set settings are saved. change this to
# wherever you want to store the settings (make sure it's gitignored)
last_settings_file = ".configure_build_last_settings.json"

# To add more options, go to CMakeScript.create_options_dict and add more
# calls to self.add_cmake_arg

ESC = 0x1B
ENTER = 10  # ascii character codes

BLACK_ON_WHITE = 6
WHITE_BG = 10  # color pair enums
LIGHT_GREEN = 20
DARK_YELLOW = 21


class CMakeScript:
    def __init__(self) -> None:
        """
        Read the settings file, load options and prepare curses.

        Raises:
            OSError: The settings JSON file exists but wasn't JSON.
            RuntimeError: The terminal is not supported.
        """
        self.options = {}
        self.create_options_dict()

        self.state = self.get_default_state()

        if os.path.exists(last_settings_file):
            with open(last_settings_file, "r", encoding="utf-8") as f:
                try:
                    loaded_state = json.load(f)
                except json.JSONDecodeError as exception:
                    raise IOError(
                        f"{last_settings_file} could not be read. either "
                        "check for mistakes or delete the file."
                    ) from exception
                for k, v in loaded_state.items():
                    for k_, v_ in v.items():
                        if k in self.state:
                            if k_ in self.state[k]:
                                self.state[k][k_] = v_

        self.state = dict(sorted(self.state.items()))
        for i in self.state.keys():
            self.state[i] = dict(sorted(self.state[i].items()))
        self.stdscr = curses.initscr()
        self.cursor = 0
        self.scroll = 0
        self.has_colors = curses.has_colors()
        self.location = ""
        if self.has_colors:
            self.setup_colors()

    def setup_colors(self):

        curses.start_color()
        curses.use_default_colors()
        try:
            curses.init_color(curses.COLOR_RED, 1000, 0, 0)
            curses.init_color(curses.COLOR_BLUE, 0, 700, 1000)
            curses.init_color(curses.COLOR_GREEN, 0, 300, 0)
            curses.init_color(curses.COLOR_YELLOW, 1000, 1000, 0)
        except:
            print(
                "Colors couldn't be set. This doesn't do anything other than make the menu slightly less vibrant."
            )
        # The default colors are dim so this brightens them

        try:
            colors = [curses.COLOR_WHITE, curses.COLOR_RED, curses.COLOR_BLUE]
            for color in colors:
                curses.init_pair(color, color, -1)
                curses.init_pair(
                    WHITE_BG + color, color, curses.COLOR_WHITE
                )  # setup color pairs
            curses.init_pair(curses.COLOR_GREEN, curses.COLOR_GREEN, -1)
            curses.init_pair(curses.COLOR_YELLOW, curses.COLOR_YELLOW, -1)
            curses.init_pair(
                curses.COLOR_GREEN + WHITE_BG, curses.COLOR_GREEN, curses.COLOR_WHITE
            )
            curses.init_pair(
                curses.COLOR_YELLOW + WHITE_BG, curses.COLOR_YELLOW, curses.COLOR_WHITE
            )
            # setup black on white color pair
            curses.init_pair(BLACK_ON_WHITE, curses.COLOR_BLACK, curses.COLOR_WHITE)
            curses.curs_set(0)  # hide the usual cursor
        except:
            if os.environ["TERM"] == "xterm-256color":
                curses.endwin()
                raise RuntimeError(
                    "Failed to load color. Your terminal doesn't seem to support color. Try prefixing 'TERM=vt220 ' to the command you're using to run this script. if that doesn't work, you'll have to use a different terminal."
                )
            else:
                curses.endwin()
                raise RuntimeError(
                    "Failed to load color. You're probably running this in a docker container. Try prefixing 'TERM=xterm-256color ' onto the command you're using to run this script."
                )

    # Start of options declarations
    def create_options_dict(self) -> None:
        """Populate self.options with the options listed."""
        # To add an option call add_cmake_arg, add_string_arg or add_file_arg
        self.add_cmake_arg(
            "DO_PACKAGING", "", ["OFF", "ON"], 1, "Poplar_packaging", False
        )
        self.add_cmake_arg(
            "INTERNAL_RELEASE", "", ["OFF", "ON"], 1, "Graphcore_target_access", False
        )
        self.add_cmake_arg(
            "POPART_USE_STACKTRACE",
            "Enable boost stacktrace reports in error messages",
            ["OFF", "ON"],
            1,
            "PopART",
        )
        self.add_cmake_arg(
            "POPART_BUILD_TESTING", "Build the popart tests", ["OFF", "ON"], 1, "PopART"
        )
        self.add_cmake_arg(
            "POPART_LOG_DEVICE_ACCESS_IN_TESTS",
            "Write a device access log (deviceaccess.log) when running ctest",
            ["OFF", "ON"],
            0,
            "PopART",
        )
        self.add_cmake_arg(
            "POPART_STRICT_COMPARATOR_CHECKS",
            "Check for nullptr and invalid pointers when comparing"
            " containers of pointers",
            ["OFF", "ON"],
            0,
            "PopART",
        )
        self.add_cmake_arg(
            "POPART_ENABLE_COVERAGE",
            "Enable compiler flags which generate code coverage files.",
            ["OFF", "ON"],
            0,
            "PopART",
        )
        self.add_cmake_arg(
            "UPLOAD_COVERAGE_REPORT",
            "Add a ctest which uploads unit test coverage to elasticsearch",
            ["OFF", "ON"],
            0,
            "PopART",
        )
        self.add_cmake_arg(
            "BUILD_DOCS", "Build the PopART documentation", ["OFF", "ON"], 0, "PopART"
        )
        self.add_cmake_arg(
            "ENABLED_TEST_VARIANTS",
            "Which tests to build",
            [
                "Cpu$<SEMICOLON>IpuModel$<SEMICOLON>Hw",
                "Cpu$<SEMICOLON>IpuModel",
                "Cpu$<SEMICOLON>Hw",
                "IpuModel$<SEMICOLON>Hw",
                "Cpu",
                "Hw",
                "IpuModel",
                "",
            ],
            0,
            "PopART",
        )
        self.add_cmake_arg(
            "CMAKE_BUILD_TYPE",
            "Changes how PopART is built (Release/Debug)",
            ["Release", "Debug"],
            0,
            "PopART",
        )
        self.add_cmake_arg(
            "CMAKE_BUILD_TYPE",
            "Changes how the rest of the view is built (Release/Debug)",
            ["Release", "Debug"],
            0,
            "root",
        )
        self.add_cmake_arg(
            "USE_LOGS", "Record build results in logfiles", ["OFF", "ON"], 1, "root"
        )
        self.add_cmake_arg(
            "BUILD_DOCS", "Build Documentation", ["OFF", "ON"], 0, "DOCS_POPLAR"
        )
        self.add_file_arg(
            "SERVER_COOKIE_FILE",
            "Absolute path to file containing authorisation cookie for the elasticsearch server.",
            "",
            "PopART",
        )
        self.add_string_arg("SWDB_PACKAGE_NAME", "SWDB Package Name", "", "PopART")

        # End of options declarations
        self.options = dict(sorted(self.options.items()))
        for i in self.options.keys():
            self.options[i] = dict(sorted(self.options[i].items()))

    def get_default_state(self) -> dict:
        """Return a state dictionary reset to the default.

        Returns:
            dict: The default state
        """
        return {
            k: {k_: v_["default"] for k_, v_ in v.items()}
            for k, v in self.options.items()
        }

    def add_cmake_arg(
        self,
        name: str,
        desc: str,
        allowed_states: List[str],
        default: int,
        location: str = "root",
        omit_if_default: bool = True,
    ) -> None:
        """Register cmake arguments and add them to the class's options dictionary.

        Args:
            name (str): The name of the option
            desc (str): A description that shows up when the option is selected
            allowed_states (List[str]): A list of allowed option values
            default (int): The index of the default value in allowed_states
            location (str): The location of the option (omit for root)
            omit_if_default (bool): Whether to omit the option if default is selected
              Alows for the final command to be more concise when True, but if defualt compiler
              setting is unknown, or doesn't match the default given, it should be False.
        """

        if location not in self.options:
            self.options[location] = {}
        self.options[location][name] = {
            "desc": desc,
            "states": allowed_states,
            "default": default,
            "use_red_green": allowed_states == ["OFF", "ON"],
            "omit": omit_if_default,
            "type": "multi",
        }

    def add_string_arg(
        self,
        name: str,
        desc: str,
        default: str,
        location: str = "root",
        omit_if_default: bool = True,
    ) -> None:
        """
        Register a cmake argument and adds it to the class' options dictionary.

        The user can type any string into it.

        Args:
            name (str): The name of the option
            desc (str): A description of the option
            default (str): The default value (can be none)
            location (str, optional): The location of the option. Defaults to "root".
            omit_if_default (bool, optional): Whether to omit the option when default is selected.
                Alows for the final command to be more concise when True, but if defualt compiler
                setting is unknown, or doesn't match the default given, it should be False. Defaults to True.
        """
        if location not in self.options:
            self.options[location] = {}
        self.options[location][name] = {
            "desc": desc,
            "default": default,
            "omit": omit_if_default,
            "type": "str",
        }

    def add_file_arg(
        self,
        name: str,
        desc: str,
        default: str,
        location: str = "root",
        allow_file: bool = True,
        allow_folder: bool = False,
        omit_if_default: bool = True,
    ) -> None:
        """
        Register a cmake argument and adds it to the class' options dictionary.

        The user select any file.

        Args:
            name (str): The name of the option
            desc (str): A description of the option
            default (str): The default value (can be none)
            location (str, optional): The location of the option. Defaults to "root".
            allow_file (bool, optional): Whether to allow selecting a file. Defaults to True.
            allow_folder (bool, optional): Whether to allow selecting a folder. Defaults to False.
            omit_if_default (bool, optional): Whether to omit the option when default is selected.
              Alows for the final command to be more concise when True, but if defualt compiler
              setting is unknown, or doesn't match the default given, it should be False.  Defaults to True.
        """
        if location not in self.options:
            self.options[location] = {}
        self.options[location][name] = {
            "desc": desc,
            "default": default,
            "omit": omit_if_default,
            "type": "file",
            "allow_file": allow_file,
            "allow_folder": allow_folder,
        }

    def get_chosen_value(self, name: str, location: str = "root") -> str:
        """Get the value the user has chosen for a given option.

        Args:
            name (str): The name of the option to get
            location (str): The location of the option (omit for root)

        Returns:
            str: The value the user has chosen
        """
        if self.options[location][name]["type"] in ["str", "file"]:
            return self.state[location][name]
        return self.options[location][name]["states"][self.state[location][name]]

    def draw_chosen_value(
        self, name: str, states: dict, height: int, i: int, cols: int
    ) -> None:
        """Draw the chosen value at column 50.

        Args:
            name (str): The name of the optino
            states (dict): The data about the option
            height (int): The height on the terminal to draw it
            i (int): the index of the current list
            cols (int): The total number of columns in the terminal
        """
        selected = self.get_chosen_value(name, self.loc())
        if states["type"] in ["str", "file"]:
            is_default_option = states["default"] == selected
        else:
            is_default_option = states["states"].index(selected) == states["default"]
        if len(selected) + 50 > cols:
            if states["type"] == "file":
                selected = "..." + selected[-(cols - 53) :]
            else:
                selected = selected[: cols - 53] + "..."
        if states["type"] == "multi" and states["use_red_green"]:
            if selected == "ON":
                color = curses.COLOR_GREEN
            else:
                color = curses.COLOR_RED
        else:
            if is_default_option:
                color = curses.COLOR_BLUE  # blue for default option
            else:
                color = curses.COLOR_YELLOW  # Yellow for not default
        if i == self.cursor:
            color += WHITE_BG
            # Makes it have a white background if it is on the cursor's row
        if self.has_colors:
            self.stdscr.addstr(height, 50, selected, curses.color_pair(color))
        else:
            if i == self.cursor:
                self.stdscr.addstr(
                    height,
                    50,
                    selected,
                    curses.color_pair(curses.COLOR_WHITE) + curses.A_REVERSE,
                )
            else:
                self.stdscr.addstr(
                    height, 50, selected, curses.color_pair(curses.COLOR_WHITE)
                )

    def draw_name_desc_cursor(
        self, name: str, i: int, height: int, rows: int, cols: int, states: dict
    ) -> None:
        """Draws the name, description and cursor.

        Args:
            name (str): The name of the option
            i (int): The index of the current list
            height (int): The height on the terminal to draw.
            rows (int): The number of rows in the terminal
            cols (int): The number of columns in the terminal
            states (dict): The data about the option
        """
        if i == self.cursor:  # if this is where the cursor is
            color = BLACK_ON_WHITE  # draw the text black with a white bg
        else:
            color = curses.COLOR_WHITE
        if self.has_colors:
            self.stdscr.addstr(height, 0, name, curses.color_pair(color))
        else:
            if i == self.cursor:
                self.stdscr.addstr(
                    height,
                    0,
                    name,
                    curses.color_pair(curses.COLOR_WHITE) + curses.A_REVERSE,
                )
            else:
                self.stdscr.addstr(
                    height, 0, name, curses.color_pair(curses.COLOR_WHITE)
                )
        if color == BLACK_ON_WHITE:
            self.stdscr.addstr(
                height,
                len(name),
                "\u2588" * (cols - len(name)),
                curses.color_pair(curses.COLOR_WHITE),
            )
            self.stdscr.addstr(
                rows - 2, 0, states["desc"], curses.color_pair(curses.COLOR_WHITE)
            )

    def draw_suboptions(self, rows: int, cols: int) -> int:
        """Draw the sub-options to the screen.

        Args:
            rows (int): The number of rows in the terminal
            cols (int): The number of cols in the terminal

        Returns:
            int: How many rows the suboptions took up
        """
        offset = 0
        if self.loc() == "root":
            for i in self.options.keys():
                height = offset - self.scroll
                if height < 0 or height >= rows - 3:
                    if i != "root":
                        offset += 1
                    continue
                if i != "root":
                    self.draw_name_desc_cursor(
                        f"{i.replace('_', ' ').lower()} options",
                        offset,
                        height,
                        rows,
                        cols,
                        {"desc": f"Options for {i.lower()}"},
                    )
                    modified = False
                    for option in self.options[i].keys():
                        if self.state[i][option] != self.options[i][option]["default"]:
                            modified = True
                            break
                    effects = 0
                    if modified:
                        color = curses.COLOR_YELLOW
                        if offset == self.cursor:
                            color += WHITE_BG
                        if not self.has_colors:
                            color = curses.COLOR_WHITE
                            if offset == self.cursor:
                                effects = curses.A_REVERSE
                        self.stdscr.addstr(
                            height, 50, "Modified", curses.color_pair(color) + effects
                        )
                    offset += 1
        return offset

    def render(self) -> None:
        """Render to the terminal."""
        rows, cols = self.stdscr.getmaxyx()
        self.stdscr.erase()  # Clear screen
        offset = self.draw_suboptions(rows, cols)
        to_iterate = self.options[self.loc()].items()
        for i, option in enumerate(to_iterate):
            i += offset
            height = i - self.scroll
            if height < 0 or height >= rows - 3:
                continue
            name, states = option
            self.draw_name_desc_cursor(name, i, height, rows, cols, states)
            self.draw_chosen_value(name, states, height, i, cols)

        if self.loc() == "root":
            self.stdscr.addstr(
                rows - 1,
                10,
                "[p]rint command [r]un cmake [q]uit [d]default",
                curses.color_pair(curses.COLOR_WHITE) + curses.A_BOLD,
            )
        else:
            self.stdscr.addstr(
                rows - 1,
                0,
                "[esc]back [p]rint command [r]un cmake [q]uit [d]default",
                curses.color_pair(curses.COLOR_WHITE) + curses.A_BOLD,
            )

        # prints the bottom row

    def handle_scroll(self, rows: int) -> None:
        """Handle the scrolling and the cursor.

        Args:
            rows (int): The number of rows in the terminal
        """
        max_scroll = self.num_options + 3 - rows

        self.cursor = max(0, min(self.cursor, self.num_options - 1))
        self.scroll = max(self.cursor + 5 - rows, min(self.scroll, self.cursor))
        self.scroll = max(0, min(self.scroll, max_scroll))

    def loc(self) -> str:
        """Return the location.

        Returns:
            str: The location that the user is.
        """
        loc = self.location
        if loc == "":
            loc = "root"
        return loc

    def handle_keypress(self, key: int, rows: int) -> Union[str, None]:
        """Handle keypress logic.

        Args:
            key (int): The key that was pressed (as a number)
            rows (int): The number of rows in the terminal

        Returns:
            Union[str, None]: Retuns one character if the keypress was to close the GUI. Returns None if GUI should not close.
        """
        if key == ESC:
            self.location = ""
            self.scroll = 0
            self.cursor = 0
            return

        if key == curses.KEY_UP:
            self.cursor -= 1
            self.cursor = max(self.cursor, 0)
            if self.cursor - self.scroll < 1:
                # scrolls the screen, keeping the cursor one from the bottom
                self.scroll = max(self.cursor - 1, 0)

        if key == curses.KEY_DOWN:
            self.cursor += 1
            if self.cursor > self.num_options - 1:
                self.cursor = self.num_options - 1
            if rows + self.scroll - self.cursor < 5:
                # it is 5 because it is 5 rows from the bottom of the terminal
                self.scroll = min(self.cursor - rows + 4, self.num_options + 3 - rows)

        if key == ENTER or key == curses.KEY_RIGHT or chr(key) == " ":
            if self.loc() == "root":
                subdirs = list(self.state.keys())
                subdirs.remove("root")
                if self.cursor < len(subdirs):
                    self.location = subdirs[self.cursor]
                    return
                else:
                    name = list(self.state[self.loc()].keys())[
                        self.cursor - len(subdirs)
                    ]
            else:
                name = list(self.state[self.loc()].keys())[self.cursor]
            if self.options[self.loc()][name]["type"] == "str":
                curses.endwin()
                print(
                    f"Set value for {self.loc()}/{name}\nPrevious value: {self.state[self.loc()][name]}"
                )
                self.state[self.loc()][name] = input("Enter new value: ")
            elif self.options[self.loc()][name]["type"] == "file":
                file_select = FileSelectMenu(
                    self.stdscr,
                    allow_file=self.options[self.loc()][name]["allow_file"],
                    allow_folder=self.options[self.loc()][name]["allow_folder"],
                )
                result = file_select.main_loop()
                if result != 0:
                    self.state[self.loc()][name] = result
            else:

                self.state[self.loc()][name] += 1
                if (
                    self.state[self.loc()][name]
                    > len(self.options[self.loc()][name]["states"]) - 1
                ):
                    self.state[self.loc()][name] = 0

        if key == curses.KEY_LEFT:
            if self.loc() == "root":
                subdirs = list(self.state.keys())
                subdirs.remove("root")
                if self.cursor < len(subdirs):
                    return
                else:
                    name = list(self.state[self.loc()].keys())[
                        self.cursor - len(subdirs)
                    ]
            else:
                name = list(self.state[self.loc()].keys())[self.cursor]
            if self.options[self.loc()][name]["type"] == "multi":
                self.state[self.loc()][name] -= 1
                if self.state[self.loc()][name] < 0:
                    self.state[self.loc()][name] = (
                        len(self.options[self.loc()][name]["states"]) - 1
                    )

        if chr(key) == "d":
            if self.loc() == "root":
                self.state = self.get_default_state()
            else:
                self.state[self.loc()] = self.get_default_state()[self.loc()]
        if chr(key) in ["p", "q", "r"]:
            return chr(key)  # if p, r or q pressed return p, r or q

    def main_loop(self) -> str:
        """Loop through the program.

        It calls the class's other functions in order as well as some curses functions.

        Raises:
            OSError: The terminal is too short. This is caught and displayed to the user

        Returns:
            str: The key that caused the function to return
        """

        while True:
            rows, _ = self.stdscr.getmaxyx()
            if rows < 6:
                raise IOError
            self.num_options = len(self.options[self.loc()].keys())
            if self.loc() == "root":
                self.num_options += len(self.options.keys()) - 1
            self.handle_scroll(rows)
            self.render()  # render the screen
            self.stdscr.refresh()  # refresh the terminal
            key = self.stdscr.getch()  # wait for keypress
            result = self.handle_keypress(key, rows)
            if result is not None:
                return result

    def generate_command(self) -> str:
        """Generate the cmake command.

        Returns:
            str: The cmake command
        """
        build_args = {"root": []}

        for loc in self.state.keys():
            for name in self.state[loc].keys():

                chosen = self.get_chosen_value(name, loc)
                # If the value was the default, it is ignored to cut down on size, unless omit is false
                if (
                    self.state[loc][name] != self.options[loc][name]["default"]
                    or not self.options[loc][name]["omit"]
                ):
                    if loc not in build_args:
                        build_args[loc] = []
                    if self.options[loc][name]["type"] == "str":
                        build_args[loc].append(f"-D{name}='{chosen}'")
                    else:
                        build_args[loc].append(f"-D{name}={chosen}")

        main_args = build_args["root"]
        del build_args["root"]

        build_args_strings = [
            f"-D{k.upper()}_CMAKE_ARGS=\"{';'.join(v)}\""
            for k, v in list(build_args.items())
        ]

        main_args += build_args_strings

        main_arg_string = " ".join(main_args)
        return f"cmake {main_arg_string} -GNinja ."


class FileSelectMenu:
    def __init__(self, stdscr, allow_file=True, allow_folder=False):
        """Initialize the file select menu.

        Args:
            stdscr (_type_): The screen handle
            allow_file (bool, optional): Whether to allow file selection. Defaults to True.
            allow_folder (bool, optional): Whether to allow folder selection. Defaults to False.
        """
        self.path = os.getcwd()
        self.stdscr = stdscr
        self.scroll = 0
        self.cursor = 0
        self.allow_file = allow_file
        self.allow_folder = allow_folder

    def render(self, rows: int, cols: int):
        """Render the file select menu.

        Args:
            rows (int): The number of rows in the terminal
            cols (int): The number of columns in the terminal
        """
        self.stdscr.erase()
        self.stdscr.addstr(0, 0, self.path, curses.COLOR_WHITE + curses.A_UNDERLINE)
        for i, v in enumerate(self.dirs + self.files):
            if i - self.scroll < 0 or i - self.scroll > rows - 4:
                continue
            effect = 0
            if i == self.cursor:
                effect = curses.A_REVERSE
                self.stdscr.addstr(
                    i + 1 - self.scroll, 0, "\u2588" * cols, curses.COLOR_WHITE
                )
            self.stdscr.addstr(i + 1 - self.scroll, 0, v, curses.COLOR_WHITE + effect)
        self.stdscr.addstr(
            rows - 1,
            0,
            "[q]cancel [arrows]navigate [enter]select",
            curses.COLOR_WHITE + curses.A_BOLD,
        )

    def handle_keypress(self, key: int, rows: int) -> Union[str, int, None]:
        """Handle the keypress logic.

        Args:
            key (int): The key that was pressed as an integer
            rows (int): The number of rows in the terminal

        Returns:
            Union[str, int, None]: return a path as a str, 0 to cancel file selection and None to carry on as normal.
        """
        if key == curses.KEY_UP:
            self.cursor -= 1
            self.cursor = max(self.cursor, 0)
            if self.cursor - self.scroll < 1:
                # scrolls the screen, keeping the cursor one from the bottom
                self.scroll = max(self.cursor - 1, 0)

        if key == curses.KEY_DOWN:
            self.cursor += 1
            if self.cursor > self.num_options - 1:
                self.cursor = self.num_options - 1
            if rows + self.scroll - self.cursor < 6:
                # it is 5 because it is 5 rows from the bottom of the terminal
                self.scroll = min(self.cursor - rows + 5, self.num_options + 4 - rows)

        if key == curses.KEY_RIGHT:
            items = self.dirs + self.files
            if items[self.cursor] in self.dirs:
                self.path = os.path.join(self.path, items[self.cursor])
                self.scroll = 0
                self.cursor = 0

        if key == curses.KEY_LEFT:
            self.path = os.path.abspath(os.path.join(self.path, ".."))
            self.scroll = 0
            self.cursor = 0

        if key == ENTER:
            items = self.dirs + self.files
            path = os.path.join(self.path, items[self.cursor])
            if os.path.isdir(path) and not self.allow_folder:
                return
            return path

        if chr(key) == "q":
            return 0

    def handle_scroll(self, rows: int):
        """Bound the scroll and cursor so they don't do anything weird.

        Args:
            rows (int): The number of rows in the terminal
        """
        max_scroll = self.num_options + 3 - rows

        self.cursor = max(0, min(self.cursor, self.num_options - 1))
        self.scroll = max(self.cursor + 5 - rows, min(self.scroll, self.cursor))
        self.scroll = max(0, min(self.scroll, max_scroll))

    def main_loop(self) -> Union[str, int]:
        """Handle the file selection.

        Returns:
            Union[str, int]: Returns a path as str or 0 to cancel
        """
        while True:
            rows, cols = self.stdscr.getmaxyx()
            present = os.listdir(self.path)
            self.dirs = sorted(
                [i for i in present if os.path.isdir(os.path.join(self.path, i))],
                key=str.lower,
            )
            if self.allow_file:
                self.files = sorted(
                    [i for i in present if os.path.isfile(os.path.join(self.path, i))],
                    key=str.lower,
                )
            else:
                self.files = []
            self.num_options = len(self.dirs) + len(self.files)
            self.render(rows, cols)
            self.stdscr.refresh()
            key = self.stdscr.getch()
            result = self.handle_keypress(key, rows)
            if result is not None:
                return result
            self.handle_scroll(rows)


@pytest.mark.parametrize(
    "statechanges,result",
    [
        (
            {},
            'cmake -DGRAPHCORE_TARGET_ACCESS_CMAKE_ARGS="-DINTERNAL_RELEASE=ON" -DPOPLAR_PACKAGING_CMAKE_ARGS="-DDO_PACKAGING=ON" -GNinja .',
        ),
        (
            {"BUILD_DOCS": 1},
            'cmake -DGRAPHCORE_TARGET_ACCESS_CMAKE_ARGS="-DINTERNAL_RELEASE=ON" -DPOPART_CMAKE_ARGS="-DBUILD_DOCS=ON" -DPOPLAR_PACKAGING_CMAKE_ARGS="-DDO_PACKAGING=ON" -GNinja .',
        ),
        (
            {"POPART_BUILD_TESTING": 0},
            'cmake -DGRAPHCORE_TARGET_ACCESS_CMAKE_ARGS="-DINTERNAL_RELEASE=ON" -DPOPART_CMAKE_ARGS="-DPOPART_BUILD_TESTING=OFF" -DPOPLAR_PACKAGING_CMAKE_ARGS="-DDO_PACKAGING=ON" -GNinja .',
        ),
        (
            {"CMAKE_BUILD_TYPE": 1},
            'cmake -DGRAPHCORE_TARGET_ACCESS_CMAKE_ARGS="-DINTERNAL_RELEASE=ON" -DPOPART_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Debug" -DPOPLAR_PACKAGING_CMAKE_ARGS="-DDO_PACKAGING=ON" -GNinja .',
        ),
    ],
)
def test_generate_command(statechanges: dict, result: str):
    """
    Test the generate command.

    This function throws an error if the command generated from a state
    is not as expected

    Args:
        statechanges (dict): The changes to the state to test
        result (str): The expected command
    """

    class FakeScript:  # A class that acts like the main class but without
        # the parts that render to screen.
        options = {}
        state = {}
        add_cmake_arg = CMakeScript.add_cmake_arg
        get_chosen_value = CMakeScript.get_chosen_value

    my_fake_script = FakeScript()
    CMakeScript.create_options_dict(my_fake_script)
    defaultstate = CMakeScript.get_default_state(my_fake_script)
    my_fake_script.state = defaultstate
    my_fake_script.state["PopART"] = {**defaultstate["PopART"], **statechanges}

    assert CMakeScript.generate_command(my_fake_script) == result


if __name__ == "__main__":

    main_script = CMakeScript()

    try:
        exit_state = curses.wrapper(lambda x: main_script.main_loop())
        if exit_state == "p" or exit_state == "r":
            # generate the command now.
            command = main_script.generate_command()
            print(command)
            if exit_state == "r":
                os.system(command)

    except curses.error:
        print("Terminal too narrow. 57 chars needed. width shown below.")
        print("<" + "=" * 55 + ">")  # shows user minimum space needed
    except IOError:
        print("Terminal too short. At least 6 rows are required")
    with open(last_settings_file, "w+", encoding="utf-8") as f:
        json.dump(main_script.state, f)
