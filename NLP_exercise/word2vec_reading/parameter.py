# -*- coding: utf-8 -*-

import os
import re

cur_path = os.path.abspath(os.path.dirname(__file__))
split_path = re.split("[/\\\]", cur_path)
working_space = split_path[:split_path.index("resource_code_reading") + 1]

WORKING_SPACE = "/".join(working_space)


if __name__ == "__main__":
    print(WORKING_SPACE)


