# import splunklib.client as client
import os
import sys
from pathlib import Path

# TODO: Move this to documentation page somewhere
# To learn more about this file go to: https://docs.python-guide.org/writing/structure/
# This is a relative path to the ot/ directory
base_path = Path("../../utils")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), base_path)))
