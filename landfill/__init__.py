# Must import __version__ first to avoid errors importing this file during the build process.
# See https://github.com/pypa/setuptools/issues/1724#issuecomment-627241822
from ._version import __version__
# from .landfill import *
from . import layers
from . import model
from . import data

