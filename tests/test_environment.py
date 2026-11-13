
import sys
import numpy
import pandas
import sklearn
import fastapi


def test_python_version():
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 13


def test_numpy_installed():
    assert numpy.__version__ is not None


def test_pandas_installed():
    assert pandas.__version__ is not None


def test_sklearn_installed():
    assert sklearn.__version__ is not None


def test_fastapi_installed():
    assert fastapi.__version__ is not None
