# This logger is needed for running tests from the repo directory.
# The actual package logger is in delphi/delphi/__init__.py
import logging

logger = logging.getLogger(__name__)
