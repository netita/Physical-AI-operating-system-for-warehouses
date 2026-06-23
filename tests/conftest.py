"""
tests/conftest.py
==================
Shared pytest configuration and fixtures for the WarehouseGPT test suite.

asyncio_mode is set to "auto" in pyproject.toml so all async test functions
are collected and run without explicit @pytest.mark.asyncio decorators.

Namespace bootstrap
--------------------
The source packages (safety_ai, world_model, warehouse_agent) live directly
at the project root, not inside a warehousegpt/ directory.  All source code
uses warehousegpt.* absolute imports, so we register a namespace package
whose __path__ points at the project root.  Python's importer then resolves
warehousegpt.safety_ai  →  <project_root>/safety_ai/
warehousegpt.world_model →  <project_root>/world_model/
etc., without any code changes needed in the source packages.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent.parent)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if "warehousegpt" not in sys.modules:
    _wg = types.ModuleType("warehousegpt")
    _wg.__path__ = [_PROJECT_ROOT]   # importer looks here for sub-packages
    _wg.__package__ = "warehousegpt"
    sys.modules["warehousegpt"] = _wg
