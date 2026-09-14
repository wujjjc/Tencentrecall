"""让 tests/ 下的用例能 `import config`（taac/ 是平铺模块，不是包）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: 需要 prepare.py 缓存的端到端测试")
