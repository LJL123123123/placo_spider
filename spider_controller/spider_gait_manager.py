"""spider_gait_manager.py

为了和你 WBIK 的结构一致，这里提供一个稳定的导入名：
- 继续复用当前仓库的 `gait_manager.py`（NumPy CPU gait planner）

这样 run_spider / spider_ik 里就可以：
    from spider_gait_manager import GaitCycleManager, GaitParams

后续如果你想把 gait_manager.py 再细分或换实现，这个 shim 不用改调用方。
"""

from gait_manager import GaitCycleManager, GaitParams, GaitPlan  # re-export

__all__ = ['GaitCycleManager', 'GaitParams', 'GaitPlan']
