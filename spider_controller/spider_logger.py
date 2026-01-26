"""spider_logger.py

轻量 CSV logger（参考你给的 wbc_logger.py 的风格）。

设计目标：
- 统一把 spider_gait_manager_qp.py 里需要记录的状态/目标写入 debug/*.csv
- 自动创建 debug/ 目录
- 首次写入时写 header，后续 append

用法示例：
    from spider_logger import SpiderCsvLogger

    logger = SpiderCsvLogger(base_dir='./debug')
    logger.write_row(
        name='com_target',
        filename='com_target_data.csv',
        header=['t','x','y','z'],
        row=[t, x, y, z],
    )

注意：
- 这里不依赖 torch；全部用 Python 标量/NumPy 标量都可以（会被 float(...) 转换）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Union
import csv


Number = Union[int, float]


@dataclass
class CsvSeries:
    path: Path
    header: List[str]
    initialized: bool = False


@dataclass
class SpiderCsvLogger:
    base_dir: Path | str = Path('./debug')
    series: Dict[str, CsvSeries] = field(default_factory=dict)

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    def _ensure(self, name: str, filename: str, header: Iterable[str]):
        if name in self.series:
            return
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.series[name] = CsvSeries(path=self.base_dir / filename, header=list(header), initialized=False)

    def write_row(self, name: str, filename: str, header: Iterable[str], row: List[Number]):
        """Append one row to a CSV series.

        - name: series key (used for one-time header init)
        - filename: CSV filename under base_dir
        - header: column names
        - row: row values (will be cast to float)
        """
        self._ensure(name, filename, header)
        s = self.series[name]

        if not s.initialized:
            s.path.parent.mkdir(parents=True, exist_ok=True)
            with s.path.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(s.header)
            s.initialized = True

        # cast values to float for consistency (NumPy scalars are ok)
        row_f = [float(x) for x in row]
        with s.path.open('a', newline='') as f:
            w = csv.writer(f)
            w.writerow(row_f)
