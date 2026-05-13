#!/usr/bin/env python3
import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入并运行主模块
from CodeAgentSrc.__main__ import main

if __name__ == "__main__":
    main()