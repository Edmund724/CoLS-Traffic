#!/usr/bin/env python3
"""运行 NAS-FBNet 贝叶斯优化搜索"""
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nas_fbnet.search import run_search

if __name__ == "__main__":
    run_search()
