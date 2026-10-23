#!/usr/bin/env python3
"""Run the VBench-I2V generator with the InfinityStar 8B 480p checkpoint."""

from tools.eval import generate_vbench_i2v_methods as i2v
from tools.eval.generate_vbench_480p_methods import build_480p_args


if __name__ == "__main__":
    i2v.build_args = build_480p_args
    i2v.main()
