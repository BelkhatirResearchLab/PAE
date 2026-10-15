"""Protect the fixed XYZ-to-ZYX mask orientation."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


def load_matching_module():
    path = Path(__file__).resolve().parents[1] / "src" / "01_match_ct_masks.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "align_mask"
    )
    namespace = {"np": np}
    exec(compile(ast.Module([function], []), str(path), "exec"), namespace)
    return namespace


class MaskAlignmentTests(unittest.TestCase):
    def test_mask_transpose_is_fixed_to_210(self) -> None:
        module = load_matching_module()
        mask_xyz = np.zeros((2, 3, 4), dtype=float)
        mask_xyz[1, 2, 3] = 1.0
        mask_zyx = module["align_mask"](mask_xyz)
        self.assertEqual(mask_zyx.shape, (4, 3, 2))
        self.assertEqual(mask_zyx[3, 2, 1], 1)
        self.assertEqual(int(mask_zyx.sum()), 1)
