#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

import numpy as np
import pandas as pd
from ax.utils.common.equality import (
    dataframe_equals,
    datetime_equals,
    equality_typechecker,
    object_attribute_dicts_find_unequal_fields,
    same_elements,
)
from ax.utils.common.testutils import TestCase


class EqualityTest(TestCase):
    def testEqualityTypechecker(self) -> None:
        @equality_typechecker
        # pyre-fixme[2]: Parameter must be annotated.
        def eq(x, y) -> bool:
            return x == y

        self.assertFalse(eq(5, 5.0))
        self.assertTrue(eq(5, 5))

    def testListsEquals(self) -> None:
        self.assertFalse(same_elements([0], [0, 1]))
        self.assertFalse(same_elements([1, 0], [0, 2]))
        self.assertTrue(same_elements([1, 0], [0, 1]))

    def testDatetimeEquals(self) -> None:
        now = datetime.now()
        self.assertTrue(datetime_equals(None, None))
        self.assertFalse(datetime_equals(None, now))
        self.assertTrue(datetime_equals(now, now))

    def testDataframeEquals(self) -> None:

        # emtpy dfs
        self.assertTrue(dataframe_equals(pd.DataFrame(), pd.DataFrame()))
        pd1 = pd.DataFrame.from_records([{"x": 100, "y": 200}])
        self.assertFalse(dataframe_equals(pd.DataFrame(), pd1))

        # Same values, different order
        pd2 = pd.DataFrame.from_records([{"y": 200, "x": 100}])
        self.assertTrue(dataframe_equals(pd1, pd2))

        # different dtypes
        pd3 = pd.DataFrame.from_records([{"x": 100.0, "y": 200}])
        self.assertFalse(dataframe_equals(pd1, pd3))

        # Approximate equality
        pd4 = pd.DataFrame.from_records([{"x": 100 + 1e-9, "y": 200}])
        self.assertTrue(dataframe_equals(pd3, pd4, check_exact=False))
        self.assertFalse(dataframe_equals(pd3, pd4, check_exact=True))
        self.assertFalse(dataframe_equals(pd3, pd4))

    def test_numpy_equals(self) -> None:
        # Simple check.
        np_0 = {"cov": np.array([[0.1, 0.0], [0.0, 0.1]])}
        np_1 = {"cov": np.array([[0.1, 0.0], [0.0, 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1), ({}, {})
        )
        # Unequal.
        np_1 = {"cov": np.array([[0.1, 0.0], [0.1, 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1),
            ({}, {"cov": (np_0["cov"], np_1["cov"])}),
        )
        # With NaNs.
        np_1 = {"cov": np.array([[0.1, float("nan")], [float("nan"), 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1),
            ({}, {"cov": (np_0["cov"], np_1["cov"])}),
        )
        np_0 = {"cov": np.array([[0.1, float("nan")], [float("nan"), 0.1]])}
        self.assertEqual(
            object_attribute_dicts_find_unequal_fields(np_0, np_1), ({}, {})
        )
