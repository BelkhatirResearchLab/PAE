"""Endpoint, leakage, OOF and checkpoint regression tests."""
from __future__ import annotations
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from src.common.regression_data import (
    CLINICAL_FEATURES, load_regression_data, prepare_run, finish_run,
    validate_oof, exclusive_run_lock,
)


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data.csv"
        self.schema = self.root / "schema.json"
        self.radio = ["shape_a", "texture_b"]
        self.schema.write_text(json.dumps({"clinical_features":CLINICAL_FEATURES,"radiomics_features":self.radio}),encoding="utf-8")
        self.table = pd.DataFrame({
            "patient_id":["PAE1","PAE2","PAE3","PAE4"],
            "age_at_pae":[60,61,62,63], "prostate_volume_preop_cc":[80,90,100,110],
            "abrams_griffiths_number":[30,35,40,45], "qmax_preop":[5,6,7,8],
            "pvr_preop":[40,50,60,70], "ipss_preop":[20,22,25,27],
            "ipss_3m":[8,12,15,18], "ipss_change_3m":[12,10,10,9],
            "shape_a":[1,2,3,4], "texture_b":[4,3,2,1],
        })
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        self.table.to_csv(self.data,index=False)

    def load(self, target="ipss_3m"):
        return load_regression_data(self.data,4,2,target,self.schema)

    def args(self, **changes):
        values = dict(data=self.data,feature_manifest=self.schema,output_dir=self.root/"run",target="ipss_3m",expected_patients=4,expected_radiomics_features=2,random_state=42,inner_folds=3,pca_variance=0.95,n_jobs=1,batch_size=2,max_outer_folds=2,bootstrap_resamples=20,resume=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_measured_followup_is_unchanged_and_change_is_excluded(self):
        table, radio = self.load()
        np.testing.assert_array_equal(table["ipss_3m"],self.table["ipss_3m"])
        self.assertEqual(radio,self.radio)
        self.assertFalse({"ipss_3m","ipss_change_3m"} & set(radio))
        table, radio = self.load("ipss_change_3m")
        np.testing.assert_array_equal(table["ipss_change_3m"],self.table["ipss_change_3m"])

    def test_prefixed_feature_schema_keeps_canonical_order(self):
        self.table = self.table.rename(columns={c:"radiomics__"+c for c in self.radio})
        self.table = self.table[self.table.columns[::-1]]
        self.save()
        _, radio = self.load()
        self.assertEqual(radio,["radiomics__"+c for c in self.radio])

    def test_unexpected_outcome_column_is_rejected(self):
        self.table["postop_qol"] = 1
        self.save()
        with self.assertRaisesRegex(ValueError,"Unexpected columns"):
            self.load()

    def test_bad_schema_and_missing_predictor_are_rejected(self):
        self.table = self.table.drop(columns="texture_b")
        self.table["ipss_change_3m_copy"] = 10
        self.save()
        with self.assertRaisesRegex(ValueError,"frozen feature manifest"):
            self.load()

    def test_invalid_ids_missing_values_range_and_change_are_rejected(self):
        original = self.table.copy()
        for column,value,message in [("patient_id","PAE2","duplicate"),("patient_id","","Blank"),("shape_a",np.nan,"missing"),("ipss_3m",36,"outside"),("ipss_change_3m",999,"inconsistent")]:
            with self.subTest(column=column,value=value):
                self.table = original.copy()
                self.table.loc[0,column] = value
                self.save()
                with self.assertRaisesRegex(ValueError,message):
                    self.load()

    def test_resume_rejects_target_seed_data_and_limit_changes(self):
        table, radio = self.load()
        args = self.args()
        self.assertFalse(prepare_run(args,table,radio,"08"))
        for changes in [{"target":"ipss_change_3m"},{"random_state":43},{"max_outer_folds":None}]:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError,"identity changed"):
                    prepare_run(self.args(resume=True,**changes),table,radio,"08")
        self.table.loc[0,"shape_a"] = 123
        self.save()
        table, radio = self.load()
        with self.assertRaisesRegex(ValueError,"identity changed"):
            prepare_run(self.args(resume=True),table,radio,"08")

    def test_resume_accepts_worker_change_but_checks_completed_artifacts(self):
        table, radio = self.load()
        args = self.args()
        prepare_run(args,table,radio,"08")
        self.assertFalse(prepare_run(self.args(resume=True,n_jobs=2),table,radio,"08"))
        artifact = args.output_dir / "result.csv"
        artifact.write_text("a\n1\n",encoding="utf-8")
        finish_run(args)
        self.assertTrue(prepare_run(self.args(resume=True),table,radio,"08"))
        artifact.write_text("a\n2\n",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"output changed"):
            prepare_run(self.args(resume=True),table,radio,"08")

    def test_unmanifested_legacy_checkpoint_is_refused(self):
        args = self.args(resume=True)
        args.output_dir.mkdir()
        (args.output_dir/"ablation_oof.partial.csv").write_text("old",encoding="utf-8")
        table, radio = self.load()
        with self.assertRaisesRegex(ValueError,"without a run manifest"):
            prepare_run(args,table,radio,"08")

    def test_oof_coverage_target_patient_and_duplicates(self):
        frame = pd.DataFrame({"patient_id":["PAE1","PAE2"],"outer_fold":[1,2],"configuration":["clinical","clinical"],"observed":[8.,12.],"predicted":[9.,11.]})
        validate_oof(frame,self.table,"ipss_3m",["clinical"],2)
        bad = frame.copy()
        bad.loc[0,"observed"] = 12
        with self.assertRaisesRegex(ValueError,"target mismatch"):
            validate_oof(bad,self.table,"ipss_3m",["clinical"],2)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_oof(pd.concat([frame,frame]),self.table,"ipss_3m",["clinical"],2)
        with self.assertRaisesRegex(ValueError,"Incomplete"):
            validate_oof(frame.iloc[:1],self.table,"ipss_3m",["clinical"],2)
        bad = frame.copy()
        bad.loc[0,"patient_id"] = "PAE4"
        with self.assertRaisesRegex(ValueError,"patient/fold mismatch"):
            validate_oof(bad,self.table,"ipss_3m",["clinical"],2)

    def test_concurrent_output_lock_is_refused(self):
        with exclusive_run_lock(self.root / "locked"):
            with self.assertRaisesRegex(RuntimeError,"Another process"):
                with exclusive_run_lock(self.root / "locked"):
                    self.fail("Second writer acquired the lock")


if __name__ == "__main__":
    unittest.main()
