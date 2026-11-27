import tempfile
import unittest
from pathlib import Path

from cancer_lab.data import write_json
from cancer_lab.split_check import check_splits


class SplitCheckTests(unittest.TestCase):
    def build(self, root):
        return dict(root=str(root),classes=['a','b'],strategy='group',splits={
            name:[dict(path=f'{name}/{i}.png',label=i,sha256=f'{name}_{i}',group=f'{name}_patient_{i}') for i in range(2)]
            for name in ('train','val','test')})

    def test_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'manifest.json';write_json(path,self.build(temp))
            self.assertTrue(check_splits(path)['passed'])

    def test_patient_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            data=self.build(temp);data['splits']['test'][0]['group']=data['splits']['train'][0]['group']
            path=Path(temp)/'manifest.json';write_json(path,data)
            with self.assertRaisesRegex(ValueError,'overlapping group'):check_splits(path)

    def test_duplicate_content(self):
        with tempfile.TemporaryDirectory() as temp:
            data=self.build(temp);data['splits']['test'][0]['sha256']=data['splits']['val'][0]['sha256']
            path=Path(temp)/'manifest.json';write_json(path,data)
            with self.assertRaisesRegex(ValueError,'overlapping sha256'):check_splits(path)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            data=self.build(temp);data['splits']['test'][0]['path']='../outside.png'
            path=Path(temp)/'manifest.json';write_json(path,data)
            with self.assertRaisesRegex(ValueError,'outside dataset root'):check_splits(path)
