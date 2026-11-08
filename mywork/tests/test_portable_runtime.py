"""Build-provenance checks without importing OpenMM or executing calculations."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import portable_runtime as runtime


class ManifestTests(unittest.TestCase):
    def config(self):
        return dict(python_prefix=sys.prefix, source_sha256={'example': 'hash'},
                    files={}, gpu_validation=None)

    def check(self, data, validated=False):
        with patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'read_text', return_value=json.dumps(data)), \
             patch.object(runtime, 'source_hashes', return_value={'example': 'hash'}):
            return runtime.load_manifest(require_validated=validated)

    def test_unvalidated_build_does_not_enable_md(self):
        self.check(self.config())
        with self.assertRaisesRegex(RuntimeError, 'GPU validation'):
            self.check(self.config(), validated=True)

    def test_validation_enables_matching_build(self):
        data = self.config()
        data['gpu_validation'] = {'completed_utc': 'test'}
        self.check(data, validated=True)

    def test_changed_source_rejected(self):
        data = self.config()
        data['source_sha256'] = {}
        with self.assertRaisesRegex(RuntimeError, 'source/model changed'):
            self.check(data)

    def test_changed_environment_rejected(self):
        data = self.config()
        data['python_prefix'] = '/does-not-exist-phyneo-environment'
        with self.assertRaisesRegex(RuntimeError, 'another Python environment'):
            self.check(data)

    def test_changed_binary_rejected(self):
        data = self.config()
        data['files'] = {'example': {'path': '/example.so', 'sha256': 'old'}}
        with patch.object(runtime, 'sha256', return_value='new'):
            with self.assertRaisesRegex(RuntimeError, 'missing or changed'):
                self.check(data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
