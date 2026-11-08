"""Lightweight runner/system tests; no CUDA Context or MD is created."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_hexane_md_pme as md


class ArgumentTests(unittest.TestCase):
    def test_four_inputs_and_timestep(self):
        args = md.parse_args([str(md.BASE/'hex_vap_430/vap_conn.pdb'), '.0011', '450', 'NVT'])
        self.assertEqual(args.steps, 2)
        self.assertEqual(args.ensemble, 'nvt')
        self.assertEqual(md.TIMESTEP_PS, .001)

    def test_bad_duration(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            md.parse_args([str(md.BASE/'hex_vap_430/vap_conn.pdb'), 'nan', '450', 'nvt'])

    def test_missing_library_rejected(self):
        with patch.object(md, 'LIBRARY', Path('/does-not-exist-native-dispersion')):
            with self.assertRaisesRegex(RuntimeError, 'missing or changed'):
                md.verify_installation()


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        md.verify_installation()
        cls.mm, cls.app, cls.unit, cls.plugin, cls.np, _ = md.previous.import_corrected_plugin()
        cls.pdb = cls.app.PDBFile(str(md.BASE/'hex_vap_430/vap_conn.pdb'))

    def test_correction_and_scales(self):
        system, native = md.previous.build_system(self.pdb, self.mm, self.app, self.unit, self.plugin)
        box = self.np.asarray(self.pdb.topology.getPeriodicBoxVectors().value_in_unit(self.unit.nanometer))
        before = system.getNumForces()
        info = md.add_dispersion(system, box, self.mm, self.unit, self.np)
        self.assertEqual(system.getNumForces(), before+1)
        self.assertEqual(info['grid'], [98, 98, 98])
        self.assertEqual(system.getForce(before).getName(), 'NativeDispersionPMEMinusCutoff')
        self.assertEqual(system.getForce(before).getForceGroup(), 30)
        arrays = [self.plugin.vectord() for _ in range(3)]
        native.getMultipoleScaleFactors(*arrays)
        self.assertEqual([list(a) for a in arrays], [md.previous.NATIVE_SCALES]*3)
        with self.assertRaisesRegex(ValueError, 'already present'):
            md.add_dispersion(system, box, self.mm, self.unit, self.np)


if __name__ == '__main__':
    unittest.main(verbosity=2)
