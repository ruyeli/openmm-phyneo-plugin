"""Runner tests: no Context or energy calculation in this test module."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_hexane_md as md


class ArgumentTests(unittest.TestCase):
    def args(self, length='1', temperature='298', ensemble='NVT'):
        return md.parse_args([str(md.BASE/'hex_vap_430/vap_conn.pdb'), length, temperature, ensemble])

    def test_units_and_ensemble(self):
        args = self.args()
        self.assertEqual(args.steps, 1000)
        self.assertEqual(args.ensemble, 'nvt')
        self.assertEqual(md.TIMESTEP_PS, .001)

    def test_round_up(self):
        self.assertEqual(self.args('0.0011').steps, 2)

    def test_invalid_values(self):
        for length, temp in [('0', '298'), ('-1', '298'), ('nan', '298'),
                             ('inf', '298'), ('1', 'nan'), ('1', '0')]:
            with self.subTest(length=length, temperature=temp), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(length, temp)

    def test_unverified_library_rejected(self):
        # Never import OpenMM here; fail before selecting or loading a library.
        with patch('portable_runtime.load_manifest', side_effect=RuntimeError('library missing or changed')):
            with self.assertRaisesRegex(RuntimeError, 'missing or changed'):
                md.import_corrected_plugin()


class SystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mm, cls.app, cls.unit, cls.plugin, cls.np, _ = md.import_corrected_plugin()
        cls.pdb = cls.app.PDBFile(str(md.BASE/'hex_vap_430/vap_conn.pdb'))

    def test_model_and_scales(self):
        system, native = md.build_system(self.pdb, self.mm, self.app, self.unit, self.plugin)
        arrays = [self.plugin.vectord() for _ in range(3)]
        native.getMultipoleScaleFactors(*arrays)
        self.assertEqual([list(a) for a in arrays], [md.NATIVE_SCALES]*3)
        self.assertEqual(native.getMutualInducedTargetEpsilon(), md.MUTUAL_EPSILON)
        self.assertEqual(native.getMutualInducedMaxIterations(), 1000)
        self.assertEqual(system.getNumParticles(), 400)
        forces = {force.getName(): force for force in system.getForces()}
        self.assertEqual(forces['HarmonicBondForce'].getNumBonds(), 380)
        self.assertFalse(forces['CustomNonbondedForce'].getUseLongRangeCorrection())
        self.assertFalse(forces['CustomNonbondedForce'].getUseSwitchingFunction())
        # Covalent15 must retain BOTH four- and five-bond shells.
        adjacency = [set() for _ in range(400)]
        for a, b in self.pdb.topology.bonds():
            adjacency[a.index].add(b.index)
            adjacency[b.index].add(a.index)
        for i in range(400):
            seen, frontier, shells = {i}, {i}, []
            for _ in range(5):
                frontier = {j for k in frontier for j in adjacency[k]} - seen
                seen.update(frontier)
                shells.append(frontier)
            # This SWIG wrapper does not expose the named covalent enum.
            # Index 3 is Covalent15, as used by the XML reconstruction script.
            self.assertEqual(set(native.getCovalentMap(i, 3)),
                             shells[3] | shells[4])

    def test_hexane_topology_and_box(self):
        box = self.np.asarray(self.pdb.topology.getPeriodicBoxVectors().value_in_unit(self.unit.nanometer))
        self.assertEqual(md.validate_hexane(self.pdb.topology, box, self.np), 20)
        with self.assertRaisesRegex(ValueError, 'twice'):
            md.validate_hexane(self.pdb.topology, self.np.eye(3), self.np)

    def test_unexpected_model_scales_rejected(self):
        tree = md.ET.parse(md.XML)
        tree.getroot().find('ADMPPmeForce').set('pScale14', '0.5')
        with patch.object(md.ET, 'parse', return_value=tree):
            with self.assertRaisesRegex(ValueError, 'mapping blindly'):
                md.build_system(self.pdb, self.mm, self.app, self.unit, self.plugin)

    def test_missing_connectivity_rejected(self):
        topology = self.app.Topology()
        residue = topology.addResidue('BAD', topology.addChain())
        topology.addAtom('C', self.app.element.carbon, residue)
        with self.assertRaisesRegex(ValueError, 'C6H14'):
            md.validate_hexane(topology, self.np.eye(3)*4, self.np)


if __name__ == '__main__':
    unittest.main(verbosity=2)
