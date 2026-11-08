"""Static full damped-dispersion comparison using original DMFF generators.

DMFF CPU and OpenMM run in separate processes/environments. Production files
are read only. No multipole/polarization, bonded, or ML energy is included.
"""
import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
CASES = {'vapor': 'hex_vap_430/vap_conn.pdb',
         'liquid': 'hex_liq_298/liq_conn.pdb',
         'slab': 'hex_slab_450_2/phyneo_slab_450_whole.pdb'}
SR = ['SlaterExForce', 'SlaterSrEsForce', 'SlaterSrPolForce',
      'SlaterSrDispForce', 'SlaterDhfForce', 'QqTtDampingForce', 'SlaterDampingForce']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_case(case):
    from prepare_input import parse_pdb
    fixture = HERE/'results'/f'{case}_input.npz'
    data = dict(np.load(fixture))
    pdb = WORK/CASES[case]
    atoms, _, _, _ = parse_pdb(pdb.read_text())
    original = ET.parse(WORK/'npt_ts_1fs/output.xml').getroot()
    names = {(r.get('name'), a.get('name')): a.get('type')
             for r in original.find('Residues') for a in r.findall('Atom')}
    types = [names[(a.line[17:20].strip(), a.line[12:16].strip())] for a in atoms.values()]
    positions = np.array([[float(a.line[k:k+8])*.1 for k in (30, 38, 46)] for a in atoms.values()])
    np.testing.assert_array_equal(positions, data['positions'])
    coefficients = {a.get('type'): [float(a.get(f'C{n}')) for n in (6, 8, 10)]
                    for a in original.find('ADMPDispPmeForce')}
    np.testing.assert_array_equal([coefficients[t] for t in types], data['coefficients'])
    for name in ['ADMPDispPmeForce', *SR]:
        np.testing.assert_array_equal([float(original.find(name).get(f'mScale{k}'))
                                       for k in range(12, 17)], data['scales'][:5])
    hashes = {str(p.relative_to(WORK)): digest(p) for p in
              [fixture, pdb, WORK/'npt_ts_1fs/output.xml', WORK/'phyneo_hexane_opls_md.xml']}
    return data, pdb, original, types, hashes


def sums(values, names):
    return (sum(values[n][0] for n in names), sum(values[n][1] for n in names))


def dmff_worker(data, original, types, output, report, diagnose_precision=False):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import jax
    import jax.numpy as jnp
    from openmm import app, unit
    from dmff.api.paramset import ParamSet
    generators = importlib.import_module('dmff.generators.admp')
    pairwise = importlib.import_module('dmff.admp.pairwise')
    import dmff
    jax.config.update('jax_enable_x64', True)

    class TypedTopology:
        # Only the generator's typing/box interface is needed. Connectivity is
        # already represented by the independently prepared bonded pair depths.
        def __init__(self):
            self._meta = {}
        def atoms(self):
            return iter(SimpleNamespace(meta={'type': t}) for t in types)
        def getNumAtoms(self):
            return len(types)
        def getPeriodicBoxVectors(self):
            return data['box']

    ffinfo = {'Forces': {n: {'meta': dict(original.find(n).attrib),
                             'node': [{'name': a.tag, 'attrib': dict(a.attrib)} for a in original.find(n)]}
                         for n in ['ADMPDispPmeForce', *SR]}}
    class_names = ['SlaterExGenerator', 'SlaterSrEsGenerator', 'SlaterSrPolGenerator',
                   'SlaterSrDispGenerator', 'SlaterDhfGenerator', 'QqTtDampingGenerator',
                   'SlaterDampingGenerator']
    params = ParamSet()
    top = TypedTopology()
    functions = {}
    gen = generators.ADMPDispPmeGenerator(ffinfo, params)
    functions['raw_pme'] = gen.createPotential(top, app.PME, float(data['cutoff'])*unit.nanometer, ethresh=1e-4)
    report['grid'] = [gen.disp_force.K1, gen.disp_force.K2, gen.disp_force.K3]
    report['alpha_per_nm'] = float(gen.disp_force.kappa)*10
    cut = generators.ADMPDispPmeGenerator(ffinfo, params)
    functions['raw_cutoff'] = cut.createPotential(top, app.CutoffPeriodic, float(data['cutoff'])*unit.nanometer)
    for name, cls in zip(SR, class_names):
        generator = getattr(generators, cls)(ffinfo, params)
        functions[name] = generator.createPotential(top, app.PME, float(data['cutoff'])*unit.nanometer)

    # This local DMFF pairwise.py uses column 3, while disp_pme.py uses column 2.
    # Duplicate the SAME bond depth in both columns; do not rely on JAX's
    # out-of-bounds clipping for a three-column array. No upstream code changes.
    pairs = jnp.asarray(np.column_stack((data['pairs'], data['pairs'][:, 2])))
    x, box = jnp.asarray(data['positions']), jnp.asarray(data['box'])
    values = {}
    timings = {}
    for name, function in functions.items():
        start = time.perf_counter()
        value, grad = jax.jit(jax.value_and_grad(function))(x, box, pairs, params.parameters)
        values[name] = float(value), -np.asarray(grad)
        timings[name] = time.perf_counter()-start
        print(name, values[name][0], flush=True)
    values['damped_cutoff'] = sums(values, ['raw_cutoff', 'SlaterDampingForce'])
    values['damped_pme'] = sums(values, ['raw_pme', 'SlaterDampingForce'])
    values['dispersion_full'] = sums(values, ['raw_pme', 'SlaterDampingForce', 'SlaterSrDispForce'])
    values['combined_short_pme'] = sums(values, ['raw_pme', *SR])

    def direct_full(pos, cell, pairs_, params_):
        return sum(functions[n](pos, cell, pairs_, params_) for n in
                   ['raw_pme', 'SlaterDampingForce', 'SlaterSrDispForce'])
    e, g = jax.jit(jax.value_and_grad(direct_full))(x, box, pairs, params.parameters)
    report['direct_vs_component_sum'] = dict(energy=float(e)-values['dispersion_full'][0],
        force_max=float(np.max(np.abs(-np.asarray(g)-values['dispersion_full'][1]))))
    values['direct_dispersion_full'] = float(e), -np.asarray(g)
    if diagnose_precision:
        # CustomNonbondedForce stores the XML per-particle parameters as float32
        # on CUDA, even in double mode. The native PME correction keeps double
        # coefficients. Round ONLY the existing short-range/cutoff parameters.
        rounded_params = jax.tree_util.tree_map(
            lambda a: jnp.asarray(np.asarray(a).astype(np.float32).astype(np.float64)), params.parameters)
        rounded = {}
        for name in ['raw_cutoff', *SR]:
            e, g = jax.jit(jax.value_and_grad(functions[name]))(x, box, pairs, rounded_params)
            rounded[name] = float(e), -np.asarray(g)
        delta = (values['raw_pme'][0]-values['raw_cutoff'][0],
                 values['raw_pme'][1]-values['raw_cutoff'][1])
        rounded['delta'] = delta
        for name, terms in {
            'damped_cutoff': ['raw_cutoff', 'SlaterDampingForce'],
            'SlaterSrDispForce': ['SlaterSrDispForce'],
            'damped_pme': ['raw_cutoff', 'SlaterDampingForce', 'delta'],
            'dispersion_full': ['raw_cutoff', 'SlaterDampingForce', 'SlaterSrDispForce', 'delta'],
            'combined_short_pme': ['raw_cutoff', *SR, 'delta']}.items():
            values['rounded_'+name] = sums(rounded, terms)
        report['rounding_diagnostic'] = 'XML short-range parameters rounded float32 then evaluated float64; native correction remains unrounded'
    module = Path(dmff.__file__).resolve().parent
    report.update(devices=[str(d) for d in jax.devices()], jax=jax.__version__,
                  dmff=str(module), timings_s=timings,
                  pair_columns='i,j,bond_depth,bond_depth; same scale for both kernels',
                  source_sha256={n: digest(module/n) for n in ['admp/disp_pme.py', 'admp/recip.py',
                                 'admp/pairwise.py', 'admp/pme.py', 'generators/admp.py']})
    old = np.load(HERE/'results'/f"{report['case']}_dmff.npz")
    report['raw_pme_vs_previous'] = dict(energy=values['raw_pme'][0]-float(old['pme_energy']),
        force_max=float(np.max(np.abs(values['raw_pme'][1]-old['pme_forces']))))
    report['energies'] = {n: v[0] for n, v in values.items()}
    assert all(np.isfinite(e) and np.all(np.isfinite(f)) for e, f in values.values())
    arrays = {f'{n}_{kind}': value for n, v in values.items() for kind, value in zip(['energy', 'forces'], v)}
    with (output/'reference.npz').open('xb') as handle:
        np.savez(handle, **arrays)
    report['status'] = 'reference_complete'


def openmm_worker(data, pdb, output, report, reference, platform):
    import openmm as mm
    from openmm import app, unit
    from native import add_correction
    ref_report = json.loads((reference/'report.json').read_text())
    if ref_report['input_sha256'] != report['input_sha256']:
        raise ValueError('Reference/input provenance mismatch')
    ref = np.load(reference/'reference.npz')
    root = ET.parse(WORK/'phyneo_hexane_opls_md.xml').getroot()
    # Use OpenMM's real XML parser and bond exclusions, retaining the production
    # CustomNonbondedForce intact. Omit unrelated forces and reconstruction Script.
    for node in list(root):
        if node.tag not in ('AtomTypes', 'Residues', 'CustomNonbondedForce'):
            root.remove(node)
    mol = app.PDBFile(str(pdb))
    np.testing.assert_allclose(mol.positions.value_in_unit(unit.nanometer), data['positions'], atol=1e-15, rtol=0)
    system = app.ForceField(io.StringIO(ET.tostring(root, encoding='unicode'))).createSystem(
        mol.topology, nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=float(data['cutoff'])*unit.nanometer, removeCMMotion=False, rigidWater=False)
    assert system.getNumForces() == 1
    combined = system.getForce(0)
    assert isinstance(combined, mm.CustomNonbondedForce)
    assert not combined.getUseLongRangeCorrection() and not combined.getUseSwitchingFunction()
    assert {tuple(sorted(combined.getExclusionParticles(i))) for i in range(combined.getNumExclusions())} == {
        tuple(pair) for pair in data['exclusions']}
    combined.setForceGroup(0)
    expression = combined.getEnergyFunction()
    energy, *definitions = expression.split(';')
    assert energy.count('-sqrt(C61*C62)') == 1
    damped_expression = energy[energy.index('-sqrt(C61*C62)'):]
    damped_expression += ';'+';'.join(definitions)
    sr_expression = '-Adisp1*Adisp2*(1+Br+Br^2/3)*exp(-Br);Br=sqrt(Bexp1*Bexp2)*r'
    for group, expr in [(1, damped_expression), (2, sr_expression)]:
        force = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(combined))
        force.setEnergyFunction(expr)
        force.setForceGroup(group)
        system.addForce(force)
    info = add_correction(system, data['coefficients'], data['box'],
                          cutoff=float(data['cutoff']), grid=ref_report['grid'])
    integrator = mm.VerletIntegrator(.001)
    cache = HERE/'build/cache'
    properties = {'Precision': 'double', 'TempDirectory': str(cache)} if platform == 'CUDA' else {}
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(platform), properties)
    context.setPositions(data['positions'])
    def evaluate(groups):
        state = context.getState(getEnergy=True, getForces=True, groups=groups)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)))
    groups = {'damped_cutoff': 1 << 1, 'SlaterSrDispForce': 1 << 2,
              'damped_pme': (1 << 1) | (1 << 30),
              'dispersion_full': (1 << 1) | (1 << 2) | (1 << 30),
              'combined_short_pme': 1 | (1 << 30)}
    comparisons = {}
    values = {}
    # Prespecified absolute tolerances; report failures rather than relaxing them.
    e_tol, f_tol = max(1e-4, len(data['positions'])*1e-7), 1e-3
    for name, mask in groups.items():
        start = time.perf_counter()
        e, f = evaluate(mask)
        values[name] = e, f
        re, rf = float(ref[name+'_energy']), ref[name+'_forces']
        de, df = e-re, f-rf
        comparisons[name] = dict(openmm_energy=e, dmff_energy=re, energy_error=de,
            force_max_error=float(np.max(np.abs(df))), force_rms_error=float(np.sqrt(np.mean(df*df))),
            reference_force_max=float(np.max(np.abs(rf))), elapsed_s=time.perf_counter()-start,
            passed=bool(abs(de) < e_tol and np.max(np.abs(df)) < f_tol))
        if 'rounded_'+name+'_energy' in ref:
            comparisons[name]['rounded_parameter_prediction'] = dict(
                energy_error=e-float(ref['rounded_'+name+'_energy']),
                force_max_error=float(np.max(np.abs(f-ref['rounded_'+name+'_forces']))))
        print(name, json.dumps(comparisons[name]), flush=True)
    report.update(platform=platform, precision='double' if platform == 'CUDA' else 'Reference double',
                  openmm=mm.__version__, dispersion=info, comparisons=comparisons,
                  tolerances=dict(energy_kj_mol=e_tol, force_component_kj_mol_nm=f_tol),
                  library_sha256=digest(HERE/'build/libNativeDispersion.so'),
                  reference_sha256=digest(reference/'reference.npz'),
                  full_dispersion_additivity_error=values['dispersion_full'][0]-values['damped_pme'][0]-values['SlaterSrDispForce'][0],
                  status='passed' if all(v['passed'] for v in comparisons.values()) else 'comparison_failed')
    with (output/'openmm.npz').open('xb') as handle:
        np.savez(handle, **{f'{n}_{kind}': v for n, vals in values.items() for kind, v in zip(['energy', 'forces'], vals)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine', choices=['dmff', 'openmm'])
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--platform', choices=['Reference', 'CUDA'], default='CUDA')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--diagnose-precision', action='store_true', help='DMFF: also predict CUDA float32 short-range parameter rounding')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.engine == 'openmm' and args.reference is None:
        parser.error('--reference is required for OpenMM')
    args.output.mkdir(parents=True, exist_ok=False)
    local_tmp = HERE/'build/tmp'
    local_tmp.mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(local_tmp)
    os.environ['MPLCONFIGDIR'] = str(HERE/'build/matplotlib_cache')
    data, pdb, original, types, hashes = load_case(args.case)
    report = dict(case=args.case, engine=args.engine, atoms=len(types), input_sha256=hashes,
                  script_sha256=digest(__file__), status='initializing',
                  scope='Damped dispersion and combined XML short-range + dispersion PME; no multipole/polarization or bonded energy')
    try:
        if args.engine == 'dmff':
            dmff_worker(data, original, types, args.output, report, args.diagnose_precision)
        else:
            openmm_worker(data, pdb, args.output, report, args.reference, args.platform)
    except Exception as exc:
        report.update(status='error', error=str(exc))
        raise
    finally:
        report['input_files_unchanged'] = all(digest(WORK/p) == h for p, h in hashes.items())
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('STATUS', report['status'], 'REPORT', args.output/'report.json', flush=True)
    if not report['input_files_unchanged'] or report['status'] == 'comparison_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
