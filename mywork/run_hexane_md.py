"""Hexane OPLS-bonded + PhyNEO physical nonbonded MD, using corrected CUDA.

Usage: python run_hexane_md.py input.pdb length_ps temperature_K nvt|npt
Fixed defaults: 1 fs, CUDA double, 1 bar for NPT. No ML terms.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import uuid
import xml.etree.ElementTree as ET

BASE = Path(__file__).resolve().parent
XML = BASE / 'phyneo_hexane_opls_md.xml'
# The CUDA fixes now live in the canonical source. Local build fingerprints
# come from portable_runtime rather than binaries compiled on another cluster.
NATIVE_SCALES = [1, 0, 0, 0, 0, 0]
TIMESTEP_PS = 0.001
PRESSURE_BAR = 1.0
CUTOFF_NM = 0.6
MUTUAL_EPSILON = 1e-5
SEED = 2026


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cuda_properties():
    # Keep reusable JIT artifacts out of /tmp and out of individual run folders.
    cache = BASE/'runs/.cuda_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['OPENMM_CACHE_DIR'] = str(cache)
    return {'Precision': 'double', 'TempDirectory': str(cache)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdb', type=Path, help='Hexane PDB with CRYST1 and connectivity')
    parser.add_argument('length_ps', help='Production MD duration in ps; rounded up to 1 fs')
    parser.add_argument('temperature_K', type=float)
    parser.add_argument('ensemble', type=str.lower, choices=['nvt', 'npt'])
    args = parser.parse_args(argv)
    try:
        duration = Decimal(args.length_ps)
        if not duration.is_finite() or duration <= 0:
            raise ValueError()
        args.steps = int((duration / Decimal('0.001')).to_integral_value(rounding=ROUND_CEILING))
        args.length_ps = float(duration)
        if not math.isfinite(args.length_ps):
            raise ValueError()
    except (InvalidOperation, ValueError, OverflowError):
        parser.error('length_ps must be a finite positive number')
    if not math.isfinite(args.temperature_K) or args.temperature_K <= 0:
        parser.error('temperature_K must be finite and positive')
    args.pdb = args.pdb.resolve()
    if not args.pdb.is_file():
        parser.error(f'PDB does not exist: {args.pdb}')
    return args


def import_corrected_plugin():
    """Choose the tested library BEFORE OpenMM automatically loads plugins."""
    if 'openmm' in sys.modules:
        raise RuntimeError('Start a fresh Python process: OpenMM was already imported.')
    from portable_runtime import load_manifest
    config = load_manifest()
    library = Path(config['files']['cuda']['path'])
    os.environ['OPENMM_PLUGIN_DIR'] = str(library.parent)
    import numpy as np
    import openmm as mm
    from openmm import app, unit
    import phyneoplugin
    loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                     if 'PhyNEO' in line and '.so' in line})
    cuda = [Path(f) for f in loaded if Path(f).name == 'libPhyNEOPluginCUDA.so']
    core = [Path(f) for f in loaded if Path(f).name == 'libPhyNEOPlugin.so']
    if cuda != [library.resolve()] or len(core) != 1 or sha256(core[0]) != config['files']['core']['sha256']:
        raise RuntimeError(f'Unexpected PhyNEO libraries loaded: {loaded}')
    return mm, app, unit, phyneoplugin, np, {f: sha256(f) for f in loaded}


def validate_hexane(topology, box, np):
    """Reject missing connectivity, non-hexane inputs, and invalid periodic boxes."""
    atoms = list(topology.atoms())
    if not atoms:
        raise ValueError('PDB contains no atoms')
    neighbors = [set() for _ in atoms]
    for a, b in topology.bonds():
        neighbors[a.index].add(b.index)
        neighbors[b.index].add(a.index)
    remaining = set(range(len(atoms)))
    molecules = 0
    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            i = stack.pop()
            new = neighbors[i] & remaining
            remaining.difference_update(new)
            component.update(new)
            stack.extend(new)
        symbols = {i: atoms[i].element.symbol if atoms[i].element else '?' for i in component}
        if Counter(symbols.values()) != Counter(C=6, H=14):
            raise ValueError('Each connected molecule must be C6H14. Check PDB CONECT records.')
        carbon_degrees = []
        for i in component:
            if symbols[i] == 'H':
                if len(neighbors[i]) != 1 or symbols[next(iter(neighbors[i]))] != 'C':
                    raise ValueError('Invalid hexane hydrogen connectivity')
            else:
                if len(neighbors[i]) != 4:
                    raise ValueError('Each hexane carbon must have four bonds')
                carbon_degrees.append(sum(symbols[j] == 'C' for j in neighbors[i]))
        if sorted(carbon_degrees) != [1, 1, 2, 2, 2, 2]:
            raise ValueError('This force field is for linear n-hexane, not branched isomers')
        molecules += 1
    volume = float(np.linalg.det(box))
    areas = [np.linalg.norm(np.cross(box[(i+1) % 3], box[(i+2) % 3])) for i in range(3)]
    if not np.all(np.isfinite(box)) or volume <= 0 or min(areas) <= 0:
        raise ValueError('PDB must contain a valid periodic box (CRYST1)')
    if min(volume / area for area in areas) <= 2*CUTOFF_NM:
        raise ValueError('Periodic box face heights must exceed twice the 0.6 nm cutoff')
    return molecules


def build_system(pdb, mm, app, unit, plugin):
    """Preserve the archived XML; apply the validated mapping to the live force."""
    root = ET.parse(XML).getroot()
    multipoles = root.find('ADMPPmeForce')
    if multipoles is None or any(
            float(multipoles.get(f'{family}Scale{distance}', 'nan')) != expected
            for family, expected in [('m', 0), ('p', 0), ('d', 1)]
            for distance in range(12, 17)):
        raise ValueError('XML scales differ from the validated hexane model; do not apply this mapping blindly')
    script = root.find('Script')
    marker = '    system.addForce(force)'
    if script is None or script.text.count(marker) != 1:
        raise ValueError('Unexpected XML reconstruction script; cannot retain typed PhyNEO force')
    script.text = script.text.replace(marker, marker+'\n    system._hexane_native = force')
    system = app.ForceField(io.StringIO(ET.tostring(root, encoding='unicode'))).createSystem(
        pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=CUTOFF_NM*unit.nanometer,
        constraints=None, rigidWater=False, removeCMMotion=True,
        polarization='mutual', ewaldErrorTolerance=1e-4,
        mutualInducedTargetEpsilon=MUTUAL_EPSILON, mutualInducedMaxIterations=1000)
    native = system._hexane_native
    # DMFF pScale controls permanent-induced coupling. BOTH native p and
    # native d (fixed-field scaling) must receive it for this model. Native
    # induced-induced coupling is already unscaled, matching original DMFF d=1.
    # This override MUST follow the XML Script, which restores the old mapping.
    native.setMultipoleScaleFactors(NATIVE_SCALES, NATIVE_SCALES, NATIVE_SCALES)
    native.setPolarizationType(plugin.PhyNEOForce.Mutual)
    native.setMutualInducedTargetEpsilon(MUTUAL_EPSILON)
    native.setMutualInducedMaxIterations(1000)
    m, p, d = plugin.vectord(), plugin.vectord(), plugin.vectord()
    native.getMultipoleScaleFactors(m, p, d)
    if any(list(values) != NATIVE_SCALES for values in (m, p, d)):
        raise RuntimeError('Native scale mapping was not applied')
    for group, force in enumerate(system.getForces()):
        force.setForceGroup(group)
    return system, native


def main(argv=None):
    args = parse_args(argv)
    from portable_runtime import load_manifest
    load_manifest(require_validated=True)
    mm, app, unit, plugin, np, libraries = import_corrected_plugin()
    pdb = app.PDBFile(str(args.pdb))
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError('PDB must supply periodic box vectors through CRYST1')
    box = np.asarray(vectors.value_in_unit(unit.nanometer))
    molecules = validate_hexane(pdb.topology, box, np)
    system, native = build_system(pdb, mm, app, unit, plugin)
    if args.ensemble == 'npt':
        barostat = mm.MonteCarloBarostat(PRESSURE_BAR*unit.bar, args.temperature_K*unit.kelvin, 25)
        barostat.setRandomNumberSeed(SEED)
        system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(args.temperature_K*unit.kelvin,
                                            1/unit.picosecond, TIMESTEP_PS*unit.picoseconds)
    integrator.setRandomNumberSeed(SEED)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = BASE/'runs'/f'{args.pdb.stem}_{args.ensemble}_{stamp}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    properties = cuda_properties()
    simulation = app.Simulation(pdb.topology, system, integrator,
                                mm.Platform.getPlatformByName('CUDA'), properties)
    actual_precision = simulation.context.getPlatform().getPropertyValue(simulation.context, 'Precision')
    if actual_precision != 'double':
        raise RuntimeError(f'Unexpected CUDA precision: {actual_precision}')
    simulation.context.setPositions(pdb.positions)
    report = dict(status='initializing', input_pdb=str(args.pdb), input_sha256=sha256(args.pdb),
                  xml=str(XML), xml_sha256=sha256(XML), openmm=mm.__version__, libraries=libraries,
                  cuda_correction='off-diagonal induced-field atom2 uses tj, not j',
                  cuda_scale_upload_correction='size_t byte count for dense scale arrays above 2 GiB',
                  native_scales={k: NATIVE_SCALES for k in ('m', 'p', 'd')},
                  precision=actual_precision, ensemble=args.ensemble, temperature_K=args.temperature_K,
                  pressure_bar=PRESSURE_BAR if args.ensemble == 'npt' else None,
                  requested_length_ps=args.length_ps, steps=args.steps, timestep_fs=1.0,
                  actual_length_ps=args.steps*TIMESTEP_PS, molecules=molecules,
                  atoms=system.getNumParticles(), cutoff_nm=CUTOFF_NM,
                  polarization='mutual', mutual_target=MUTUAL_EPSILON, mutual_max_iterations=1000,
                  electrostatics='PME', dispersion='truncated, no tail correction',
                  friction_per_ps=1.0, seed=SEED, minimization_max_iterations=100)

    def save_report():
        (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')

    def snapshot(label):
        state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(np.isfinite(positions)):
            raise RuntimeError(f'Nonfinite energy, forces or positions at {label}')
        metrics = dict(energy_kj_mol=energy,
                       max_force_kj_mol_nm=float(np.max(np.linalg.norm(forces, axis=1))),
                       box_nm=np.asarray(state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tolist())
        print(label, json.dumps(metrics), flush=True)
        return metrics

    print(f'Outputs: {output}\nCUDA double; corrected kernel; native m=p=d={NATIVE_SCALES}', flush=True)
    save_report()
    try:
        report['initial'] = snapshot('initial')
        report['actual_pme'] = list(native.getPMEParametersInContext(simulation.context))
        simulation.minimizeEnergy(maxIterations=100)
        report['minimized'] = snapshot('minimized')
        simulation.context.setVelocitiesToTemperature(args.temperature_K*unit.kelvin, SEED)
        interval = min(1000, args.steps)
        simulation.reporters.append(app.DCDReporter(str(output/'trajectory.dcd'), interval))
        simulation.reporters.append(app.StateDataReporter(
            str(output/'thermodynamics.csv'), interval, step=True, time=True,
            potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True, speed=True))
        simulation.reporters.append(app.CheckpointReporter(str(output/'restart.chk'), interval))
        report['status'] = 'running'
        save_report()
        simulation.step(args.steps)
        report['final'] = snapshot('final')
        final = simulation.context.getState(getPositions=True, getVelocities=True)
        simulation.topology.setPeriodicBoxVectors(final.getPeriodicBoxVectors())
        with (output/'final.pdb').open('w') as handle:
            app.PDBFile.writeFile(simulation.topology, final.getPositions(), handle)
        simulation.saveCheckpoint(str(output/'restart.chk'))
        (output/'final_state.xml').write_text(mm.XmlSerializer.serialize(final))
        report['status'] = 'complete'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        report['completed_steps'] = simulation.currentStep
        save_report()
    print(f'Completed {args.steps} steps ({args.steps*TIMESTEP_PS:g} ps). Saved {output}', flush=True)


if __name__ == '__main__':
    main()
