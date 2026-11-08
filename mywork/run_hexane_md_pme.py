"""Hexane MD with OPLS bonded + PhyNEO physical forces + C6/C8/C10 dispersion PME.

Usage: python run_hexane_md_pme.py input.pdb length_ps temperature_K nvt|npt
Defaults: 1 fs, CUDA double, 1 bar isotropic NPT, DCD every 1 ps. No ML/JAX.
Optional output destination: PHYNEO_OUTPUT_DIR (otherwise mywork/runs).
The native dispersion implementation remains experimental; validate long runs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import run_hexane_md as previous

BASE = Path(__file__).resolve().parent
XML = previous.XML
LIBRARY = BASE/'native_dispersion/build/libNativeDispersion.so'
DISPERSION_GROUP = 30
TIMESTEP_PS = previous.TIMESTEP_PS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdb', help='PhyNEO-named hexane PDB with CRYST1 and CONECT')
    parser.add_argument('length_ps', help='Production duration in ps, rounded up to 1 fs')
    parser.add_argument('temperature_K', help='Temperature in kelvin')
    parser.add_argument('ensemble', type=str.lower, choices=['nvt', 'npt'])
    args = parser.parse_args(argv)
    return previous.parse_args([args.pdb, args.length_ps, args.temperature_K, args.ensemble])


def verify_installation(require_validated=False):
    if not LIBRARY.is_file():
        raise RuntimeError('Native dispersion library missing or changed; run scripts/build_portable.sh')
    from portable_runtime import load_manifest
    config = load_manifest(require_validated=require_validated)
    if Path(config['files']['dispersion']['path']).resolve() != LIBRARY.resolve():
        raise RuntimeError('Manifest selects a different native dispersion library')


def add_dispersion(system, box, mm, unit, np):
    """Read the actual existing force's coefficients; add only PME-minus-cutoff."""
    from native_dispersion.native import add_correction
    if any(f.getForceGroup() == DISPERSION_GROUP or f.getName() == 'NativeDispersionPMEMinusCutoff'
           for f in system.getForces()):
        raise ValueError('Dispersion already present or reserved force group 30 occupied')
    candidates = [f for f in system.getForces() if isinstance(f, mm.CustomNonbondedForce)]
    if len(candidates) != 1:
        raise ValueError('Expected exactly one existing damped short-range force')
    force = candidates[0]
    if force.getUseLongRangeCorrection() or force.getUseSwitchingFunction():
        raise ValueError('PME correction requires unswitched cutoff dispersion without a tail correction')
    cutoff = force.getCutoffDistance().value_in_unit(unit.nanometer)
    if force.getNonbondedMethod() != force.CutoffPeriodic or abs(cutoff-previous.CUTOFF_NM) > 1e-12:
        raise ValueError('Unexpected short-range cutoff method/distance')
    names = [force.getPerParticleParameterName(i) for i in range(force.getNumPerParticleParameters())]
    columns = [names.index(name) for name in ['C6', 'C8', 'C10']]
    coefficients = np.array([[force.getParticleParameters(i)[j] for j in columns]
                             for i in range(system.getNumParticles())])
    return add_correction(system, coefficients, box, cutoff=cutoff, group=DISPERSION_GROUP)


def main(argv=None):
    args = parse_args(argv)
    verify_installation(require_validated=True)
    mm, app, unit, plugin, np, libraries = previous.import_corrected_plugin()
    pdb = app.PDBFile(str(args.pdb))
    vectors = pdb.topology.getPeriodicBoxVectors()
    if vectors is None:
        raise ValueError('PDB must have a CRYST1 periodic box')
    box = np.asarray(vectors.value_in_unit(unit.nanometer))
    molecules = previous.validate_hexane(pdb.topology, box, np)
    if not np.allclose(box, np.diag(np.diag(box)), atol=1e-12, rtol=0):
        raise ValueError('Native dispersion currently requires an orthorhombic box')
    system, native = previous.build_system(pdb, mm, app, unit, plugin)
    dispersion = add_dispersion(system, box, mm, unit, np)
    if args.ensemble == 'npt':
        barostat = mm.MonteCarloBarostat(previous.PRESSURE_BAR*unit.bar,
                                       args.temperature_K*unit.kelvin, 25)
        barostat.setRandomNumberSeed(previous.SEED)
        system.addForce(barostat)
        print('NPT uses isotropic volume moves at 1 bar; use NVT for the fixed-box slab.', flush=True)
    integrator = mm.LangevinMiddleIntegrator(args.temperature_K*unit.kelvin,
        1/unit.picosecond, TIMESTEP_PS*unit.picosecond)
    integrator.setRandomNumberSeed(previous.SEED)
    destination = Path(os.environ.get('PHYNEO_OUTPUT_DIR', str(BASE/'runs'))).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = destination/f'{args.pdb.stem}_disp_pme_{args.ensemble}_{stamp}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    libraries[str(LIBRARY)] = previous.sha256(LIBRARY)
    interval = min(1000, args.steps)
    report = dict(status='initializing', input_pdb=str(args.pdb), input_sha256=previous.sha256(args.pdb),
        xml=str(XML), xml_sha256=previous.sha256(XML), libraries=libraries, openmm=mm.__version__,
        runner_sha256=previous.sha256(__file__), base_runner_sha256=previous.sha256(previous.__file__),
        precision='double', ensemble=args.ensemble, temperature_K=args.temperature_K,
        pressure_bar=previous.PRESSURE_BAR if args.ensemble == 'npt' else None,
        barostat_interval_steps=25 if args.ensemble == 'npt' else None,
        requested_length_ps=args.length_ps, steps=args.steps, timestep_fs=TIMESTEP_PS*1000,
        actual_length_ps=args.steps*TIMESTEP_PS, atoms=system.getNumParticles(), molecules=molecules,
        cutoff_nm=previous.CUTOFF_NM, electrostatics='PME', dispersion=dispersion,
        dispersion_description='C6/C8/C10 3D PME plus existing damping/Slater terms; no homogeneous tail',
        dispersion_grid_policy='grid and alpha fixed at construction; current box used in evaluations',
        native_scales={k: previous.NATIVE_SCALES for k in ('m', 'p', 'd')},
        cuda_corrections=['induced-field off-diagonal tj index', 'size_t scale uploads above 2 GiB'],
        polarization='mutual', mutual_target=previous.MUTUAL_EPSILON, mutual_max_iterations=1000,
        friction_per_ps=1, seed=previous.SEED, minimization_max_iterations=100,
        trajectory_format='DCD', report_interval_steps=interval,
        experimental_dispersion=True, output_directory=str(output))
    simulation = None

    def save_report():
        (output/'run.json').write_text(json.dumps(report, indent=2)+'\n')

    def snapshot(label):
        state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if not np.isfinite(energy) or not np.all(np.isfinite(forces)) or not np.all(np.isfinite(positions)):
            raise RuntimeError(f'Nonfinite state at {label}')
        correction = simulation.context.getState(getEnergy=True, groups=1 << DISPERSION_GROUP)
        result = dict(energy_kj_mol=energy,
            dispersion_correction_kj_mol=correction.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            max_force_kj_mol_nm=float(np.max(np.linalg.norm(forces, axis=1))),
            box_nm=np.asarray(state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tolist())
        print(label, json.dumps(result), flush=True)
        return result

    print(f'Outputs: {output}\nCUDA double, 1 fs, corrected scales/kernels, native dispersion PME. '
          'Experimental implementation; no JAX runtime.', flush=True)
    save_report()
    try:
        simulation = app.Simulation(pdb.topology, system, integrator,
            mm.Platform.getPlatformByName('CUDA'), previous.cuda_properties())
        if simulation.context.getPlatform().getPropertyValue(simulation.context, 'Precision') != 'double':
            raise RuntimeError('CUDA double precision was not selected')
        simulation.context.setPositions(pdb.positions)
        report['initial'] = snapshot('initial')
        report['actual_electrostatic_pme'] = list(native.getPMEParametersInContext(simulation.context))
        simulation.minimizeEnergy(maxIterations=100)
        report['minimized'] = snapshot('minimized')
        simulation.context.setVelocitiesToTemperature(args.temperature_K*unit.kelvin, previous.SEED)
        simulation.reporters.append(app.DCDReporter(str(output/'trajectory.dcd'), interval))
        simulation.reporters.append(app.StateDataReporter(str(output/'thermodynamics.csv'), interval,
            step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
            temperature=True, volume=True, density=True, speed=True))
        simulation.reporters.append(app.CheckpointReporter(str(output/'restart.chk'), interval))
        report['status'] = 'running'
        save_report()
        simulation.step(args.steps)
        report['final'] = snapshot('final')
        final = simulation.context.getState(getPositions=True, getVelocities=True)
        simulation.topology.setPeriodicBoxVectors(final.getPeriodicBoxVectors())
        # One final topology/coordinate snapshot, never a PDB trajectory.
        with (output/'final.pdb').open('w') as handle:
            app.PDBFile.writeFile(simulation.topology, final.getPositions(), handle)
        simulation.saveCheckpoint(str(output/'restart.chk'))
        (output/'final_state.xml').write_text(mm.XmlSerializer.serialize(final))
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report['completed_steps'] = simulation.currentStep if simulation is not None else 0
        save_report()
    print(f'Completed {args.steps} steps ({args.steps*TIMESTEP_PS:g} ps). Saved {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
