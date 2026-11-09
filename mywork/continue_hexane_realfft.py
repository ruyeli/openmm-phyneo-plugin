"""Resume NVT hexane with validated double real FFTs for all PME components.

Usage: python continue_hexane_realfft.py source_run [additional_ps]
Without additional_ps, finish the source run's original target step.
PHYNEO_OUTPUT_DIR controls the destination; default is source_run's parent.
Always create a new output directory. Never append to the source trajectory.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import signal
import uuid

import run_hexane_md_pme as md

BASE = Path(__file__).resolve().parent


def duration_steps(text):
    try:
        value = Decimal(text)
        if not value.is_finite() or value <= 0:
            raise ValueError()
        return int((value/Decimal('0.001')).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, ValueError, OverflowError):
        raise argparse.ArgumentTypeError('additional_ps must be finite and positive')


def atomic_write(path, data):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_source(old):
    required = dict(ensemble='nvt', precision='double', timestep_fs=1.0,
                    cutoff_nm=md.previous.CUTOFF_NM, polarization='mutual',
                    mutual_target=md.previous.MUTUAL_EPSILON,
                    mutual_max_iterations=1000, friction_per_ps=1,
                    xml_sha256=md.previous.sha256(md.XML),
                    native_scales={k: md.previous.NATIVE_SCALES for k in ('m', 'p', 'd')})
    for key, expected in required.items():
        if old.get(key) != expected:
            raise ValueError(f'Source setting {key} is not supported: {old.get(key)!r}; expected {expected!r}')
    if not (0 < float(old['temperature_K']) < float('inf')):
        raise ValueError('Invalid source temperature')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_run', type=Path)
    parser.add_argument('additional_ps', nargs='?', type=duration_steps,
                        help='Optional ADDITIONAL duration, in ps; otherwise finish original target')
    args = parser.parse_args(argv)
    source = args.source_run.expanduser().resolve()
    old = json.loads((source/'run.json').read_text())
    validate_source(old)
    pdb_path = Path(os.environ.get('PHYNEO_INPUT_PDB', old['input_pdb'])).expanduser().resolve()
    if md.previous.sha256(pdb_path) != old['input_sha256']:
        raise ValueError('Original topology PDB has changed')
    checkpoint_path = source/'restart.chk'
    checkpoint = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
    if checkpoint_hash != md.previous.sha256(checkpoint_path):
        raise RuntimeError('Source checkpoint is changing; stop its writer before restarting')
    # Canonical source now supplies all real-FFT kernels. Use the locally built,
    # hash-verified and GPU-validated libraries, not another cluster's binaries.
    md.verify_installation(require_validated=True)
    mm, app, unit, plugin, np, libraries = md.previous.import_corrected_plugin()
    if mm.__version__ != old['openmm']:
        raise RuntimeError('Binary checkpoint requires the original OpenMM version; no silent State/PDB fallback')
    pdb = app.PDBFile(str(pdb_path))
    box = np.asarray(pdb.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
    molecules = md.previous.validate_hexane(pdb.topology, box, np)
    if not np.allclose(box, np.diag(np.diag(box)), atol=1e-12, rtol=0):
        raise ValueError('This all-real-FFT runner requires an orthorhombic box')
    system, native = md.previous.build_system(pdb, mm, app, unit, plugin)
    dispersion = md.add_dispersion(system, box, mm, unit, np)
    if dispersion != old['dispersion'] or system.getNumParticles() != old['atoms']:
        raise ValueError('Rebuilt dispersion settings or atom count differ from source')
    integrator = mm.LangevinMiddleIntegrator(old['temperature_K']*unit.kelvin,
        old['friction_per_ps']/unit.picosecond, md.TIMESTEP_PS*unit.picosecond)
    integrator.setRandomNumberSeed(old['seed'])
    simulation = app.Simulation(pdb.topology, system, integrator,
        mm.Platform.getPlatformByName('CUDA'), md.previous.cuda_properties())
    if simulation.context.getPlatform().getPropertyValue(simulation.context, 'Precision') != 'double':
        raise RuntimeError('CUDA double was not selected')
    # Do not set positions, velocities, time, or step count after this load.
    simulation.context.loadCheckpoint(checkpoint)
    start_step = simulation.currentStep
    original_target = int(old.get('target_total_steps', old['steps']))
    target = original_target if args.additional_ps is None else start_step+args.additional_ps
    if target <= start_step:
        raise ValueError(f'Checkpoint step {start_step} already reaches target {target}; specify additional_ps')
    actual_pme = list(native.getPMEParametersInContext(simulation.context))
    if actual_pme != old['actual_electrostatic_pme']:
        raise ValueError('Electrostatic PME grid/alpha changed')

    def snapshot():
        state = simulation.context.getState(getPositions=True, getVelocities=True,
                                             getEnergy=True, getForces=True)
        arrays = [state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                  state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer/unit.picosecond),
                  state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)]
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        current_box = np.asarray(state.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
        if not all(np.isfinite(a).all() for a in arrays+[current_box, np.asarray([energy, kinetic])]):
            raise RuntimeError('Nonfinite restart/simulation state')
        if not np.allclose(current_box, box, atol=1e-10, rtol=0):
            raise ValueError('NVT checkpoint box differs from the original orthorhombic box')
        return state, dict(step=simulation.currentStep, time_ps=state.getTime().value_in_unit(unit.picosecond),
            energy_kj_mol=energy, kinetic_kj_mol=kinetic,
            max_force_kj_mol_nm=float(np.max(np.linalg.norm(arrays[2], axis=1))), box_nm=current_box.tolist())

    _, initial = snapshot()
    destination = Path(os.environ.get('PHYNEO_OUTPUT_DIR', str(source.parent))).expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = destination/f'{pdb_path.stem}_allreal_restart_{stamp}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True, exist_ok=False)
    libraries[str(md.LIBRARY)] = md.previous.sha256(md.LIBRARY)
    interval = min(1000, target-start_step)
    report = dict(old)
    for stale in ('initial', 'minimized', 'final', 'error', 'completed_steps'):
        report.pop(stale, None)
    report.update(status='running', source_run=str(source), source_checkpoint_sha256=checkpoint_hash,
        source_manifest_sha256=md.previous.sha256(source/'run.json'),
        output_directory=str(output), input_pdb=str(pdb_path), libraries=libraries,
        runner_sha256=md.previous.sha256(__file__), base_runner_sha256=md.previous.sha256(md.previous.__file__),
        pme_runner_sha256=md.previous.sha256(md.__file__), dispersion=dispersion,
        actual_electrostatic_pme=actual_pme, start_step=start_step, start_time_ps=initial['time_ps'],
        original_target_total_steps=original_target, target_total_steps=target,
        steps=target-start_step, requested_length_ps=(target-start_step)*md.TIMESTEP_PS,
        actual_length_ps=(target-start_step)*md.TIMESTEP_PS, report_interval_steps=interval,
        minimization_max_iterations=0, velocities_reassigned=False, restart_method='binary checkpoint',
        electrostatic_fft='double D2Z/Z2D: permanent multipoles and induced-dipole iterations',
        electrostatic_fft_optimization='real half-spectrum; skip redundant initial induced FFT pair',
        dispersion_fft='double D2Z/Z2D for C6/C8/C10',
        dispersion_real_space='GPU cell/Verlet list, 0.2 nm skin',
        initial=initial, continuation_steps=0, completed_steps=start_step,
        validation_caveat='FFT equivalence validated; shared NVE fluctuations and prior initcheck warnings unresolved')

    def save_report():
        atomic_write(output/'run.json', (json.dumps(report, indent=2)+'\n').encode())

    class AtomicCheckpointReporter(app.CheckpointReporter):
        def report(self, sim, state):
            atomic_write(output/'restart.chk', sim.context.createCheckpoint())

    stop = []
    def request_stop(signum, frame):
        stop.append(signum)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    atomic_write(output/'source_checkpoint.chk', checkpoint)
    atomic_write(output/'restart.chk', simulation.context.createCheckpoint())
    save_report()
    simulation.reporters.append(app.DCDReporter(str(output/'trajectory.dcd'), interval))
    simulation.reporters.append(app.StateDataReporter(str(output/'thermodynamics.csv'), interval,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True))
    simulation.reporters.append(AtomicCheckpointReporter(str(output/'restart.chk'), interval))
    print(f'Outputs: {output}\nRestart step {start_step} ({initial["time_ps"]:g} ps); '
          f'target {target}; additional {(target-start_step)*md.TIMESTEP_PS:g} ps.\n'
          'CUDA double; all PME real FFT; no minimization or new velocities.', flush=True)
    try:
        while simulation.currentStep < target and not stop:
            simulation.step(min(1000, target-simulation.currentStep))
        final_state, report['final'] = snapshot()
        atomic_write(output/'restart.chk', simulation.context.createCheckpoint())
        atomic_write(output/'final_state.xml', mm.XmlSerializer.serialize(final_state).encode())
        report['status'] = 'complete' if simulation.currentStep >= target else 'interrupted'
        report['stop_signals'] = stop
    except Exception as exc:
        # Keep the last successful periodic checkpoint; do not overwrite it with
        # a potentially invalid Context after a CUDA/force evaluation failure.
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report['completed_steps'] = simulation.currentStep
        report['continuation_steps'] = simulation.currentStep-start_step
        report['source_checkpoint_unchanged'] = md.previous.sha256(checkpoint_path) == checkpoint_hash
        save_report()
    print(f'{report["status"]}: step {simulation.currentStep}. Saved {output}', flush=True)
    return output


if __name__ == '__main__':
    main()
