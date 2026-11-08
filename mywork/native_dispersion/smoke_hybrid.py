"""Separate CUDA-double hybrid smoke test; does not modify the production runner."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
sys.path.insert(0,str(WORK))
import run_hexane_md as md


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdb',type=Path)
    p.add_argument('--steps',type=int,default=10)
    p.add_argument('--minimize',type=int,default=0)
    a=p.parse_args()
    if a.steps<0 or a.minimize<0:p.error('Step counts must be nonnegative')
    output=HERE/'results'/('hybrid_'+a.pdb.stem+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    mm,app,unit,plugin,np,libraries=md.import_corrected_plugin()
    from native import add_correction
    from prepare_input import prepare
    data=prepare(a.pdb,WORK/'npt_ts_1fs/output.xml')
    pdb=app.PDBFile(str(a.pdb.resolve()))
    system,_=md.build_system(pdb,mm,app,unit,plugin)
    info=add_correction(system,data['coefficients'],data['box'],md.CUTOFF_NM)
    integrator=mm.LangevinMiddleIntegrator(450*unit.kelvin,1/unit.picosecond,.001*unit.picosecond)
    integrator.setRandomNumberSeed(2026)
    simulation=app.Simulation(pdb.topology,system,integrator,mm.Platform.getPlatformByName('CUDA'),md.cuda_properties())
    simulation.context.setPositions(pdb.positions)
    report=dict(status='initializing',pdb=str(a.pdb.resolve()),pdb_sha256=md.sha256(a.pdb),
                xml_sha256=md.sha256(md.XML),library_sha256=md.sha256(HERE/'build/libNativeDispersion.so'),
                baseline_libraries=libraries,dispersion=info,steps_requested=a.steps,
                timestep_fs=1,temperature_K=450,ensemble='nvt',precision='double',
                mutual_epsilon=md.MUTUAL_EPSILON,native_scales=md.NATIVE_SCALES)
    def state(groups=-1):
        s=simulation.context.getState(getEnergy=True,getForces=True,groups=groups)
        e=s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        f=s.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        if not np.isfinite(e) or not np.all(np.isfinite(f)):raise RuntimeError('Nonfinite energy/force')
        return e,f
    print('Output:',output,flush=True)
    try:
        baseline,bf=state((1<<30)-1)
        start=time.perf_counter();correction,cf=state(1<<30);report['correction_cold_s']=time.perf_counter()-start
        start=time.perf_counter();state(1<<30);report['correction_warm_s']=time.perf_counter()-start
        total,tf=state()
        report.update(baseline_energy=baseline,correction_energy=correction,total_energy=total,
                      energy_additivity_error=total-baseline-correction,
                      force_additivity_max_error=float(np.max(np.abs(tf-bf-cf))))
        assert abs(report['energy_additivity_error'])<1e-5
        assert report['force_additivity_max_error']<1e-3
        print(json.dumps(report),flush=True)
        if a.minimize:
            simulation.minimizeEnergy(maxIterations=a.minimize)
            report['minimized_energy']=state()[0]
        if a.steps:
            simulation.context.setVelocitiesToTemperature(450*unit.kelvin,2026)
            simulation.reporters.append(app.DCDReporter(str(output/'trajectory.dcd'),max(1,min(10,a.steps))))
            start=time.perf_counter();simulation.step(a.steps)
            report['final_energy']=state()[0]
            report['steps_elapsed_s']=time.perf_counter()-start
            report['estimated_ns_per_day']=a.steps/report['steps_elapsed_s']*.0864
            simulation.saveCheckpoint(str(output/'restart.chk'))
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        report['completed_steps']=simulation.currentStep
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
