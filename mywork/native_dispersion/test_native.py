"""Native OpenMM correction regression against original CPU DMFF snapshots."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
cache=HERE/'build/cache';cache.mkdir(parents=True,exist_ok=True)
os.environ['OPENMM_CACHE_DIR']=str(cache)
os.environ['TMPDIR']=str(HERE/'build/tmp')
import openmm as mm
from openmm import unit
from native import add_correction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--platform',choices=['Reference','CPU','CUDA'],required=True)
    p.add_argument('--no-finite-difference',action='store_true')
    p.add_argument('--correction-only',action='store_true',help='Assert the new correction only; still report raw-cutoff/total discrepancies')
    p.add_argument('--diagnose-raw-precision',action='store_true',help='Compare raw cutoff with float32-rounded parameters evaluated in NumPy float64')
    a=p.parse_args()
    if a.output.exists():p.error('Output exists')
    data=np.load(a.input);ref=np.load(a.reference);meta=json.loads(a.reference.with_suffix('.json').read_text())
    system=mm.System()
    for _ in data['positions']:system.addParticle(12.)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row)*unit.nanometer for row in data['box']])
    raw=mm.CustomNonbondedForce('-c61*c62/r^6-c81*c82/r^8-c101*c102/r^10')
    for param in ['c6','c8','c10']:raw.addPerParticleParameter(param)
    for coeff in data['coefficients']:raw.addParticle(np.sqrt(coeff).tolist())
    for i,j in data['exclusions']:raw.addExclusion(int(i),int(j))
    raw.setNonbondedMethod(raw.CutoffPeriodic);raw.setCutoffDistance(float(data['cutoff']))
    raw.setUseLongRangeCorrection(False);raw.setForceGroup(0);system.addForce(raw)
    info=add_correction(system,data['coefficients'],data['box'],cutoff=float(data['cutoff']),grid=meta['grid'])
    integrator=mm.VerletIntegrator(.001)
    props={'Precision':'double','TempDirectory':str(cache)} if a.platform=='CUDA' else ({'Threads':'1'} if a.platform=='CPU' else {})
    context=mm.Context(system,integrator,mm.Platform.getPlatformByName(a.platform),props)
    context.setPositions(data['positions'])
    def evaluate(group):
        state=context.getState(getEnergy=True,getForces=True,groups=group)
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),np.asarray(state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer))
    start=time.perf_counter();energy,forces=evaluate(1<<30);cold=time.perf_counter()-start
    start=time.perf_counter();e2,f2=evaluate(1<<30);warm=time.perf_counter()-start
    ecut,fcut=evaluate(1)
    total,ftotal=evaluate((1<<30)|1)
    report=dict(platform=a.platform,atoms=len(forces),info=info,correction_energy=energy,
                asserted_scope='correction only' if a.correction_only else 'correction and reconstructed raw dispersion',
                input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),
                library_sha256=hashlib.sha256((HERE/'build/libNativeDispersion.so').read_bytes()).hexdigest(),
                cutoff_energy=ecut,total_dispersion=total,cold_s=cold,warm_s=warm,
                correction_energy_error=energy-float(ref['correction_energy']),
                correction_force_max_error=float(np.max(np.abs(forces-ref['correction_forces']))),
                total_energy_error=total-float(ref['pme_energy']),
                total_force_max_error=float(np.max(np.abs(ftotal-ref['pme_forces']))),
                reference_total_force_max=float(np.max(np.abs(ref['pme_forces']))),
                raw_cutoff_energy_error=ecut-float(ref['cutoff_energy']),
                raw_cutoff_force_max_error=float(np.max(np.abs(fcut-ref['cutoff_forces']))),
                repeat_energy_error=e2-energy,repeat_force_max_error=float(np.max(np.abs(forces-f2))))
    if a.diagnose_raw_precision:
        pairs=data['pairs'];pairs=pairs[data['scales'][pairs[:,2]-1]!=0]
        i,j=pairs[:,0],pairs[:,1]
        dr=data['positions'][i]-data['positions'][j];box=np.diag(data['box'])
        dr-=np.rint(dr/box)*box;r2=np.sum(dr*dr,axis=1)
        q=np.sqrt(data['coefficients']).astype(np.float32).astype(np.float64)
        terms=q[i]*q[j]/r2[:,None]**np.array([3,4,5])
        rounded_energy=-np.sum(terms)
        pair_forces=-np.sum(terms*np.array([6,8,10]),axis=1)[:,None]*dr/r2[:,None]
        rounded_forces=np.zeros_like(data['positions'])
        np.add.at(rounded_forces,i,pair_forces);np.add.at(rounded_forces,j,-pair_forces)
        report['raw_float32_parameter_diagnostic']=dict(energy=float(rounded_energy),
            energy_error_vs_dmff=float(rounded_energy-ref['cutoff_energy']),
            energy_error_vs_openmm=float(rounded_energy-ecut),
            force_max_error_vs_dmff=float(np.max(np.abs(rounded_forces-ref['cutoff_forces']))),
            force_max_error_vs_openmm=float(np.max(np.abs(rounded_forces-fcut))))
    report['status']='incomplete'
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    assert abs(report['correction_energy_error'])<1e-5
    assert report['correction_force_max_error']<1e-4
    # OpenMM CustomNonbondedForce parameter storage introduces small CUDA/CPU single-precision effects.
    report['raw_plus_correction_meets_absolute_tolerances']=(abs(report['total_energy_error'])<max(1e-4,len(forces)*1e-7)
                                                           and report['total_force_max_error']<1e-3)
    if not a.correction_only:
        assert report['raw_plus_correction_meets_absolute_tolerances']
    if not a.no_finite_difference:
        errors=[]
        for i,k in [(0,0),(31,2)]:
            delta=1e-5;x=data['positions'].copy();x[i,k]+=delta;context.setPositions(x);plus=evaluate(1<<30)[0]
            x[i,k]-=2*delta;context.setPositions(x);minus=evaluate(1<<30)[0]
            errors.append(abs(-(plus-minus)/(2*delta)-forces[i,k]))
        report['finite_difference_max_error']=max(errors);assert max(errors)<1e-3
        x=data['positions'].copy();x[0]+=data['box'][0];context.setPositions(x);shifted,sf=evaluate(1<<30)
        report['periodic_image_energy_error']=shifted-energy
        report['periodic_image_force_error']=float(np.max(np.abs(sf-forces)))
        assert abs(shifted-energy)<1e-7 and np.max(np.abs(sf-forces))<1e-5
    report['status']='passed'
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS',a.platform,flush=True)


if __name__=='__main__':main()
