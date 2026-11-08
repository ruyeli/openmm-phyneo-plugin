"""Original DMFF C6/C8/C10 energy and force reference, CPU ONLY."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
import argparse
from pathlib import Path
import json
import time
import hashlib
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():p.error('Output exists')
    import jax
    import jax.numpy as jnp
    from dmff.admp.disp_pme import ADMPDispPmeForce
    import dmff
    jax.config.update('jax_enable_x64',True)
    data=np.load(a.input)
    x=jnp.asarray(data['positions']);box=jnp.asarray(data['box']);pairs=jnp.asarray(data['pairs'])
    coefficients=jnp.asarray(np.sqrt(data['coefficients'])*np.array([1e3,1e4,1e5]))
    scales=jnp.asarray(data['scales']);cutoff=float(data['cutoff'])
    full=ADMPDispPmeForce(box*10,cutoff*10,1e-4,10,lpme=True)
    cut=ADMPDispPmeForce(box*10,cutoff*10,1e-4,10,lpme=False)
    # The original generator negates ADMPDispPmeForce.get_energy; retain this sign.
    def pme(pos):return -full.get_energy(pos*10,box*10,pairs,coefficients,scales)
    def truncated(pos):return -cut.get_energy(pos*10,box*10,pairs,coefficients,scales)
    results={};timings={}
    for name,fn in [('pme',pme),('cutoff',truncated),('correction',lambda pos:pme(pos)-truncated(pos))]:
        start=time.perf_counter();ev=jax.jit(jax.value_and_grad(fn));energy,gradient=ev(x)
        results[name+'_energy']=float(energy);results[name+'_forces']=-np.asarray(gradient)
        timings[name+'_compile_evaluate_s']=time.perf_counter()-start
        start=time.perf_counter();ev(x)[0].block_until_ready();timings[name+'_warm_s']=time.perf_counter()-start
        print(name,results[name+'_energy'],timings,flush=True)
    assert all(np.all(np.isfinite(v)) for v in results.values())
    with a.output.open('xb') as handle:np.savez(handle,**results)
    module=Path(dmff.__file__).resolve().parent
    report=dict(input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],grid=[full.K1,full.K2,full.K3],
                alpha_per_nm=float(full.kappa)*10,atoms=len(x),pairs=len(pairs),timings=timings,
                energies={k:v for k,v in results.items() if k.endswith('_energy')},
                source_sha256={name:hashlib.sha256((module/name).read_bytes()).hexdigest() for name in
                               ['admp/disp_pme.py','admp/recip.py','admp/pme.py','generators/admp.py']})
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
