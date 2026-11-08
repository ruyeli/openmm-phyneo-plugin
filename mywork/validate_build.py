"""Validate a local build on a GPU node before enabling the MD runners."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from portable_runtime import BASE, ROOT, MANIFEST, load_manifest, sha256


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    config = load_manifest()
    # Invalidate an earlier successful validation before starting a new one.
    config['gpu_validation'] = None
    MANIFEST.write_text(json.dumps(config, indent=2)+'\n')
    output = BASE/'validation'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    env = os.environ.copy()
    env['OPENMM_PLUGIN_DIR'] = str(Path(config['files']['cuda']['path']).parent)
    env['TMPDIR'] = str(ROOT/'build-portable/tmp')
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    fixtures = BASE/'native_dispersion/fixtures'
    commands = [
        [sys.executable, str(BASE/'tests/test_portable_runtime.py')],
        ['ctest', '--test-dir', str(ROOT/'build-portable'), '--output-on-failure'],
        ['ctest', '--test-dir', str(BASE/'native_dispersion/build'), '--output-on-failure'],
        [sys.executable, str(BASE/'tests/test_hexane_md.py')],
        [sys.executable, str(BASE/'tests/test_hexane_md_pme.py')],
    ]
    for platform in ('Reference', 'CUDA'):
        commands.append([sys.executable, str(BASE/'native_dispersion/test_native.py'),
                         str(fixtures/'vapor_input.npz'), str(fixtures/'vapor_dmff.npz'),
                         str(output/(platform+'.json')), '--platform', platform])
    commands.append([sys.executable, str(BASE/'native_dispersion/smoke_hybrid.py'),
                     str(BASE/'hex_vap_430/vap_conn.pdb'), '--minimize', '10', '--steps', '20'])
    records = []
    for index, command in enumerate(commands):
        log = output/f'{index:02d}.log'
        print('Running:', ' '.join(command), '\nLog:', log, flush=True)
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Validation failed ({result.returncode}); inspect {log}. MD remains disabled.')
        records.append(dict(command=command, log=str(log), sha256=sha256(log)))
    # Verify that no source or library changed during the tests.
    config = load_manifest()
    config['gpu_validation'] = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
                                    precision='double', commands=records)
    MANIFEST.write_text(json.dumps(config, indent=2)+'\n')
    print('PASS: local build validated; MD runners enabled. Reports:', output)


if __name__ == '__main__':
    main()
