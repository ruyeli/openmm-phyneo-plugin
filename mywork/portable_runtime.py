"""Local build provenance, never machine-specific binaries committed to Git."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
MANIFEST = BASE/'build_manifest.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT/'CMakeLists.txt', ROOT/'python/setup.py', ROOT/'python/CMakeLists.txt',
             BASE/'phyneo_hexane_opls_md.xml', BASE/'native_dispersion/CMakeLists.txt',
             ROOT/'scripts/build_portable.sh', BASE/'native_dispersion/build.sh']
    for directory in ['openmmapi', 'serialization', 'platforms', 'python', 'mywork/native_dispersion/src']:
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and
                     (p.suffix in ('.h', '.cuh', '.cpp', '.cu', '.i', '.in', '.cmake') or p.name == 'CMakeLists.txt'))
    paths.append(BASE/'native_dispersion/vendor/pocketfft_hdronly.h')
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


def load_manifest(require_validated=False):
    if not MANIFEST.is_file():
        raise RuntimeError('Build manifest missing or changed. Run bash scripts/build_portable.sh in the active environment.')
    data = json.loads(MANIFEST.read_text())
    if Path(data['python_prefix']).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError('Build belongs to another Python environment; rebuild here')
    if data['source_sha256'] != source_hashes():
        raise RuntimeError('Plugin source/model changed since build; rebuild and revalidate')
    for item in data['files'].values():
        p = Path(item['path'])
        if not p.is_file() or sha256(p) != item['sha256']:
            raise RuntimeError(f'Built library/model missing or changed: {p}')
    if require_validated and not data.get('gpu_validation'):
        raise RuntimeError('Local build needs GPU validation: python mywork/validate_build.py')
    return data


def record_build():
    prefix = Path(sys.prefix).resolve()
    wrapper = importlib.util.find_spec('_phyneoplugin')
    if wrapper is None:
        raise RuntimeError('Python wrapper was not installed')
    files = dict(core=prefix/'lib/libPhyNEOPlugin.so', cuda=prefix/'lib/plugins/libPhyNEOPluginCUDA.so',
                 dispersion=BASE/'native_dispersion/build/libNativeDispersion.so',
                 xml=BASE/'phyneo_hexane_opls_md.xml', wrapper=Path(wrapper.origin))
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = (BASE/'native_dispersion/build/CMakeCache.txt').read_text()
    for option in ('NATIVE_DISPERSION_ALL_PAIRS', 'NATIVE_DISPERSION_COMPLEX_FFT'):
        if f'{option}:BOOL=OFF' not in cache.splitlines():
            raise RuntimeError(f'{option} must be OFF for this optimized MD workflow')
    # Hashing a successful build is provenance, NOT scientific/GPU validation.
    data = dict(python_prefix=str(prefix), source_sha256=source_hashes(), gpu_validation=None,
                implementation='real-pme-fft-and-cell-verlet-v1',
                files={k: dict(path=str(p.resolve()), sha256=sha256(p)) for k, p in files.items()})
    MANIFEST.write_text(json.dumps(data, indent=2)+'\n')
    print('Recorded local build. Run python mywork/validate_build.py on a GPU node before MD.')


if __name__ == '__main__':
    record_build()
