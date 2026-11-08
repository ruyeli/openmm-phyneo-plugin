"""Build the hexane OPLS-bonded/PhyNEO-nonbonded force field (stdlib only)."""

import argparse
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET


# The current plugin parser sets only Covalent12/13/14.  Its native API has
# Covalent15 but no Covalent16.  For THIS model, both have identical m/p/d
# scales, so their union can safely occupy Covalent15.  Rebuild a typed force
# because OpenMM returns third-party forces as base-class Python proxies.
EXCLUSION_SCRIPT = '''
def _hexane_complete_covalent_maps(system, data, args):
    import xml.etree.ElementTree as ET
    import openmm
    import phyneoplugin

    candidates = [(i, f) for i, f in enumerate(system.getForces())
                  if f.getName() == "PhyNEOForce"]
    if len(candidates) != 1:
        raise ValueError("Hexane XML requires exactly one PhyNEOForce")
    index, original = candidates[0]
    root = ET.fromstring(openmm.XmlSerializer.serialize(original))
    force = phyneoplugin.PhyNEOForce()
    force.setName(original.getName())
    force.setForceGroup(original.getForceGroup())
    for name, conversion in (
        ("NonbondedMethod", int), ("PolarizationType", int),
        ("MutualInducedMaxIterations", int), ("CutoffDistance", float),
        ("MutualInducedTargetEpsilon", float), ("EwaldErrorTolerance", float)
    ):
        getattr(force, "set" + name)(conversion(root.get(name[0].lower() + name[1:])))
    grid = root.find("MultipoleParticleGridDimension")
    force.setPMEParameters(float(root.get("aEwald")),
                           *[int(grid.get("d" + str(i))) for i in range(3)])
    coefficients = root.find("ExtrapolationCoefficients")
    force.setExtrapolationCoefficients(
        [float(coefficients.get("c" + str(i))) for i in range(len(coefficients.attrib))])
    # Serialization currently omits these settings. Restore them explicitly.
    force.setDefaultTholeWidth(float(args.get("defaultTholeWidth", 5.0)))
    force.setMultipoleScaleFactors([1, 0, 0, 0, 0, 0],
                                   [1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1])
    for particle in root.find("MultipoleParticles"):
        def vector(tag, keys):
            node = particle.find(tag)
            return [float(node.get(key)) for key in keys.split()]
        force.addMultipole(
            float(particle.get("charge")), vector("Dipole", "dX dY dZ"),
            vector("Quadrupole", "qXX qXY qYY qXZ qYZ qZZ"),
            vector("Octopole", "oXXX oXXY oXYY oYYY oXXZ oXYZ oYYZ oXZZ oYZZ oZZZ"),
            *[int(particle.get(key)) for key in
              ("axisType", "multipoleAtomZ", "multipoleAtomX", "multipoleAtomY")],
            float(particle.get("thole")),
            [float(particle.get("polarizability" + axis)) for axis in ("XX", "YY", "ZZ")])
    for i in range(system.getNumParticles()):
        seen, frontier, shells = {i}, {i}, []
        for depth in range(5):
            frontier = {j for k in frontier for j in data.bondedToAtom[k]} - seen
            shells.append(set(frontier))
            seen.update(frontier)
        for kind, neighbors in enumerate(shells[:3] + [shells[3] | shells[4]]):
            force.setCovalentMap(i, kind, sorted(neighbors))
    system.removeForce(index)
    system.addForce(force)
    for other in system.getForces():
        if isinstance(other, openmm.CustomNonbondedForce):
            other.setUseLongRangeCorrection(False)
            other.setUseSwitchingFunction(False)

_hexane_complete_covalent_maps(sys, data, args)
'''


def bonded_blocks(opls):
    """Collapse LigParGen's per-atom classes, refusing conflicting parameters."""
    mapping = {t.get('class'): {'C': 'CT', 'H': 'HC'}[t.get('element')]
               for t in opls.find('AtomTypes')}
    blocks = []
    for tag, child, count in [('HarmonicBondForce', 'Bond', 2),
                              ('HarmonicAngleForce', 'Angle', 3),
                              ('PeriodicTorsionForce', 'Proper', 4)]:
        source = opls.find(tag)
        if any(term.tag != child for term in source):
            raise ValueError(f'Unsupported term in {tag}')
        terms = {}
        for term in source:
            pattern = tuple(mapping[term.get(f'class{i}')] for i in range(1, count + 1))
            pattern = min(pattern, pattern[::-1])
            params = {k: v for k, v in term.attrib.items() if not k.startswith('class')}
            if pattern in terms and terms[pattern] != params:
                raise ValueError(f'Cannot generalize conflicting {tag} parameters: {pattern}')
            terms[pattern] = params
        block = ET.Element(tag)
        for pattern, params in sorted(terms.items()):
            ET.SubElement(block, child,
                          {**{f'class{i}': c for i, c in enumerate(pattern, 1)}, **params})
        blocks.append(block)
    return blocks


def build(opls_path, phyneo_path):
    opls, source = ET.parse(opls_path).getroot(), ET.parse(phyneo_path).getroot()
    result = ET.Element('ForceField')
    result.append(ET.Comment(
        ' Hexane hybrid: OPLS bonded terms from npt_ts_1fs/LIG.openmm.xml; '
        'physical nonbonded terms from npt_ts_1fs/output.xml. '
        'Electrostatic PME; dispersion truncated at nonbondedCutoff, no tail correction. '
        'No ML, OPLS charges, or Lennard-Jones. Import phyneoplugin before loading. '
        'Generated by build_hexane_opls_phyneo.py. '))
    for tag in ('AtomTypes', 'Residues'):
        result.append(copy.deepcopy(source.find(tag)))
    result.extend(bonded_blocks(opls))
    multipoles = copy.deepcopy(source.find('ADMPPmeForce'))
    for family, expected in [('m', 0), ('p', 0), ('d', 1)]:
        assert all(float(multipoles.get(f'{family}Scale{i}')) == expected for i in range(12, 17))
    result.append(multipoles)

    amplitudes = {'Aexch': 'SlaterExForce', 'Aelec': 'SlaterSrEsForce',
                  'Aind': 'SlaterSrPolForce', 'Adhf': 'SlaterDhfForce',
                  'Adisp': 'SlaterSrDispForce'}
    tables = {tag: {a.get('type'): a.attrib for a in source.find(tag)} for tag in
              [*amplitudes.values(), 'QqTtDampingForce', 'SlaterDampingForce', 'ADMPDispPmeForce']}
    for tag in tables:
        assert all(float(source.find(tag).get(f'mScale{i}')) == 0 for i in range(12, 17))
    expression = ('A*(1+Br+Br^2/3)*exp(-Br)'
                  '-138.935455846*Q1*Q2*exp(-Br)*(1+Br)/r')
    for n in (6, 8, 10):
        series = '+'.join(f'x^{k}/{math.factorial(k)}' for k in range(n + 1))
        expression += f'-sqrt(C{n}1*C{n}2)*(1-exp(-x)*({series}))/r^{n}'
    expression += (';A=Aexch1*Aexch2-Aelec1*Aelec2-Aind1*Aind2-Adhf1*Adhf2-Adisp1*Adisp2'
                   ';x=Br-(2*Br^2+3*Br)/(Br^2+3*Br+3);Br=sqrt(Bexp1*Bexp2)*r')
    custom = ET.SubElement(result, 'CustomNonbondedForce', bondCutoff='5', energy=expression)
    for name in [*amplitudes, 'Bexp', 'Q', 'C6', 'C8', 'C10']:
        ET.SubElement(custom, 'PerParticleParameter', name=name)
    for atom_type in ('6', '7', '8', '9'):
        params = {name: tables[tag][atom_type]['A'] for name, tag in amplitudes.items()}
        b = tables['SlaterExForce'][atom_type]['B']
        assert all(float(table[atom_type]['B']) == float(b)
                   for table in tables.values() if 'B' in table[atom_type])
        params.update(Bexp=b, Q=tables['QqTtDampingForce'][atom_type]['Q'])
        for n in (6, 8, 10):
            key = f'C{n}'
            params[key] = tables['ADMPDispPmeForce'][atom_type][key]
            assert params[key] == tables['SlaterDampingForce'][atom_type][key]
        ET.SubElement(custom, 'Atom', type=atom_type, **params)
    ET.SubElement(result, 'Script').text = EXCLUSION_SCRIPT
    ET.indent(result, space='  ')
    return ET.ElementTree(result)


if __name__ == '__main__':
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opls', type=Path, default=base / 'npt_ts_1fs/LIG.openmm.xml')
    parser.add_argument('--phyneo', type=Path, default=base / 'npt_ts_1fs/output.xml')
    parser.add_argument('--output', type=Path, default=base / 'phyneo_hexane_opls_md.xml')
    args = parser.parse_args()
    build(args.opls, args.phyneo).write(args.output, encoding='utf-8', xml_declaration=True)
    print(f'Wrote {args.output}')
