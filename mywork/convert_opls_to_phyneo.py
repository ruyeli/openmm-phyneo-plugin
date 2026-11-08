"""Convert connected n-hexane PDBs to this project's PhyNEO TE1/IN1 naming.

Usage: python convert_opls_to_phyneo.py input.pdb [output.pdb]
Requires explicit CONECT bonds and all hydrogens. Standard library only.
Coordinates and CRYST1 are preserved; an atom mapping CSV is also written.
This converts PDB naming/layout, NOT force-field parameters or charges.
"""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from io import StringIO
import math
from pathlib import Path
import re
import string


@dataclass(frozen=True)
class Atom:
    serial: int
    element: str
    line: str


def parse_pdb(text):
    atoms, edges, cryst1 = {}, set(), []
    model_count = 0
    for number, raw in enumerate(text.splitlines(), 1):
        record = raw[:6].strip()
        if record == 'MODEL':
            model_count += 1
            if model_count > 1:
                raise ValueError('Multiple MODEL frames are unsupported; provide one configuration')
        elif record == 'CRYST1':
            cryst1.append(raw)
            if len(cryst1) > 1:
                raise ValueError('Multiple CRYST1 records are ambiguous')
        elif record in ('ATOM', 'HETATM'):
            if len(raw) < 54:
                raise ValueError(f'Line {number}: truncated atom/coordinate record')
            try:
                serial = int(raw[6:11])
                coordinates = [float(raw[i:i+8]) for i in (30, 38, 46)]
            except ValueError as exc:
                raise ValueError(f'Line {number}: invalid standard fixed-width PDB atom record') from exc
            if not all(math.isfinite(v) for v in coordinates):
                raise ValueError(f'Line {number}: nonfinite coordinates')
            if serial <= 0 or serial in atoms:
                raise ValueError(f'Line {number}: nonpositive or duplicate atom serial {serial}')
            if raw[16:17].strip():
                raise ValueError(f'Line {number}: alternate locations unsupported; select one conformer first')
            element = raw[76:78].strip().upper()
            if not element:
                match = re.fullmatch(r'([CH])(?:[0-9].*)?', raw[12:16].strip().upper())
                element = match.group(1) if match else ''
            if element not in ('C', 'H'):
                raise ValueError(f'Line {number}: requires C/H atoms with unambiguous element fields')
            atoms[serial] = Atom(serial, element, raw.ljust(80))
        elif record == 'CONECT':
            # Do NOT split on whitespace: e.g. CONECT 9986 9985 9998 999910000.
            fields = [raw[i:i+5].strip() for i in range(6, len(raw), 5)]
            try:
                values = [int(field) for field in fields if field]
            except ValueError as exc:
                raise ValueError(f'Line {number}: invalid fixed-width CONECT record') from exc
            if not values:
                raise ValueError(f'Line {number}: empty CONECT record')
            source, *partners = values
            if len(set(partners)) != len(partners):
                raise ValueError(f'Line {number}: repeated CONECT partner/bond order unsupported for hexane')
            for target in partners:
                if source == target:
                    raise ValueError(f'Line {number}: self bond at atom {source}')
                edges.add(tuple(sorted((source, target))))
    if not atoms or not edges:
        raise ValueError('PDB must contain atoms and explicit CONECT bonds')
    if len(atoms) > 99999:
        raise ValueError('More than 99999 atoms exceed the supported standard PDB serial range')
    neighbors = {serial: set() for serial in atoms}
    for a, b in edges:
        if a not in atoms or b not in atoms:
            raise ValueError(f'CONECT refers to missing atom in bond ({a}, {b})')
        neighbors[a].add(b)
        neighbors[b].add(a)
    return atoms, neighbors, edges, cryst1


def ordered_hexanes(atoms, neighbors):
    """Find molecules from connectivity, independent of residue/atom names."""
    remaining = set(atoms)
    molecules = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        component, stack = {first}, [first]
        while stack:
            current = stack.pop()
            new = neighbors[current] & remaining
            remaining.difference_update(new)
            component.update(new)
            stack.extend(new)
        composition = Counter(atoms[i].element for i in component)
        if composition != Counter(C=6, H=14):
            raise ValueError(f'Component containing atom {first} is {dict(composition)}, not C6H14; check connectivity')
        carbons = {i for i in component if atoms[i].element == 'C'}
        carbon_neighbors = {i: neighbors[i] & carbons for i in carbons}
        for i in component:
            if atoms[i].element == 'C':
                if len(neighbors[i]) != 4:
                    raise ValueError(f'Carbon {i} must have four explicitly connected neighbors')
            elif len(neighbors[i]) != 1 or not neighbors[i] <= carbons:
                raise ValueError(f'Hydrogen {i} must have exactly one carbon neighbor')
        if sorted(len(v) for v in carbon_neighbors.values()) != [1, 1, 2, 2, 2, 2]:
            raise ValueError(f'Component at atom {first} is not linear n-hexane')
        # Deterministic orientation: start at the lower-serial terminal carbon.
        current = min(i for i in carbons if len(carbon_neighbors[i]) == 1)
        chain = []
        while current is not None:
            chain.append(current)
            candidates = carbon_neighbors[current] - set(chain)
            if len(candidates) > 1:
                raise ValueError('Branched carbon chain')
            current = next(iter(candidates)) if candidates else None
        if len(chain) != 6:
            raise ValueError('Carbon connectivity is not one six-carbon path')
        molecules.append(chain)
    return molecules


def convert_text(text):
    atoms, neighbors, edges, cryst1 = parse_pdb(text)
    molecules = ordered_hexanes(atoms, neighbors)
    lines = ['REMARK   Converted n-hexane connectivity to PhyNEO TE1/IN1 residues.',
             'REMARK   Coordinates unchanged; bonds and atom serials remapped.', *cryst1]
    mapping, new_serials = [], {}
    chain_ids = string.ascii_uppercase + string.ascii_lowercase + string.digits
    chain_index, residue_number = 0, 0
    for molecule_number, carbons in enumerate(molecules, 1):
        # Keep each molecule within one chain and avoid overflowing resSeq.
        if residue_number+6 > 9999:
            chain_index += 1
            residue_number = 0
        if chain_index >= len(chain_ids):
            raise ValueError('Too many chains for standard PDB output')
        chain_id = chain_ids[chain_index]
        for position, carbon in enumerate(carbons, 1):
            residue_number += 1
            residue = 'TE1' if position in (1, 6) else 'IN1'
            hydrogens = sorted(neighbors[carbon] - set(carbons))
            expected = 3 if residue == 'TE1' else 2
            if len(hydrogens) != expected:
                raise ValueError(f'Carbon {carbon} has incorrect hydrogen count')
            for hindex, old_serial in enumerate([carbon, *hydrogens]):
                atom = atoms[old_serial]
                new_serial = len(mapping)+1
                atom_name = 'C00' if hindex == 0 else f'H{hindex:02d}'
                new_serials[old_serial] = new_serial
                prefix = f'ATOM  {new_serial:5d} {atom_name:>4s} {residue:3s} {chain_id}{residue_number:4d}    '
                assert len(prefix) == 30
                # Coordinates, occupancy, B-factor and segment fields stay as
                # their original text. Fill/normalize only the element field.
                lines.append(prefix + atom.line[30:76] + f'{atom.element:>2s}' + atom.line[78:])
                mapping.append(dict(
                    old_serial=old_serial, old_atom_name=atom.line[12:16].strip(),
                    old_residue_name=atom.line[17:20].strip(), old_chain=atom.line[21:22].strip(),
                    old_residue_id=atom.line[22:27].strip(),
                    new_serial=new_serial, new_atom_name=atom_name, new_residue_name=residue,
                    new_chain=chain_id, new_residue_id=residue_number,
                    molecule=molecule_number, carbon_position=position,
                    phyneo_type=('8' if hindex == 0 else '9') if residue == 'TE1'
                    else ('6' if hindex == 0 else '7')))
    # Write reciprocal connectivity, using remapped serials (4 targets max).
    for old_serial in sorted(new_serials, key=new_serials.get):
        targets = sorted(new_serials[j] for j in neighbors[old_serial])
        lines.append('CONECT'+f'{new_serials[old_serial]:5d}'+''.join(f'{j:5d}' for j in targets))
    lines.append('END')
    summary = dict(molecules=len(molecules), atoms=len(atoms), bonds=len(edges),
                   residues=6*len(molecules), terminal_residues=2*len(molecules),
                   internal_residues=4*len(molecules), periodic_box_preserved=bool(cryst1))
    return '\n'.join(lines)+'\n', mapping, summary


def convert_file(input_path, output_path=None):
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(input_path.stem+'_phyneo.pdb')
    map_path = output_path.with_suffix('.atom_map.csv')
    if input_path.resolve() in (output_path.resolve(), map_path.resolve()):
        raise ValueError('Output must not overwrite the input PDB')
    for path in (output_path, map_path):
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {path}')
    converted, mapping, summary = convert_text(input_path.read_text())
    csv_text = StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=list(mapping[0]))
    writer.writeheader()
    writer.writerows(mapping)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in ((output_path, converted), (map_path, csv_text.getvalue())):
            with path.open('x', newline='') as handle:
                created.append(path)
                handle.write(content)
    except Exception:
        # Remove only files created by this invocation, never existing files.
        for path in created:
            path.unlink()
        raise
    return output_path, map_path, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', nargs='?', type=Path,
                        help='Default: <input_stem>_phyneo.pdb; refuses to overwrite')
    args = parser.parse_args()
    try:
        output, mapping, counts = convert_file(args.input, args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Conversion failed: {exc}\n')
    print(f'Converted {counts["molecules"]} n-hexane molecules: '
          f'{counts["atoms"]} atoms, {counts["bonds"]} bonds, {counts["residues"]} residues.')
    print(f'PDB: {output}\nAtom mapping: {mapping}')
    if not counts['periodic_box_preserved']:
        print('Warning: input has no CRYST1. Supply a periodic box before using the MD runner.')


if __name__ == '__main__':
    main()
