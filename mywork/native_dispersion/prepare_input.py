"""Prepare typed coordinates/pairs from original XML; no DMFF or GPU imports."""
import argparse
from pathlib import Path
import sys
import json
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK))
from convert_opls_to_phyneo import parse_pdb, ordered_hexanes


def prepare(pdb, xml, cutoff=.6):
    atoms, neighbors, edges, boxes = parse_pdb(pdb.read_text())
    ordered_hexanes(atoms,neighbors)
    ids = list(atoms)
    lookup = {serial:i for i,serial in enumerate(ids)}
    if len(boxes)!=1 or [float(boxes[0][k:k+7]) for k in (33,40,47)] != [90.,90.,90.]:
        raise ValueError('Orthorhombic periodic PDB required')
    lengths=np.array([float(boxes[0][k:k+9])*.1 for k in (6,15,24)])
    positions=np.array([[float(atoms[i].line[k:k+8])*.1 for k in (30,38,46)] for i in ids])
    root=ET.parse(xml).getroot()
    types={(res.get('name'),a.get('name')):a.get('type') for res in root.find('Residues') for a in res.findall('Atom')}
    node=root.find('ADMPDispPmeForce')
    params={a.get('type'):[float(a.get('C'+str(n))) for n in (6,8,10)] for a in node.findall('Atom')}
    coefficients=np.array([params[types[(atoms[i].line[17:20].strip(),atoms[i].line[12:16].strip())]] for i in ids])
    scales=np.array([float(node.get('mScale'+str(k))) for k in range(12,17)]+[1.])
    pairs=cKDTree(positions%lengths,boxsize=lengths).query_pairs(cutoff,output_type='ndarray')
    pairs=np.sort(pairs,axis=1);pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
    depths={}
    for serial in ids:
        seen,front={serial},{serial}
        for depth in range(1,6):
            front={j for i in front for j in neighbors[i]}-seen
            seen.update(front)
            for target in front:
                depths[(lookup[serial],lookup[target])]=depth
    pairs=np.column_stack((pairs,[depths.get(tuple(pair),0) for pair in pairs])).astype(np.int32)
    # All exceptions, including those outside the pair cutoff, for OpenMM raw-cutoff tests.
    exclusions=np.array([pair for pair,depth in depths.items() if pair[0]<pair[1] and scales[depth-1]==0],dtype=np.int32)
    if not np.array_equal(scales, [0,0,0,0,0,1]):
        raise ValueError('This hexane validation fixture expects the original zero 1-2 through 1-6 scales')
    return dict(positions=positions,box=np.diag(lengths),coefficients=coefficients,pairs=pairs,
                scales=scales,exclusions=exclusions,cutoff=np.array(cutoff))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pdb',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();xml=WORK/'npt_ts_1fs/output.xml'
    if a.output.exists():p.error('Output exists')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    data=prepare(a.pdb,xml)
    with a.output.open('xb') as handle:np.savez(handle,**data)
    print(json.dumps(dict(atoms=len(data['positions']),pairs=len(data['pairs']),
        pdb=str(a.pdb.resolve()),pdb_sha256=hashlib.sha256(a.pdb.read_bytes()).hexdigest(),
        xml_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),output=str(a.output))))
