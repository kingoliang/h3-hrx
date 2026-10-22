#!/usr/bin/env python3
"""Compare native kernel fixtures with pinned comfy-kitchen's VSA eager oracle.

H3_VSA_FIXTURE=target/vsa-fixture cargo test --release --lib vsa::device::tests
python scripts/vsa_reference.py target/vsa-fixture
"""
import ast
import json
import sys
import urllib.request
from pathlib import Path
import torch

REV='f8b7e97a1418d2e529bd5405fbf86312be198ae6'
u=f'https://raw.githubusercontent.com/Comfy-Org/comfy-kitchen/{REV}/comfy_kitchen/backends/eager/sol_attn.py'
source=urllib.request.urlopen(u).read().decode()
tree=ast.parse(source)
# Only the pure eager implementation; no package backend registration or custom ops.
nodes=[n for n in tree.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and not n.name.startswith('_op_')]
scope={'torch':torch}
exec(compile(ast.Module(body=nodes,type_ignores=[]),u,'exec'),scope)
root=Path(sys.argv[1]); layout=json.loads((root/'layout.json').read_text())
src=torch.tensor(layout['src']); inverse=torch.tensor(layout['inverse']);h=layout['heads'];t=len(inverse)
def read(name):
    return torch.frombuffer(bytearray((root/name).read_bytes()),dtype=torch.float16).view(1,t,h,128)
def padded(x):
    return torch.where((src>=0).view(1,-1,1,1),x[:,src.clamp_min(0)],0)
q,k,v,gate=[padded(read(name+'.f16')) for name in ['q','k','v','gate']]
for gated in [False,True]:
    expected=scope['sol_attn'](q,k,v,topk_ratio=layout['keep_percent']/100,
         sink_blocks=[0,layout['prefix']],sink_q=[0,layout['prefix']],
         tail=False,block_len=torch.tensor(layout['lengths']),coarse_gate=gate if gated else None)[:,inverse].float()
    for bits,limit in [(16,.01),(8,.03)]:
        actual=read(f'{"gated" if gated else "out"}-{bits}.f16').float()
        nrmse=((actual-expected).square().sum()/expected.square().sum()).sqrt().item()
        print(json.dumps(dict(revision=REV,gated=gated,bits=bits,normalized_rmse=nrmse)))
        assert nrmse<=limit
