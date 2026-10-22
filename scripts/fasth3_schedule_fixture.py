#!/usr/bin/env python3
"""Refresh FP32 scheduler fixtures from pinned ComfyUI source (requires torch)."""
import ast
import json
import struct
import urllib.request
from pathlib import Path
import torch

REV = '7a28332047b6af638dc6c7f72ef3481a718769fd'
ROOT = f'https://raw.githubusercontent.com/comfyanonymous/ComfyUI/{REV}/'

def definitions(path, names):
    source = urllib.request.urlopen(ROOT + path).read().decode()
    tree = ast.parse(source)
    return '\n'.join(ast.get_source_segment(source, node) for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)

scope = {'torch': torch}
exec(definitions('comfy/model_sampling.py', {'time_snr_shift', 'ModelSamplingDiscreteFlow'}), scope)
exec(definitions('comfy/samplers.py', {'simple_scheduler'}), scope)
exec(definitions('comfy/ldm/minimax/model.py', {'time_shift_sigma'}), scope)
bits = lambda tensor: [struct.unpack('<I', struct.pack('<f', x))[0] for x in tensor.tolist()]
cases = []
for shift in [10, 12]:
    model = scope['ModelSamplingDiscreteFlow']()
    model.set_parameters(shift=shift)
    for n in [1, 7, 8, 10, 12, 999]:
        video = scope['simple_scheduler'](model, n)
        audio = scope['time_shift_sigma'](video, shift, 3)
        cases.append(dict(evaluations=n, video_shift=shift, audio_shift=3,
                          video=bits(video), audio=bits(audio),
                          video_times=bits(1-video[:-1]), audio_times=bits(1-audio[:-1])))
Path('tests/fixtures/fasth3_schedule.json').write_text(json.dumps(dict(comfyui_revision=REV, cases=cases), indent=2)+'\n')
