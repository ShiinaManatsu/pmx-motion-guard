"""Run with Blender --background full-test.blend --python this_file."""
import importlib
import json
import os
from pathlib import Path
import sys
import types

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ['RETARGET_GUARD_CACHE'] = str(ROOT / 'native')
package = types.ModuleType('motion_guard_sequence_test')
package.__path__ = [str(ROOT)]
sys.modules[package.__name__] = package
capture = importlib.import_module(package.__name__ + '.sequence_capture')
mapping_module = importlib.import_module(package.__name__ + '.mapping')
solver = importlib.import_module(package.__name__ + '.sequence_solver')
obj = next(o for o in bpy.context.scene.objects if o.type == 'MESH'
           and o.find_armature() and not o.get('pmx_guard_result') and len(o.data.vertices) > 1000)
mapping = mapping_module.automatic(obj.find_armature())
mapping_module.validate(mapping, obj.find_armature())
slots = list(range(len(obj.material_slots)))
arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
start, end = map(int, arguments[:2]) if len(arguments) >= 2 else (100, 159)
character, poses, positions, report = capture.capture_sequence(obj, mapping, slots, start, end)
character.save(ROOT / 'test_outputs' / 'sequence_character.npz')
np.savez_compressed(ROOT / 'test_outputs' / 'sequence_input.npz', poses=poses, positions=positions,
                    start=start, end=end)
report['source'] = obj.name
baseline, report['continuous'] = solver.solve_sequence(character, poses, screen=False)
result, report['segmented'] = solver.solve_sequence(character, poses, screen=True, workers=4)
report['max_matrix_difference'] = float(np.max(np.abs(result-baseline)))
destination = ROOT / 'test_outputs' / 'pmx_sequence_benchmark.json'
destination.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
