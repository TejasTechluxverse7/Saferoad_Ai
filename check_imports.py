# -*- coding: utf-8 -*-
"""Quick import check for AegisRoad AI modules."""
import sys
sys.path.insert(0, '.')

errors = []
mods = [
    'tracking.bytetrack_wrapper',
    'prediction.trajectory_engine',
    'engine.severity_classifier',
    'engine.evidence_buffer',
    'engine.victim_detector',
    'engine.forensics_engine',
    'engine.gradcam',
    'fusion.reid_engine',
    'api.routing',
    'api.saferoad_api',
]
for m in mods:
    try:
        __import__(m)
        print('OK  ' + m)
    except Exception as e:
        print('ERR ' + m + ': ' + str(e))
        errors.append(m)

print()
print('Result: %d/%d modules OK' % (len(mods) - len(errors), len(mods)))
