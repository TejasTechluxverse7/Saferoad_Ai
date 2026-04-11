# -*- coding: utf-8 -*-
import ast

files = ['saferoad_main.py', 'model_train.py']
ok = 0
for f in files:
    try:
        with open(f, encoding='utf-8') as fh:
            ast.parse(fh.read())
        print('OK  ' + f)
        ok += 1
    except SyntaxError as e:
        print('ERR ' + f + ': ' + str(e))

print()
print('Syntax: %d/%d OK' % (ok, len(files)))
