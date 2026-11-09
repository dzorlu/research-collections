"""Generate the drill stubs in distributed/ from the reference solutions in distributed/reference/.

Every block between `# >>> TODO` and `# <<< TODO` becomes `# TODO` + `raise NotImplementedError`, and the
reference-only sys.path line is dropped (the stubs sit next to _harness.py).

Run: python reference/_make_stubs.py
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)

for path in sorted(glob.glob(os.path.join(HERE, "day*.py"))):
    src = open(path).read()
    src = re.sub(r"^sys\.path\.insert.*# reference only\n", "", src, flags=re.M)
    src = re.sub(r"^import sys\n", "", src, flags=re.M)
    src = re.sub(r"^import os\n", "", src, flags=re.M) if "os." not in src.replace("import os", "") else src
    src = re.sub(r"^( *)# >>> TODO\n.*?^ *# <<< TODO\n",
                 lambda m: f"{m.group(1)}# TODO\n{m.group(1)}raise NotImplementedError\n",
                 src, flags=re.M | re.S)
    dst = os.path.join(OUT, os.path.basename(path))
    open(dst, "w").write(src)
    print("wrote", os.path.relpath(dst, OUT))
