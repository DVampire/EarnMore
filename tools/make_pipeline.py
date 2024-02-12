import warnings
warnings.filterwarnings("ignore")
import os
import sys
from pathlib import Path
from glob import glob

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT)

def main():
    paths = glob(os.path.join(ROOT, "scripts", "*", "*.sh"))

    with open(os.path.join(ROOT, "tools", "pipeline.sh"), "w") as f:
        f.write("#!/bin/bash\n")
        for path in paths:

            path = path.replace(ROOT, ".")

            f.write(f"bash {path} > /dev/null 2>&1 &\n")

if __name__ == '__main__':
    main()