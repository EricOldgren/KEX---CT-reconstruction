#!/usr/bin/env python
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib

backends = matplotlib.rcsetup.interactive_bk
# validate backends
backends_valid = []
for b in backends:
    try:
        plt.switch_backend(b)
        backends_valid += [b]
    except:
        continue
print(backends_valid)