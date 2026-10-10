import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
for d in ('src', 'tools'):
    sys.path.insert(0, os.path.join(here, '..', d))
