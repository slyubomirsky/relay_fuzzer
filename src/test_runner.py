import sys
import pickle

import tvm
from tvm import relay

target_file = sys.argv[1]

with open(target_file, mode='rb') as f:
    pickled_data = f.read()
    irmodule = pickle.loads(pickled_data)
    
    # do something with irmodule...
    print(len(irmodule["main"].astext(show_meta_data=True)))
