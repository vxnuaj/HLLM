'''
NOTE
Will be a script to narrow down on the possible selection of configs to be the fastest, given the set of json files.

So, I want to get the best times for all sweeps across attn_types (and of course when using gqa would liek to get the best across all n_groups)

'''

import json
import os

# returns a list of sub-directories under 'conf_prof/results' in sorted format.
list_dirs = sorted(os.listdir('conf_prof/results')) 

results = []

for i in list_dirs: # gets the results in .json format for all runs in the sweep
    with open(i, 'r') as f: 
        result = json.load(f)
    results.append(result)