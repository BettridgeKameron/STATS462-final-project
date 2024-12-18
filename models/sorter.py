import json
import sys
from collections import OrderedDict

def sort_transitions(data):
    sorted_outer = OrderedDict(sorted(data['transitions'].items()))
    for char in sorted_outer:
        sorted_inner = OrderedDict(sorted(sorted_outer[char].items(), key=lambda x: x[1], reverse=True))
        sorted_outer[char] = sorted_inner
    return {'transitions': dict(sorted_outer), 'total_transitions': data['total_transitions']}

input_file = sys.argv[1]
output_file = input_file.rsplit('.', 1)[0] + '_output.json'

with open(input_file, 'r') as f:
    data = json.load(f)

sorted_data = sort_transitions(data)

with open(output_file, 'w') as f:
    json.dump(sorted_data, f, indent=2)
