import json 
import csv
import pathlib

def files_with_extension(path=".", pattern="*.json"):
    # define the path
    path = pathlib.Path(path)
    for file in sorted(path.glob(pattern)):
        yield file

write_header = True

with open(f'results_combined.csv', 'a') as csvfile:
    for file in files_with_extension():
        neighbors_structures = "_"+file.name.split('_')[0]
        print(neighbors_structures)
        with open(file.name, 'r') as json_file:
            for line in json_file:
                cases = json.loads(line)
                for case in cases:
                    case['neighbors_structures'] = neighbors_structures
                    writer = csv.DictWriter(csvfile, fieldnames=case.keys())
                    if write_header:
                        writer.writeheader()
                        write_header = False                        
                    # write dictionary to file
                    writer.writerow(case)