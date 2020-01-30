import json 
import csv
import pathlib


def files_with_extension(path=".", pattern="*.json"):
    """ returns file list that satisfy pattern """
    # define the path
    path = pathlib.Path(path)
    for file in sorted(path.glob(pattern)):
        yield file


def json_to_csv(out_fname="db_results_combined.csv", 
                path=".", 
                pattern="*.json"):
    """ convert results given in json format to csv format """
    write_header = True
    with open(out_fname, 'a') as csvfile:
        for file in files_with_extension(path, pattern):
            with open(file.name, 'r') as json_file:
                for line in json_file:
                    cases = json.loads(line)
                    for case in cases:
                        writer = csv.DictWriter(csvfile, fieldnames=case.keys())
                        if write_header:
                            writer.writeheader()
                            write_header = False                        
                        # write dictionary to file
                        writer.writerow(case)

def combine_jsons(out_fname="db_results_combined.json", path=".", pattern="*.json"):
    # combine json files in current folder
    with open(out_fname, "w") as outfile:
        outfile.write('[{}]'.format(','.join([open(f, "r").read() for f in files_with_extension(path, pattern)])))

# json to csv
json_to_csv()
combine_jsons()