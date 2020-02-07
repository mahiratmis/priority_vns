import json 
import csv
import pathlib

import pandas as pd 

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
            with open(file, 'r') as json_file:
                for line in json_file:
                    cases = json.loads(line)
                    for case in cases:
                        writer = csv.DictWriter(csvfile, fieldnames=case.keys())
                        if write_header:
                            writer.writeheader()
                            write_header = False                        
                        # write dictionary to file
                        writer.writerow(case)



def json_to_csv2(out_fname="db_results_combined.csv", 
                path=".", 
                pattern="*.json"):
    """ convert results given in json format to csv format """
    for file in files_with_extension(path, pattern):
        with open(file, 'r') as json_file:
            for line in json_file:
                cases = json.loads(line)
                cases_df = pd.DataFrame(cases)
                cases_df.to_csv(out_fname, index=False)                  


def combine_jsons(out_fname="db_results_combined.json", path=".", pattern="*.json"):
    # combine json files in current folder
    with open(out_fname, "w") as outfile:
        outfile.write('[{}]'.format(','.join([open(f, "r").read() for f in files_with_extension(path, pattern)])))

# json to csv
lamda = 1 #test 0, 0.5, 1
var_level = 0.05

#
db_nodb = "nodb"
pth = "benchmark"
o_dir_json = pth + "/combined/json/combined_vns_{}_v{}_l{}_priority_simopt_riskaverse".format(db_nodb, int(var_level*100), int(lamda*100))
o_dir_csv = pth + "/combined/csv/combined_vns_{}_v{}_l{}_priority_simopt_riskaverse".format(db_nodb, int(var_level*100), int(lamda*100))
pttern = "*_{}_v{}_l{}*.json".format(db_nodb, int(var_level*100), int(lamda*100))
combine_jsons(path=pth, out_fname=o_dir_json+".json", pattern=pttern)
json_to_csv2(path=pth, out_fname=o_dir_csv+".csv", pattern=pttern)