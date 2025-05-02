from gwdistributions import parse, generators
POP_FOLDER="nested_folder"
EG = generators.EventGenerator(*parse.parse_config(f"{POP_FOLDER}/dist.ini", verbose=True))
print(EG)

# Problem is with the recursion step in parse_mass_pairing_section not accessing the folder