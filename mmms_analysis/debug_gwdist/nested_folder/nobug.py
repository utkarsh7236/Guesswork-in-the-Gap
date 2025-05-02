from gwdistributions import parse, generators
# POP_FOLDER="conversion_scripts/pdbNG_betaSplit_brokenG"
EG = generators.EventGenerator(*parse.parse_config(f"dist.ini", verbose=True))
print(EG)