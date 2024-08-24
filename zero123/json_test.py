import json

# Path to your JSON file
json_file = '/export/compvis-nfs/user/rbarlog/zero123/zero123/valid_paths.json'

# Read and print the contents of the JSON file
with open(json_file, 'r') as f:
    data_info = json.load(f)

print("Contents of the JSON file:")
print(data_info)