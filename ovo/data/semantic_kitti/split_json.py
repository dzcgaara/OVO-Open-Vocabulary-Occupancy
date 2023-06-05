import json
import os


chunk_size = 500
json_file = ''  # path to large json file
save_dir = 'small_files_occ'


with open(json_file, 'r') as f:
    data = json.load(f)
print("length", len(data.keys()))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

count = 0

temp_list = []
small_json = {}

for item in data:
    small_json[item] = data[item]
    temp_list.append(item)

    if len(temp_list) == chunk_size:
        filename = save_dir + f'/small_file_{count}.json'

        with open(filename, 'w') as f:
            json.dump(small_json, f)

        temp_list = []

        count += 1
        small_json = {}

if temp_list:
    filename = save_dir + f'/small_file_{count}.json'

    with open(filename, 'w') as f:
        json.dump(small_json, f)
