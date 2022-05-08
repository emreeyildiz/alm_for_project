import json

print("Started Reading JSON file which contains multiple JSON document")

commits = {}
parents = {}

with open('storybookresult.json') as f:
    for (i, jsonObj) in enumerate(f):
        info = json.loads(jsonObj)
        commits[info["data"]["commit"]] = info
        parents[info["data"]["commit"]] = [commits[parent] for parent in info["data"]["parents"]]

a = parents["d9f1e643a6d867c8b0350b88e44de0dbb078f445"]

i = 0
while len(a[0]["data"]["parents"]) > 0:
    id = a[0]["data"]["parents"][0]
    a = parents[id]
    print("iteration", i, id)
    i += 1

print(a)
