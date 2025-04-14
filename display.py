
import json
with open('/Users/fanninghan/code/adhd/Eye/files/result.json', 'r') as file:
    data = json.load(file)
data = data[0]
data = data['result']
for idx, item in enumerate(data):
    print(f"{idx}: {item}")
    if idx > 500:
        break
    