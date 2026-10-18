import json

if __name__ == '__main__':
    with open('../data/embedded_data_2.json', 'r') as f:
        data = json.load(f)
    
    for i, chunk in enumerate(data):
        print(chunk['metadata']['pmcid'])