import pickle

with open('data/elan/citystreet_sunny_day_2025-09-25-15-38-56/3dbox_result.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
if isinstance(data, list):
    print(f"Length: {len(data)}")
    print(f"First item: {data[0]}")
elif isinstance(data, dict):
    print(f"Keys: {list(data.keys())[:5]}")
    first_key = list(data.keys())[0]
    print(f"First item (key '{first_key}'): {data[first_key]}")
else:
    print(data)
