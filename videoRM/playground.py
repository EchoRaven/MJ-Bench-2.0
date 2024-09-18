import json

# Load the JSON data from the uploaded file
file_path = 'MJ-Bench-Trans-Info.json'

# Open and read the file
with open(file_path, 'r') as f:
    data = json.load(f)

# Process the data to match the required format
results = []

# Group the data by the numeric part of the id (ignoring the source part)
grouped_data = {}
for item in data:
    # Extract the group id, e.g., from alignment_image0_0 to 0_0
    group_id = item["id"].split('_')[2]
    if group_id not in grouped_data:
        grouped_data[group_id] = []
    grouped_data[group_id].append(item)

# For each group, assign the positive and negative video paths
for group_id, items in grouped_data.items():
    if len(items) == 2:
        # Based on the label, determine which is positive and which is negative
        if items[0]['label'] == 1:
            positive_video = items[1]['video_path']
            negative_video = items[0]['video_path']
        else:
            positive_video = items[0]['video_path']
            negative_video = items[1]['video_path']

        # Create the transformed entry
        results.append({
            "id": f"request-{group_id.replace('_', '-')}",
            "source": f"MJ-Bench/{items[0]['source'].split('/')[0]}",  # Assume source is the same, take from the first item
            "positive_video_path": positive_video,
            "negative_video_path": negative_video
        })

# Save the transformed data back to a JSON file
output_file_path = 'transformed_MJ_Bench_Info.json'
with open(output_file_path, 'w') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

