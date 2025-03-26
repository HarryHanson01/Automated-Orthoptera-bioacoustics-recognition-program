import os

csv_files = [f for f in os.listdir() if f.startswith('GCDetections_') and f.endswith('.csv')]

combined_data = []

header_saved = False
for file in csv_files:
    with open(file, 'r') as f:
        lines = f.readlines()
        
        if not header_saved:
            combined_data.append(lines[0])
            header_saved = True
            
        combined_data.extend(lines[1:])


with open('Combined_GCDetections.csv', 'w') as f:
    f.writelines(combined_data)

print(f"Combined {len(csv_files)} files into Combined_GCDetections.csv")
