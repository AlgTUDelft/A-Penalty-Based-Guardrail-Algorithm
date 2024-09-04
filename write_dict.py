import csv

# Example dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

# Specify the filename
filename = 'output.csv'

# Write to CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data.keys())

    # Write the header (keys)
    writer.writeheader()

    # Write the rows (zipping the values together)
    for row in zip(*data.values()):
        writer.writerow(dict(zip(data.keys(), row)))

print(f"Data saved to {filename}")