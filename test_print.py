import csv
print('Success')

with open('/data/val.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)