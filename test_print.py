import csv
print('Success')

with open('/data/norwegian.val.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)