import csv
print('Success')

with open('usunne-samtaler/data/norwegian.val.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)