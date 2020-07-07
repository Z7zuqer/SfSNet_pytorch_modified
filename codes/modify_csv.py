import csv

train_csv_path = './data/celeba_20k/train.csv'
test_csv_path = './data/celeba_20k/test.csv'

with open(train_csv_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        print(', '.join(row))
