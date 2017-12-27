import csv,copy

csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

class CSVHelper(object):
    """docstring for CSVHelper"""
    def __init__(self):
        super(CSVHelper, self).__init__()


    @staticmethod
    def load_csv(path):
        rows = []
        reader = csv.reader(open(path, encoding='mac_roman')) # mac roman encoding
        print('starting for loop!')
        for row in reader:
            rows.append(', '.join(row))
        return rows


# example
# tweets = CSVHelper.load_csv("Tweets_2016London.csv")