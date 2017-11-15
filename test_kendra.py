import io
from pandas import read_json, read_csv
from matplotlib import pyplot
# load dataset
dataset = read_json('candles_from_1510095600000_to_1510759800000',convert_dates=[0])

# get only 5 rows
#dataset = dataset.head(5)



dataset.to_csv('test.csv')

dataset = read_csv('test.csv', index_col=1)
dataset.columns = ['no','open','close','high','low','volume']
dataset.index.name = 'date'
dataset.drop('no', axis=1, inplace=True)

print(dataset)

values = dataset.values

groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
