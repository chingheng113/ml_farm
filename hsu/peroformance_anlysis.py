from my_util import data_util
from sklearn.metrics import accuracy_score

result = data_util.load_all('see.csv')
y_label = result.iloc[:, 1:18]
y_pred = result.iloc[:, 18:]
threshold = 0.3
y_pred.iloc[(y_pred >= threshold)] = 1
y_pred.iloc[(y_pred < threshold)] = 0

print(accuracy_score(y_label, y_pred))