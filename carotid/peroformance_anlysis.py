from my_util import data_util
from carotid import carotid_data_util as cdu
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

target = 'RCCA'
soure = 'ex'
classifier = 'svm'

result = cdu.get_result(classifier+'_'+soure+'_'+target+'.csv')
label = result['label']
probas_ = result[['0', '1']].values
predict = data_util.labelize(probas_)
print(confusion_matrix(label, predict))
print('Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.')
print(classification_report(label, predict, digits=4))
fpr, tpr, thresholds = roc_curve(label, probas_[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr,
         label=' (AUC = %0.3f )' % roc_auc,
         lw=1, alpha=.8)
plt.show()


# for inx in range(0, 3, 1):
#     result = cdu.get_result(classifier+'_'+soure+'_'+target+'_'+str(inx)+'.csv')
#     label = list(result['label'].values)
#     probas_ = result[['0', '1']].values
#     predict = list(data_util.labelize(probas_))
#     print(classification_report(label, predict, digits=4))
#     if inx == 0:
#         labels = label
#         predicts = predict
#     else:
#         labels.extend(label)
#         predicts.extend(predict)
# print('---')
# print(classification_report(labels, predicts, digits=4))
