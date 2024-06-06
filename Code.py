import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score

nbaStats = pd.read_csv('nba_stats.csv')
dummyData = pd.read_csv('dummy_test.csv')
FeatureColumns= ['ORB', 'DRB', 'PF', 'TOV', 'AST', 'FGA', 'STL', '3PA', 'FTA', '2P', 'PTS', '3P%', 'FG', 'FT%']
feature= nbaStats[FeatureColumns]
target_column = nbaStats['Pos']


TrainFeature, TestFeature, TrainClass, TestClass = train_test_split(feature,target_column, test_size=0.2,stratify=target_column, random_state=0)

svm = SVC(random_state=0 ,kernel= 'linear') #SVM Classification
svm.fit(TrainFeature, TrainClass)
# Print training and testing scores on nba stats
print("\nModel trained using SVM Classification")
print("\nTraining set Accuracy: {:.3f}".format(svm.score(TrainFeature, TrainClass)))
print("\nValidation set Accuracy: {:.3f}".format(svm.score(TestFeature, TestClass)))

# Calculate confusion matrix for Training Set
predicted_classes = svm.predict(TrainFeature)
conf_matrix = confusion_matrix(TrainClass, predicted_classes)

# Convert confusion matrix to a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, columns=svm.classes_, index=svm.classes_)
conf_matrix_df.index.name = 'True'
conf_matrix_df.columns.name = 'Predicted'

conf_matrix_df['All'] = conf_matrix_df.sum(axis=1)
conf_matrix_df.loc['All'] = conf_matrix_df.sum()

# Print confusion matrix
print("\n\nConfusion matrix for Training Set:\n")
print(conf_matrix_df)

# Calculate confusion matrix for Training Set
predicted_classes = svm.predict(TestFeature)
conf_matrix = confusion_matrix(TestClass, predicted_classes)

# Convert confusion matrix to a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, columns=svm.classes_, index=svm.classes_)
conf_matrix_df.index.name = 'True'
conf_matrix_df.columns.name = 'Predicted'

conf_matrix_df['All'] = conf_matrix_df.sum(axis=1)
conf_matrix_df.loc['All'] = conf_matrix_df.sum()

# Print confusion matrix
print("\n\nConfusion matrix for Validation Set:\n")
print(conf_matrix_df)

##Cross validation
scores = cross_val_score(svm, feature, target_column, cv=10)
print("\n\nCross-Validation Scores:", scores)

#Calculate average score
average_score = scores.mean()
print("\nAverage Cross-Validation Score:", average_score)

#Calculating Accuracy for Dummy test data
dummy_features = dummyData[FeatureColumns]
d_predict = svm.predict(dummy_features)
accuracy = accuracy_score(dummyData['Pos'], d_predict)

print(f"\n\nAccuracy on Dummy Test data: {accuracy:.3f}")


# Calculate confusion matrix for dummy test data
d_predict = svm.predict(dummy_features)
conf_matrix = confusion_matrix(dummyData['Pos'], d_predict, labels=svm.classes_)

# Convert confusion matrix to a DataFrame for better visualization
conf_matrix_df = pd.DataFrame(conf_matrix, columns=svm.classes_, index=svm.classes_)
conf_matrix_df.index.name = 'True'
conf_matrix_df.columns.name = 'Predicted'

conf_matrix_df['All'] = conf_matrix_df.sum(axis=1)
conf_matrix_df.loc['All'] = conf_matrix_df.sum()

# Print confusion matrix
print("\nConfusion matrix on dummy test:\n")
print(conf_matrix_df)

