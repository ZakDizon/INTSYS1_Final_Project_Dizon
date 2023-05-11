import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('top2018.csv')

# Preprocessing
X = data.drop(['id', 'name', 'artists', 'key'], axis=1)
y = data['key']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on test set
y_pred = lr.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Plot number of songs in each key
key_counts = y.value_counts()
key_counts.plot.bar()
plt.title('Number of Songs in Each Key')
plt.xlabel('Key')
plt.ylabel('# of Songs')
plt.show()