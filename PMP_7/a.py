import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('auto-mpg.csv')
print(data.head())

data.dropna(subset=['horsepower', 'mpg'], inplace=True)
plt.scatter(data['horsepower'], data['mpg'])
plt.title('Relația dintre cp și mgb')
plt.xlabel('CP')
plt.ylabel('mpg')
plt.show()
