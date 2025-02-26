import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris_df = pd.read_csv('tp0/iris.csv')

print("head:")
print(iris_df.head())

print("info:")
print(iris_df.info())

print("describe:")
print(iris_df.describe())

#ver la cantidad de cada especie
sums = {}
for species, count in iris_df['Species'].value_counts().items():
    sums[species] = count
print("sums: " + str(sums))
    
    
# Create a boxplot for sepalLength, sepalWidth, petalWidth, and petalLength, separated by species
plt.figure(figsize=(12, 8))

# Boxplot for sepalLength
plt.subplot(2, 2, 1)
sns.boxplot(x='Species', y='SepalLengthCm', data=iris_df)
plt.title('Boxplot of Sepal Length by Species')

# Boxplot for sepalWidth
plt.subplot(2, 2, 2)
sns.boxplot(x='Species', y='SepalWidthCm', data=iris_df)
plt.title('Boxplot of Sepal Width by Species')

# Boxplot for petalWidth
plt.subplot(2, 2, 3)
sns.boxplot(x='Species', y='PetalWidthCm', data=iris_df)
plt.title('Boxplot of Petal Width by Species')

# Boxplot for petalLength
plt.subplot(2, 2, 4)
sns.boxplot(x='Species', y='PetalLengthCm', data=iris_df)
plt.title('Boxplot of Petal Length by Species')

plt.tight_layout()
plt.show()
