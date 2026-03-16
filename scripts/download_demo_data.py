import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.to_csv("data/titanic_train.csv", index=False)
print(f"Downloaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution:\n{df['Survived'].value_counts()}")
