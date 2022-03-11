import pandas as pd
from apriori_python import apriori

Gold_Daily= pd.read_csv("base\Gold_Daily.csv")

agrupar = Gold_Daily.groupby('Abertura')['Maximo'].apply(list)

print(Gold_Daily.head(20))

freqItemSet, rules = apriori(agrupar, minSup=0.002, minConf=0.05)
print(len(agrupar))
print(rules)