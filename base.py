# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/desafio_indicium_imdb.csv")

# %%
df.shape
df.info()
df.head(5)
# %%
df['Gross'] = df['Gross'].replace(',', '', regex=True) 
df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
df['Gross'] = df['Gross'].fillna(df['Gross'].median())
df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mean())
df['No_of_Votes'] = pd.to_numeric(df['No_of_Votes'], errors='coerce')

df['Certificate'] = df['Certificate'].fillna("Não Informado")


#### temos 999 linhas e 16 colunas contando com a index
#### verifica se ainda existe valores vazios
# %%
df.isnull().sum()
df[df.isnull().any(axis=1)].head(10)
df.duplicated().sum()

# %%

df.describe(include=object)
# %%
df['Released_Year'].unique()
df['Genre'].value_counts().unique()
# %%
df[df['Released_Year'] == 'PG']

### não entendi o que seria esse PG mas vou manter
############ até aqui entendemos a base, analisamos valores vazios e duplicados

########## analise 
# %%
df['IMDB_Rating'].hist(bins=20)
plt.xlabel("Nota IMDb")
plt.ylabel("Quantidade filmes")
plt.title("Distribuição das notas")
plt.show()

# %%

##### com base na nota do IMDB e Nº de votos seriam esses filmes que eu indicaria
df[['Series_Title', 'IMDB_Rating','No_of_Votes']].sort_values(by=['IMDB_Rating','No_of_Votes'], ascending=[False, False]).head(5)

# %%

plt.scatter(df['No_of_Votes'], df['Gross'])
plt.xlabel("Número de votos")
plt.ylabel("Faturamento (Gross)")
plt.title("Votos vs Faturamento")
plt.show()

# %%

df.corr()

# %%
tem_relacao = [
    'IMDB_Rating',
    'No_of_Votes',
    'Meta_score'
]
# %%

corr = df[tem_relacao].corr()
ax = plt.subplots(figsize=(11, 8))
sns.heatmap(corr,  annot=True, annot_kws={"size": 10})



######### não sei como fazer a prveisao ainda, tem que usar bibliotecas de ML que ainda não domino