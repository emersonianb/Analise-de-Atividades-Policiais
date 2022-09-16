#!/usr/bin/env python
# coding: utf-8

# <div class = "alert alert-block alert-warning">
#     <center><h2>Análise de Atividades Policiais em Rhode Island</h2>
# <center><h2>Estatística Aplicada a Computação</h2>
# <center>Aluno: Emerson Ian Bezerra de Sousa
#     <div>

# ### Bibliotecas importadas

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Dados importados

# In[2]:


police = pd.read_csv('police.csv', sep = ',')


# In[3]:


weather = pd.read_csv('weather.csv', sep = ',')


# <div class = "alert alert-block alert-success">
# <h2>TAREFA 1
# <div>

# ##### Quantidade de variáveis e registros policiais disponíveis

# In[4]:


li, co = police.shape


# In[5]:


print(f"O DataFrame 'police' apresenta {co} variáveis e {li} registros policiais")


# ##### Número de dados faltosos de cada variável

# In[6]:


dadosf = pd.DataFrame(police.isna().sum()).reset_index()
dadosf.columns = ['Variáveis', 'Qtd. NaN']
dadosf


# #####  Por trabalhar somente com dados de um único estado as variáveis county_name e state foram excluídas

# In[7]:


police = police.drop(columns=['state', 'county_name'], axis = 1)
police.head()


# ##### Remoação dos dados faltosos da coluna driver_gender

# In[8]:


print('Total de linhas no DataFrame')
print(len(police))
police = police.dropna(subset=['driver_gender'])
print('Total de linhas no DataFrame sem os valores faltosos em Gênero:')
print(len(police))


# #####  Verificação se as variáveis search_conducted, is_arrested e district são do tipo booeano e transformaão das que não são para o tipo booleano

# In[9]:


police['search_conducted'].dtype


# In[10]:


police['district'].dtype


# In[11]:


police['is_arrested'].dtype


# In[12]:


police['is_arrested'].astype(bool)


# #####  Combinação das coulunas stop_date e stop_time em uma coluna stop_datetime e converção pro formato data e hora (tipo datetime)

# In[13]:


stop_datetime = police['stop_date'] + ", " + police['stop_time']
police['stop_datetime'] = pd.to_datetime(stop_datetime)
police = police.drop(columns=['stop_date', 'stop_time'], axis=1)
police.head()


# #####  Tranformação da coluna stop_datetime no índice do dataframe

# In[14]:


police.set_index('stop_datetime', inplace=True)
police


# <div class =  "alert alert-block alert-success">
#     <h2>TAREFA 2
# <div>

# ##### Distribuição de frequências da variável violation: qual a infração mais comum e a menos notificada?

# In[15]:


fv = pd.DataFrame(police['violation'].value_counts()).reset_index()
fv.columns = ['Violação', 'Frequencia']
print('Violação mais comum:')
fv.head(1)


# In[16]:


print('Violação menos comum:')
fv.tail(1)


# #####  Motoristas do sexo masculino e feminino tendem a cometer diferentes tipos de infrações de trânsito?
# - Para isso, crie uma tabela de contingência para frequência absoluta e outra para frequência relativa, contendo a distribuição conjunta das variáveis driver_gender e violation.

# In[17]:


filtragem = police['driver_gender'] == 'M'
gen_m = police[filtragem]
gen_m = pd.DataFrame(gen_m['violation'].value_counts()).reset_index()


# In[18]:


filtragem = police['driver_gender'] == 'F'
gen_f = police[filtragem]
gen = pd.DataFrame(gen_f['violation'].value_counts()).reset_index()
gen.columns = ['Violação', 'Fa_F']
gen['Fa_M'] = gen_m['violation']
gen


# In[19]:


fr_f = ((gen['Fa_F']/gen['Fa_F'].sum())*100).round(decimals=2)
fr_m = ((gen['Fa_M']/gen['Fa_M'].sum())*100).round(decimals=2)
gen_r = gen.copy()
gen_r['Fa_F'], gen_r['Fa_M'] = fr_f, fr_m
gen_r.rename(columns={'Fa_F': 'Fr_F', 'Fa_M': 'Fr_M'})


# ##### Gráfico de barras agrupadas para ilustrar os dados das tabelas de contingência construídas;

# In[20]:


plt.show(gen.plot.barh('Violação', title = 'Frequencia Absoluta'))


# In[21]:


plt.show(gen_r.plot.barh('Violação', title = 'Frequencia Relativa', color=['black','gray']))


# #####  Quando um motorista é parado por excesso de velocidade, muitas pessoas acreditam que o gênero influencia se o motorista receberá uma multa ou um aviso. 
# - Para tentar responder essa pergunta, crie uma tabela de contingência considerando as variáveis driver_gender e stop_outcome e então vai comparar a porcentagem de paradas resultados de uma "Citation" versus um "Warning"

# In[22]:


filtragem = police['driver_gender'] == 'M'
f = police[filtragem]
df_m = pd.DataFrame(((f['stop_outcome'].value_counts(normalize=True))*100).round(decimals=2)).head(2)


# In[23]:


filtragem = police['driver_gender'] == 'F'
f = police[filtragem]
df_f = pd.DataFrame(((f['stop_outcome'].value_counts(normalize=True))*100).round(decimals=2)).head(2)
df_f.columns = ['Freq. F']
df_f['Freq. M'] = df_m['stop_outcome']
df_f


# ##### O gênero afeta a escolha de veículos a serem revistados?
# - Primeiro, calcule a porcentagem de todas as paradas no DataFrame que resultam em uma revista de veículo

# In[24]:


((police['search_conducted'].value_counts()/police['search_conducted'].value_counts().sum())*100).round(decimals=2)


# In[25]:


rev = pd.DataFrame(police.groupby('driver_gender')['search_conducted'].value_counts())
rev['search_conducted'] = ((rev['search_conducted']/rev['search_conducted'].sum())*100).round(decimals=2)
rev


# In[26]:


mask = police['search_conducted'] == True
rev_tipo = pd.DataFrame(police[mask].groupby('driver_gender')['search_type'].value_counts())
rev_tipo['search_type'] = ((rev_tipo['search_type']/rev_tipo['search_type'].sum())*100).round(decimals=2)
rev_tipo.columns = ['Porcentagem']
rev_tipo


# <div class =  "alert alert-block alert-success">
# <h2>Tarefa 3
# <div>

# ##### A taxa de prisão varia de acordo com a hora do dia?
# - Primeiro, você calculará a taxa de prisão em todas as paradas no DataFrame, calculando a média da coluna is_arrested

# In[27]:


pd.DataFrame(police['is_arrested'].value_counts())


# - Em seguida, você calculará a taxa de prisão por hora usando o atributo de hora do índice. No final crie uma nova variável hourly_arrest_rate com os valores encontrados da taxa de prisão por hora;

# In[28]:


police.reset_index(inplace=True)
police['hora'] = police['stop_datetime'].dt.hour
mask = police['is_arrested'] == True
hourly_arrest_rate = pd.DataFrame(police[mask].groupby('is_arrested')['hora'].value_counts(normalize=True)*100).round(decimals=2
                                                                                                                     )
hourly_arrest_rate.columns = ['Taxa de prisão']
hourly_arrest_rate = hourly_arrest_rate.reset_index().drop(columns=['is_arrested']).set_index('hora')
hourly_arrest_rate.sort_values(by='hora')


# ##### Gráfico de linha mostrando a variável hourly_arrest_rate

# In[29]:


plt.show(hourly_arrest_rate.plot.bar(y='Taxa de prisão', title = 'Taxa de Prisões por Hora do Dia', color=['green']))


# ##### Em uma pequena parte das paradas de trânsito, drogas são encontradas no veículo durante uma busca. Essas interrupções relacionadas à drogas estão se tornando mais comuns com o tempo?
# - A coluna booleana drug_related_stop indica se drogas foram encontradas durante uma determinada parada. Você calculará a taxa anual de drogas reamostrando essa coluna

# In[30]:


police['ano'] = police['stop_datetime'].dt.year
mask = police['drugs_related_stop'] == True
taxa_droga_ano = pd.DataFrame(police[mask].groupby('drugs_related_stop')['ano'].value_counts(normalize=True)*100).round(
                                                                                                            decimals=2)
taxa_droga_ano.columns = ['Taxa anual de drogas']
taxa_droga_ano = taxa_droga_ano.reset_index().drop(columns=['drugs_related_stop']).set_index('ano')
taxa_droga_ano.sort_values(by='ano')


# ##### Gráfico de linha para visualizar como a taxa mudou ao longo do tempo;

# In[31]:


plt.show(taxa_droga_ano.plot.bar(title = 'Taxa anual de drogas', color=['red']))


# #####  Consideremos a hipótese de que, o aumento ou a diminuição das apreensões de drogas estão associadas ao aumento ou diminuição das abordagens políciais, ou seja, mais abordagens, geram mais apreensões e menos abordagens, menos apreensões de drogas.
# - Para isso, calcule a taxa de pesquisa anual reamostrando a coluna search_conducted e salve o resultado como Annual_search_rate. Concatene Annual_drug_rate e Annual_search_rate ao longo do eixo das colunas

# In[32]:


taxa_ano = pd.DataFrame(police['ano'].value_counts(normalize=True)*100).round(decimals=2)
taxa_ano.columns = ['Taxa anual']
taxa_ano = taxa_ano.sort_index(ascending=True)
taxa_ano['Taxa anual de drogas'] = taxa_droga_ano['Taxa anual de drogas']


# - Gráficos de linha para os resultados da concatenação

# In[33]:


plt.show(taxa_ano.plot.bar(title = 'Relação', color=['green', 'red']))


# ##### O estado de Rhode Island está dividido em seis distritos policiais, também conhecidos como zonas. Como as zonas se comparam em termos de quais infrações são detectadas pela polícia?
# - Para isso, crie uma distribuição conjunta entre as variáveis district e violation, usando uma tabela de contingência. Depois, selecione as linhas das zonas ’Zona K1’ a ’Zona K3’

# In[34]:


zonas = pd.DataFrame(police.groupby('district')['violation'].value_counts(normalize=True)*100).round(decimals=2)
zonas.columns = ['Taxa de infrações']
zonas = zonas.reset_index()
zonak1 = zonas[zonas['district'] == 'Zone K1']
zonak3 = zonas[zonas['district'] == 'Zone K3']
zonak13 = pd.concat([zonak1, zonak3])
zonak13.set_index(['district', 'violation'])


# ##### Gráfico de barras agrupadas para ilustrar os resultados obtidos na tabela

# In[35]:


plt.show(zonak13.plot.bar('violation', title = (f'Relação\nZona K1 x Zona K3'), color=['green']))


# Como as zonas se comparam em termos de quais infrações são detectadas pela polícia?
# - R: Em relação as infrações mais ocorrentes, Velocidade e Violações em movimento, a taxa da Zona K1 é maior. Ja sobre as outras ocorrências, as taxas da Zona K3 são maiores, além de acrescentar outra infração que não há na Zona K1, Cinto de Segurança.

# <div class =  "alert alert-block alert-success">
# <h2>Tarefa 4
# <div>

# ##### Explorando as temperaturas apresentadas no conjunto de dados
# - Carregue o conjunto, selecione as variáveis relativas à temperatura, imprima as principais medidas resumo usando o comando describe e plote os três boxplots dessas variáveis em um mesmo gráfico.

# In[36]:


tavg = weather[['TMIN', 'TAVG', 'TMAX']]
(tavg.describe()).round(decimals=2)


# In[37]:


tavg.plot.box()


# - O que você poderia comentar sobre as temperaturas, com base nos resultados obtidos?
#  - R: A temperatura mais baixa registrada foi de 4,48F, o que em Celsius siginifica 6,3°C, ou seja, uma temperatura muito fria, quase anormal ao comparar com a realidade brasieira.

# ##### Criação da variável TDIFF, que representa a diferença entre as temperaturas
# - apresente as medidas resumo e plote um histograma para essa variável.

# In[38]:


weather['TDIFF']=weather['TMAX']-weather['TMIN']
pd.DataFrame((weather['TDIFF'].describe()).round(decimals=2))


# In[39]:


weather['TDIFF'].hist()


# - O que pode dizer sobre a distribuição de dados?
#     - R: Os resultados das diferenças das temperaturas estão muito concetrados entre 10 e 20, segundo o histograma

# ###### Preparação dos DataFrames para serem mesclados.
# - No DataFrame sobre abordagens no trânsito, você transformará o índice stop_datetime para uma coluna (reset_index), pois o índice será perdido durante a mesclagem

# In[40]:


police.head()


# - Colocar DATE em um novo DataFrame (a coluna rating não existe)

# In[41]:


DATE = pd.DataFrame(weather['DATE'])
DATE


# ###### Mesclagem dos DataFrames.
# - Assim que a mesclagem for concluída, defina stop_datetime novamente como o índice

# In[42]:


police_weather = pd.merge(police, weather, left_index=True, right_index=True).set_index('stop_datetime')
police_weather


# ###### Levante duas questões e as responda usando qualquer técnica que ache necessária.

# ### Quais as porcentagens de paradas em relação a etnia?

# In[43]:


etnia = pd.DataFrame(police_weather['driver_race'].value_counts(normalize=True)).round(decimals=2)
etnia


# In[49]:


plt.show(etnia.plot.pie(y = 'driver_race'))


# ### Quais as porcentagens das violações cometidas por pessoas Pretas?

# In[45]:


mask = police_weather['driver_race'] == 'Black'
pp = pd.DataFrame(police_weather[mask]['violation'].value_counts()).round(decimals=2)
pp


# In[46]:


pp.plot.pie(y = 'violation')


# In[ ]:




