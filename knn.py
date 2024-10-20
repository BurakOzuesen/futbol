# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# %%
# CSV verisini yükle
df = pd.read_csv('matches.csv')

# %%
df

# %%
df = df.dropna(axis='columns')
df = df.drop(["Saat", "Lig"], axis='columns')
df

# %%
# Virgüllü sayıları noktaya çevirip float yapacak fonksiyon (sadece sayısal olanları dönüştürür)
def convert_columns_to_float(df, columns):
    for col in columns:
        if col in df.columns:  # Sütunun veri çerçevesinde olup olmadığını kontrol et
            try:
                # Sadece sayısal olan ve virgül içeren değerleri float'a çevir
                df[col] = df[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) and ',' in x else x)
                df[col] = df[col].apply(lambda x: float(x.replace(',', '.')) if isinstance(x, str) else x)
            except ValueError:
                print(f"'{col}' sütunu dönüştürülemedi, bazı değerler sayısal değil.")
    return df

# Dönüştürülmesi gereken sütunlar
columns_to_convert = ['1', 'X', '2', 'IY1', 'IY2', 'IYX', '1.5A', '1.5U', '2.5A', '2.5U', '3.5A', '3.5U', 
                      '0-1', '2-3', '4-6', '7+', 'Var', 'Yok', '1/X', '1/2', 'X/2', 
                      'h', 'H1', 'HX', 'H2', 'IYA', 'IYU']

# Fonksiyonu uygulayalım
df = convert_columns_to_float(df, columns_to_convert)
df

# %%
# test = df.copy()
# while test.iloc[0].name != 744:
#     train, test = train_test_split(df, test_size=1)
# train, test = train_test_split(df, test_size=0.2)
train = df.copy()

# %%
test = pd.read_csv("test.csv")
test = test.dropna(axis='columns')
test = test.drop(["Saat", "Lig"], axis='columns')
test = convert_columns_to_float(test, columns_to_convert)
test

# %%
print(train.head(5))

# %%
print(test.head(5))
print(test.columns)

# %%
features = ['1', 'X', '2', 'IY1', 'IYX', 'IY2', 'H1', 'HX', 'H2', 'Var',
       'Yok', '1/X', '1/2', 'X/2', 'IYA', 'IYU', '1.5A', '1.5U', '2.5A',
       '2.5U', '3.5A', '3.5U', '0-1', '2-3', '4-6', '7+']

test_maci = test.iloc[77]

for train_index, train_maci in train.iterrows():
    total_fark = []
    for feature in features:
       oran_1 = test_maci[feature]
       oran_2 = train_maci[feature]
       try:
              oran_farki = abs(oran_1 - oran_2)
              total_fark.append((oran_farki))
       except:
              continue
    
    if total_fark:
       ortalama_fark = sum(total_fark) / len(total_fark)
    else:
       ortalama_fark = 0  # Eğer total_fark boşsa ortalama farkı 0 olarak ayarla
    # print(ortalama_fark)
    # print(train_index)
    train.loc[train_index, "Fark"] = ortalama_fark  # loc ile atama yapıyoruz
    # print(test.iloc[0])
    # print(index, row)

# %%
pd.set_option("display.max_columns", None)

# %%
print(train.sort_values("Fark").head(100)["MS"].values)

# %%
my_preds = train.sort_values("Fark").head(n=100)
scores = my_preds["MS"].values
iy_scores = my_preds["IY"].values

home_win = 0
draw = 0
away_win = 0
over_2_5 = 0
under_2_5 = 0
under_3_5 = 0
goal_2_3 = 0
tahmin = 0
oburu = 0

for index, score in enumerate(scores):
    try:
        
        home, away = score.split(" - ")
        home, away = int(home), int(away)
        
        iy_home, iy_away = iy_scores[index].split(" - ")
        iy_home, iy_away = int(iy_home), int(iy_away)
    except Exception as e:
        continue

    
    if home > away:
        home_win += 1
    if home == away:
        draw += 1
    if away > home:
        away_win += 1
    if away + home > 2.5:
        over_2_5 += 1
    if away + home < 2.5:
        under_2_5 += 1
    if away + home < 3.5:
        under_3_5 += 1
    if away + home == 2 or away + home == 3:
        goal_2_3 += 1

    if iy_home < iy_away and home < away:
        tahmin += 1
    if iy_home + iy_away < 1.5:
        oburu += 1


print("Home", home_win/len(scores))
print("Draw", draw/len(scores))
print("Away", away_win/len(scores))
print("Over 2.5", over_2_5/len(scores))
print("Under 2.5", under_2_5/len(scores))
print("Under 3.5", under_3_5/len(scores))
print("2_3", goal_2_3/len(scores))
print("tahmin", tahmin/len(scores))
print(oburu/len(scores))

# %%
print("Values")
print(home_win * test_maci["1"] / len(scores))
print(draw * test_maci["X"] / len(scores))
print(away_win * test_maci["2"] / len(scores))
print(over_2_5 * test_maci["2.5U"] / len(scores))
print(under_2_5 * test_maci["2.5A"] / len(scores))
print(under_3_5 * test_maci["3.5A"] / len(scores))
# print((draw + away_win) * test_maci["X/2"] / len(scores))
print(tahmin*2.29 / len(scores))
print(oburu*1.33 / len(scores))

# %%
test

# %%
error

features = ['1', 'X', '2', 'IY1', 'IYX', 'IY2', 'H1', 'HX', 'H2', 'Var',
       'Yok', '1/X', '1/2', 'X/2', 'IYA', 'IYU', '1.5A', '1.5U', '2.5A',
       '2.5U', '3.5A', '3.5U', '0-1', '2-3', '4-6', '7+']

from tqdm import tqdm
bulten = {}
for index, test_maci in tqdm(test.iterrows()):
    print(test_maci)
    
    for train_index, train_maci in train.iterrows():
        
        print(train_maci)
        continue
        total_fark = []
        for feature in features:
            oran_1 = test_maci[feature]
            oran_2 = train_maci[feature]
            
            try:
                    oran_farki = abs(oran_1 - oran_2)
                    total_fark.append((oran_farki))
            except:
                    continue
        
        if total_fark:
            ortalama_fark = sum(total_fark) / len(total_fark)
        else:
            ortalama_fark = 0  # Eğer total_fark boşsa ortalama farkı 0 olarak ayarla
            # print(ortalama_fark)
            # print(train_index)
            print(train.loc[train_index, "Fark"])
            train.loc[train_index, "Fark"] = ortalama_fark  # loc ile atama yapıyoruz
            print(train.loc[train_index, "Fark"])
            # print(test.iloc[0])
            # print(index, row)
    
    my_preds = train.sort_values("Fark").head(100)
    scores = my_preds["MS"].values
    # print(my_preds)

    home_win = 0
    draw = 0
    away_win = 0
    over_2_5 = 0
    under_3_5 = 0

    for score in scores:

        try:
             
            home, away = score.split(" - ")
            home, away = int(home), int(away)
        except Exception as e:
             continue
        if home > away:
            home_win += 1
        if home == away:
            draw += 1
        if away > home:
            away_win += 1
        if away + home > 2.5:
            over_2_5 += 1
        if away + home < 3.5:
            under_3_5 += 1

    print(test_maci["2.5U"])
    print(over_2_5)
    continue

    # print("Home", home_win/len(scores))
    # print("Draw", draw/len(scores))
    # print("Away", away_win/len(scores))
    # print("Over 2.5", over_2_5/len(scores))
    # print("Under 3.5", under_3_5/len(scores))
    # print("Values")
    # print(home_win * test_maci["1"] / len(scores))
    # print(draw * test_maci["X"] / len(scores))
    # print(away_win * test_maci["2"] / len(scores))
    # print(over_2_5 * test_maci["2.5U"] / len(scores))
    # print(under_3_5 * test_maci["3.5A"] / len(scores))
    try:
         
        bulten[index] = over_2_5 * test_maci["2.5U"] / len(scores)
    except Exception as E:
         continue

# %%
bulten

# %%
sorted_dict = {k: v for k, v in sorted(bulten.items(), key=lambda item: item[1], reverse=True)}
sorted_dict

# %%
test_maci

# %%
test.head(1)

# %%
df = pd.read_csv('matches.csv')
df.loc[test.iloc[0].name]