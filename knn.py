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

test_maci = test.iloc[0]

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

bahis_tipleri = pd.read_csv("bahis_tipleri.csv")
bahis_sayacı = [0 for i in range(bahis_tipleri.shape[0])]

for index, score in enumerate(scores):
    try:
        
        home, away = score.split(" - ")
        home, away = int(home), int(away)
        
        iy_home, iy_away = iy_scores[index].split(" - ")
        iy_home, iy_away = int(iy_home), int(iy_away)

        home_first_half = iy_home
        away_first_half = iy_away
        home_second_half = home - iy_home
        away_second_half = away - iy_away
    except Exception as e:
        continue

    # Maç Sonucu Bahisleri
    if home > away:
        bahis_sayacı[0] += 1
    elif home == away:
        bahis_sayacı[1] += 1
    elif away > home:
        bahis_sayacı[2] += 1

    # Alt Üst Bahisleri
    total_goals = home + away
    if total_goals < 0.5:
        bahis_sayacı[3] += 1
    elif total_goals > 0.5:
        bahis_sayacı[4] += 1
    
    if total_goals < 1.5:
        bahis_sayacı[5] += 1
    elif total_goals > 1.5:
        bahis_sayacı[6] += 1
    
    if total_goals < 2.5:
        bahis_sayacı[7] += 1
    elif total_goals > 2.5:
        bahis_sayacı[8] += 1

    if total_goals < 3.5:
        bahis_sayacı[9] += 1
    elif total_goals > 3.5:
        bahis_sayacı[10] += 1

    if total_goals < 4.5:
        bahis_sayacı[11] += 1
    elif total_goals > 4.5:
        bahis_sayacı[12] += 1
    
    if total_goals > 5.5:
        bahis_sayacı[13] += 1
    elif total_goals < 5.5:
        bahis_sayacı[14] += 1

    # İY/MS Tahminleri
    if iy_home > iy_away and home > away:
        bahis_sayacı[15] += 1  # İlk Yarı / Maç Sonucu 1 / 1
    elif iy_home > iy_away and home == away:
        bahis_sayacı[16] += 1  # İlk Yarı / Maç Sonucu 1 / 0
    elif iy_home > iy_away and home < away:
        bahis_sayacı[17] += 1  # İlk Yarı / Maç Sonucu 1 / 2
    elif iy_home == iy_away and home > away:
        bahis_sayacı[18] += 1  # İlk Yarı / Maç Sonucu 0 / 1
    elif iy_home == iy_away and home == away:
        bahis_sayacı[19] += 1  # İlk Yarı / Maç Sonucu 0 / 0
    elif iy_home == iy_away and home < away:
        bahis_sayacı[20] += 1  # İlk Yarı / Maç Sonucu 0 / 2
    elif iy_home < iy_away and home > away:
        bahis_sayacı[21] += 1  # İlk Yarı / Maç Sonucu 2 / 1
    elif iy_home < iy_away and home == away:
        bahis_sayacı[22] += 1  # İlk Yarı / Maç Sonucu 2 / 0
    elif iy_home < iy_away and home < away:
        bahis_sayacı[23] += 1  # İlk Yarı / Maç Sonucu 2 / 2

    if home > 0 and away > 0:
        bahis_sayacı[24] += 1  # Karşılıklı Gol VAR
    else:
        bahis_sayacı[25] += 1  # Karşılıklı Gol YOK
    
    if iy_home + iy_away < 0.5:
        bahis_sayacı[26] += 1  # İlk Yarı Alt 0.5
    elif iy_home + iy_away > 0.5:
        bahis_sayacı[27] += 1  # İlk Yarı Üst 0.5

    if iy_home + iy_away < 1.5:
        bahis_sayacı[28] += 1  # İlk Yarı Alt 1.5
    elif iy_home + iy_away > 1.5:
        bahis_sayacı[29] += 1  # İlk Yarı Üst 1.5

    if iy_home + iy_away < 2.5:
        bahis_sayacı[30] += 1  # İlk Yarı Alt 2.5
    elif iy_home + iy_away > 2.5:
        bahis_sayacı[31] += 1  # İlk Yarı Üst 2.5
    
    if iy_home > iy_away:
        bahis_sayacı[32] += 1  # İlk Yarı Sonucu 1 (Ev Sahibi Önde)
    elif iy_home == iy_away:
        bahis_sayacı[33] += 1  # İlk Yarı Sonucu 0 (Beraberlik)
    elif iy_home < iy_away:
        bahis_sayacı[34] += 1  # İlk Yarı Sonucu 2 (Deplasman Önde)

    # Maç Sonucu ve Alt / Üst 1.5
    if home > away and total_goals < 1.5:
        bahis_sayacı[35] += 1  # Maç Sonucu 1 ve 1.5 Alt
    elif home == away and total_goals < 1.5:
        bahis_sayacı[36] += 1  # Maç Sonucu 0 ve 1.5 Alt
    elif home < away and total_goals < 1.5:
        bahis_sayacı[37] += 1  # Maç Sonucu 2 ve 1.5 Alt
    elif home > away and total_goals > 1.5:
        bahis_sayacı[38] += 1  # Maç Sonucu 1 ve 1.5 Üst
    elif home == away and total_goals > 1.5:
        bahis_sayacı[39] += 1  # Maç Sonucu 0 ve 1.5 Üst
    elif home < away and total_goals > 1.5:
        bahis_sayacı[40] += 1  # Maç Sonucu 2 ve 1.5 Üst

    # Maç Sonucu ve Alt / Üst 2.5
    if home > away and total_goals < 2.5:
        bahis_sayacı[41] += 1  # Maç Sonucu 1 ve 2.5 Alt
    elif home == away and total_goals < 2.5:
        bahis_sayacı[42] += 1  # Maç Sonucu 0 ve 2.5 Alt
    elif home < away and total_goals < 2.5:
        bahis_sayacı[43] += 1  # Maç Sonucu 2 ve 2.5 Alt
    elif home > away and total_goals > 2.5:
        bahis_sayacı[44] += 1  # Maç Sonucu 1 ve 2.5 Üst
    elif home == away and total_goals > 2.5:
        bahis_sayacı[45] += 1  # Maç Sonucu 0 ve 2.5 Üst
    elif home < away and total_goals > 2.5:
        bahis_sayacı[46] += 1  # Maç Sonucu 2 ve 2.5 Üst

    # Maç Sonucu ve Alt / Üst 3.5
    if home > away and total_goals < 3.5:
        bahis_sayacı[47] += 1  # Maç Sonucu 1 ve 3.5 Alt
    elif home == away and total_goals < 3.5:
        bahis_sayacı[48] += 1  # Maç Sonucu 0 ve 3.5 Alt
    elif home < away and total_goals < 3.5:
        bahis_sayacı[49] += 1  # Maç Sonucu 2 ve 3.5 Alt
    elif home > away and total_goals > 3.5:
        bahis_sayacı[50] += 1  # Maç Sonucu 1 ve 3.5 Üst
    elif home == away and total_goals > 3.5:
        bahis_sayacı[51] += 1  # Maç Sonucu 0 ve 3.5 Üst
    elif home < away and total_goals > 3.5:
        bahis_sayacı[52] += 1  # Maç Sonucu 2 ve 3.5 Üst

    # Maç Sonucu ve Alt / Üst 4.5
    if home > away and total_goals < 4.5:
        bahis_sayacı[53] += 1  # Maç Sonucu 1 ve 4.5 Alt
    elif home == away and total_goals < 4.5:
        bahis_sayacı[54] += 1  # Maç Sonucu 0 ve 4.5 Alt
    elif home < away and total_goals < 4.5:
        bahis_sayacı[55] += 1  # Maç Sonucu 2 ve 4.5 Alt
    elif home > away and total_goals > 4.5:
        bahis_sayacı[56] += 1  # Maç Sonucu 1 ve 4.5 Üst
    elif home == away and total_goals > 4.5:
        bahis_sayacı[57] += 1  # Maç Sonucu 0 ve 4.5 Üst
    elif home < away and total_goals > 4.5:
        bahis_sayacı[58] += 1  # Maç Sonucu 2 ve 4.5 Üst

    # Çifte Şans
    if home >= away:
        bahis_sayacı[59] += 1  # Çifte Şans 1 ve 0
    if home != away:
        bahis_sayacı[60] += 1  # Çifte Şans 1 ve 2
    if away >= home:
        bahis_sayacı[61] += 1  # Çifte Şans 0 ve 2

    # Ev Sahibi Alt / Üst
    if home < 0.5:
        bahis_sayacı[62] += 1  # Ev Sahibi 0.5 Alt
    if home >= 0.5:
        bahis_sayacı[63] += 1  # Ev Sahibi 0.5 Üst
    if home < 1.5:
        bahis_sayacı[64] += 1  # Ev Sahibi 1.5 Alt
    if home >= 1.5:
        bahis_sayacı[65] += 1  # Ev Sahibi 1.5 Üst
    if home < 2.5:
        bahis_sayacı[66] += 1  # Ev Sahibi 2.5 Alt
    if home >= 2.5:
        bahis_sayacı[67] += 1  # Ev Sahibi 2.5 Üst

    # Deplasman Alt / Üst
    if away < 0.5:
        bahis_sayacı[68] += 1  # Deplasman 0.5 Alt
    if away >= 0.5:
        bahis_sayacı[69] += 1  # Deplasman 0.5 Üst
    if away < 1.5:
        bahis_sayacı[70] += 1  # Deplasman 1.5 Alt
    if away >= 1.5:
        bahis_sayacı[71] += 1  # Deplasman 1.5 Üst
    if away < 2.5:
        bahis_sayacı[72] += 1  # Deplasman 2.5 Alt
    if away >= 2.5:
        bahis_sayacı[73] += 1  # Deplasman 2.5 Üst

    # Handikaplı Maç Sonucu
    # (0:1) Handikap
    if home > (away + 1):
        bahis_sayacı[74] += 1  # Handikaplı 1 (0:1)
    if home == (away + 1):
        bahis_sayacı[75] += 1  # Handikaplı 0 (0:1)
    if home < (away + 1):
        bahis_sayacı[76] += 1  # Handikaplı 2 (0:1)

    # (1:0) Handikap
    if (home + 1) > away:
        bahis_sayacı[77] += 1  # Handikaplı 1 (1:0)
    if (home + 1) == away:
        bahis_sayacı[78] += 1  # Handikaplı 0 (1:0)
    if (home + 1) < away:
        bahis_sayacı[79] += 1  # Handikaplı 2 (1:0)

    # (2:0) Handikap
    if (home + 2) > away:
        bahis_sayacı[80] += 1  # Handikaplı 1 (2:0)
    if (home + 2) == away:
        bahis_sayacı[81] += 1  # Handikaplı 0 (2:0)
    if (home + 2) < away:
        bahis_sayacı[82] += 1  # Handikaplı 2 (2:0)

    # (0:2) Handikap
    if home > (away + 2):
        bahis_sayacı[83] += 1  # Handikaplı 1 (0:2)
    if home == (away + 2):
        bahis_sayacı[84] += 1  # Handikaplı 0 (0:2)
    if home < (away + 2):
        bahis_sayacı[85] += 1  # Handikaplı 2 (0:2)

    # Toplam Gol
    if total_goals >= 0 and total_goals <= 1:
        bahis_sayacı[86] += 1  # Toplam Gol: 0-1 gol
    if total_goals >= 2 and total_goals <= 3:
        bahis_sayacı[87] += 1  # Toplam Gol: 2-3 gol
    if total_goals >= 4 and total_goals <= 5:
        bahis_sayacı[88] += 1  # Toplam Gol: 4-5 gol
    if total_goals >= 6:
        bahis_sayacı[89] += 1  # Toplam Gol: 6+ gol

    # Hangi Yarıda Daha Çok Gol Olur
    if home_first_half + away_first_half > home_second_half + away_second_half:
        bahis_sayacı[90] += 1  # İlk Yarıda Daha Çok Gol Olur
    elif home_first_half + away_first_half == home_second_half + away_second_half:
        bahis_sayacı[91] += 1  # Eşit
    else:
        bahis_sayacı[92] += 1  # İkinci Yarıda Daha Çok Gol Olur

    # Ev Sahibi Takım Hangi Yarıda Daha Çok Gol Atar
    if home_first_half > home_second_half:
        bahis_sayacı[93] += 1  # Ev Sahibi İlk Yarıda Daha Çok Gol Atar
    elif home_first_half == home_second_half:
        bahis_sayacı[94] += 1  # Ev Sahibi Eşit Gol Atar
    else:
        bahis_sayacı[95] += 1  # Ev Sahibi İkinci Yarıda Daha Çok Gol Atar

    # Deplasman Takım Hangi Yarıda Daha Çok Gol Atar
    if away_first_half > away_second_half:
        bahis_sayacı[96] += 1  # Deplasman İlk Yarıda Daha Çok Gol Atar
    elif away_first_half == away_second_half:
        bahis_sayacı[97] += 1  # Deplasman Eşit Gol Atar
    else:
        bahis_sayacı[98] += 1  # Deplasman İkinci Yarıda Daha Çok Gol Atar

    # Ev Sahibi Takım Gol Yemeden Kazanır mı?
    if away == 0 and home > away:
        bahis_sayacı[99] += 1  # Ev Sahibi Gol Yemeden Kazanır
    else:
        bahis_sayacı[100] += 1  # Ev Sahibi Gol Yiyerek Kazanır

    # Deplasman Takım Gol Yemeden Kazanır mı?
    if home == 0 and away > home:
        bahis_sayacı[101] += 1  # Deplasman Takım Gol Yemeden Kazanır
    else:
        bahis_sayacı[102] += 1  # Deplasman Takım Gol Yiyerek Kazanır

    # Ev Sahibi Takım Her İki Yarıyı da Kazanır mı?
    if iy_home > iy_away and home_second_half > away_second_half:
        bahis_sayacı[103] += 1  # Ev Sahibi Her İki Yarıyı da Kazanır
    else:
        bahis_sayacı[104] += 1  # Ev Sahibi Her İki Yarıyı da Kazanamaz

    # Deplasman Takımı Her İki Yarıyı da Kazanır mı?
    if iy_away > iy_home and away_second_half > home_second_half:
        bahis_sayacı[105] += 1  # Deplasman Takımı Her İki Yarıyı da Kazanır
    else:
        bahis_sayacı[106] += 1  # Deplasman Takımı Her İki Yarıyı da Kazanamaz

    # İlk Yarı Çifte Şans, 1 ve 0
    if iy_home > iy_away or iy_home == iy_away:
        bahis_sayacı[107] += 1  # Ev Sahibi 1 veya 0 ya da Deplasman 1 ve 0

    # İlk Yarı Çifte Şans, 1 ve 2
    if iy_home > iy_away or iy_away > iy_home:
        bahis_sayacı[108] += 1  # Ev Sahibi 1 veya Deplasman 2

    # İlk Yarı Çifte Şans, 0 ve 2
    if iy_home == iy_away or iy_away > iy_home:
        bahis_sayacı[109] += 1  # Ev Sahibi 0 ya da Deplasman 2

    # Hangi Takım Kaç Farkla Kazanır

    # Ev sahibi takım 3+ farkla kazanır mı?
    if home - away >= 3:
        bahis_sayacı[110] += 1  # Ev 3+ farkla kazanır

    # Ev sahibi takım 2 farkla kazanır mı?
    if home - away == 2:
        bahis_sayacı[111] += 1  # Ev 2 farkla kazanır

    # Ev sahibi takım 1 farkla kazanır mı?
    if home - away == 1:
        bahis_sayacı[112] += 1  # Ev 1 farkla kazanır

    # Deplasman takım 1 farkla kazanır mı?
    if away - home == 1:
        bahis_sayacı[113] += 1  # Dep 1 farkla kazanır

    # Deplasman takım 2 farkla kazanır mı?
    if away - home == 2:
        bahis_sayacı[114] += 1  # Dep 2 farkla kazanır

    # Deplasman takım 3+ farkla kazanır mı?
    if away - home >= 3:
        bahis_sayacı[115] += 1  # Dep 3+ farkla kazanır

    # Beraberlik durumu
    if home == away:
        bahis_sayacı[116] += 1  # Fark 0, beraberlik

    # İkinci Yarı Sonucu
    # İkinci yarı gol farkını hesapla (örnek: ikinci yarıdaki goller)

    # İkinci yarı ev sahibi takım kazanır mı?
    if home_second_half > away_second_half:
        bahis_sayacı[117] += 1  # İkinci Yarı Sonucu,1 (Ev sahibi kazanır)

    # İkinci yarı berabere mi biter?
    elif home_second_half == away_second_half:
        bahis_sayacı[118] += 1  # İkinci Yarı Sonucu,0 (Beraberlik)

    # İkinci yarı deplasman takımı kazanır mı?
    elif away_second_half > home_second_half:
        bahis_sayacı[119] += 1  # İkinci Yarı Sonucu,2 (Deplasman kazanır)

    # Toplam gol sayısı tek mi?
    if total_goals % 2 != 0:
        bahis_sayacı[120] += 1  # Tek (total gol sayısı tek)

    # Toplam gol sayısı çift mi?
    else:
        bahis_sayacı[121] += 1  # Çift (total gol sayısı çift)

    if score == "1 - 0":
        bahis_sayacı[122] += 1
    elif score == "2 - 0":
        bahis_sayacı[123] += 1
    elif score == "2 - 1":
        bahis_sayacı[124] += 1
    elif score == "3 - 0":
        bahis_sayacı[125] += 1
    elif score == "3 - 1":
        bahis_sayacı[126] += 1
    elif score == "3 - 2":
        bahis_sayacı[127] += 1
    elif score == "4 - 0":
        bahis_sayacı[128] += 1
    elif score == "4 - 1":
        bahis_sayacı[129] += 1
    elif score == "4 - 2":
        bahis_sayacı[130] += 1
    elif score == "5 - 0":
        bahis_sayacı[131] += 1
    elif score == "5 - 1":
        bahis_sayacı[132] += 1
    elif score == "6 - 0":
        bahis_sayacı[133] += 1
    elif score == "0 - 0":
        bahis_sayacı[134] += 1
    elif score == "1 - 1":
        bahis_sayacı[135] += 1
    elif score == "2 - 2":
        bahis_sayacı[136] += 1
    elif score == "3 - 3":
        bahis_sayacı[137] += 1
    elif score == "0 - 1":
        bahis_sayacı[138] += 1
    elif score == "0 - 2":
        bahis_sayacı[139] += 1
    elif score == "1 - 2":
        bahis_sayacı[140] += 1
    elif score == "0 - 3":
        bahis_sayacı[141] += 1
    elif score == "1 - 3":
        bahis_sayacı[142] += 1
    elif score == "2 - 3":
        bahis_sayacı[143] += 1
    elif score == "0 - 4":
        bahis_sayacı[144] += 1
    elif score == "1 - 4":
        bahis_sayacı[145] += 1
    elif score == "2 - 4":
        bahis_sayacı[146] += 1
    elif score == "0 - 5":
        bahis_sayacı[147] += 1
    elif score == "1 - 5":
        bahis_sayacı[148] += 1
    elif score == "0 - 6":
        bahis_sayacı[149] += 1
    elif score == "Diğer":
        bahis_sayacı[150] += 1

    # Maç Sonucu ve Karşılıklı Gol
    # (home == 0 or away == 0) KGYOK
    # (home != 0 and away != 0) KGVAR
    if (home > away) and (home != 0 and away != 0):
        bahis_sayacı[151] += 1
    if (home > away) and not(home != 0 and away != 0):
        bahis_sayacı[152] += 1
    if (home == away) and (home != 0 and away != 0):
        bahis_sayacı[153] += 1
    if (home == away) and not(home != 0 and away != 0):
        bahis_sayacı[154] += 1
    if (home < away) and (home != 0 and away != 0):
        bahis_sayacı[155] += 1
    if (home < away) and not(home != 0 and away != 0):
        bahis_sayacı[156] += 1

    # İlk Yarı Sonucu ve İlk Yarı Karşılıklı Gol
    if (home_first_half > away_first_half) and (home_first_half != 0 and away_first_half != 0):
        bahis_sayacı[157] += 1
    if (home_first_half > away_first_half) and not(home_first_half != 0 and away_first_half != 0):
        bahis_sayacı[158] += 1
    if (home_first_half == away_first_half) and (home_first_half != 0 and away_first_half != 0):
        bahis_sayacı[159] += 1
    if (home_first_half == away_first_half) and not(home_first_half != 0 and away_first_half != 0):
        bahis_sayacı[160] += 1
    if (home_first_half < away_first_half) and (home_first_half != 0 and away_first_half != 0):
        bahis_sayacı[161] += 1
    if (home_first_half < away_first_half) and not(home_first_half != 0 and away_first_half != 0):
        bahis_sayacı[162] += 1

    # Altı/Üstü ve Karşılıklı Gol
    # (home == 0 or away == 0) KGYOK
    # (home != 0 and away != 0) KGVAR
    if (total_goals < 2.5) and (home != 0 and away != 0):
        bahis_sayacı[163] += 1
    if (total_goals > 2.5) and (home != 0 and away != 0):
        bahis_sayacı[164] += 1
    if (total_goals < 2.5) and not(home != 0 and away != 0):
        bahis_sayacı[165] += 1
    if (total_goals > 2.5) and not(home != 0 and away != 0):
        bahis_sayacı[166] += 1

    # İlk Yarı Skoru
    if iy_home == 0 and iy_away == 0:
        bahis_sayacı[167] += 1
    elif iy_home == 1 and iy_away == 1:
        bahis_sayacı[168] += 1
    elif iy_home == 2 and iy_away == 2:
        bahis_sayacı[169] += 1
    elif iy_home == 1 and iy_away == 0:
        bahis_sayacı[170] += 1
    elif iy_home == 2 and iy_away == 0:
        bahis_sayacı[171] += 1
    elif iy_home == 2 and iy_away == 1:
        bahis_sayacı[172] += 1
    elif iy_home == 0 and iy_away == 1:
        bahis_sayacı[173] += 1
    elif iy_home == 0 and iy_away == 2:
        bahis_sayacı[174] += 1
    elif iy_home == 1 and iy_away == 2:
        bahis_sayacı[175] += 1
    else:
        bahis_sayacı[176] += 1

    # İY/MS Skorları
    if iy_home == 0 and iy_away == 0 and home == 0 and away == 0:
        bahis_sayacı[177] += 1
    if iy_home == 0 and iy_away == 0 and home == 0 and away == 1:
        bahis_sayacı[178] += 1
    if iy_home == 0 and iy_away == 0 and home == 0 and away == 2:
        bahis_sayacı[179] += 1
    if iy_home == 0 and iy_away == 0 and home == 0 and away == 3:
        bahis_sayacı[180] += 1
    if iy_home == 0 and iy_away == 0 and home == 1 and away == 0:
        bahis_sayacı[181] += 1
    if iy_home == 0 and iy_away == 0 and home == 1 and away == 1:
        bahis_sayacı[182] += 1
    if iy_home == 0 and iy_away == 0 and home == 1 and away == 2:
        bahis_sayacı[183] += 1
    if iy_home == 0 and iy_away == 0 and home == 2 and away == 0:
        bahis_sayacı[184] += 1
    if iy_home == 0 and iy_away == 0 and home == 2 and away == 1:
        bahis_sayacı[185] += 1
    if iy_home == 0 and iy_away == 0 and home == 3 and away == 0:
        bahis_sayacı[186] += 1
    if iy_home == 0 and iy_away == 0 and ((home + away) > 4):
        bahis_sayacı[187] += 1
    if iy_home == 1 and iy_away == 0 and home == 1 and away == 0:
        bahis_sayacı[188] += 1
    if iy_home == 1 and iy_away == 0 and home == 1 and away == 1:
        bahis_sayacı[189] += 1
    if iy_home == 1 and iy_away == 0 and home == 1 and away == 2:
        bahis_sayacı[190] += 1
    if iy_home == 1 and iy_away == 0 and home == 2 and away == 0:
        bahis_sayacı[191] += 1
    if iy_home == 1 and iy_away == 0 and home == 2 and away == 1:
        bahis_sayacı[192] += 1
    if iy_home == 1 and iy_away == 0 and home == 3 and away == 0:
        bahis_sayacı[193] += 1
    if iy_home == 1 and iy_away == 0 and ((home + away) > 4):
        bahis_sayacı[194] += 1
    if iy_home == 0 and iy_away == 1 and home == 0 and away == 1:
        bahis_sayacı[195] += 1
    if iy_home == 0 and iy_away == 1 and home == 0 and away == 2:
        bahis_sayacı[196] += 1
    if iy_home == 0 and iy_away == 1 and home == 0 and away == 3:
        bahis_sayacı[197] += 1
    if iy_home == 0 and iy_away == 1 and home == 1 and away == 1:
        bahis_sayacı[198] += 1
    if iy_home == 0 and iy_away == 1 and home == 1 and away == 2:
        bahis_sayacı[199] += 1
    if iy_home == 0 and iy_away == 1 and home == 2 and away == 1:
        bahis_sayacı[200] += 1
    if iy_home == 0 and iy_away == 1 and ((home + away) > 4):
        bahis_sayacı[201] += 1
    if iy_home == 2 and iy_away == 0 and home == 2 and away == 0:
        bahis_sayacı[202] += 1
    if iy_home == 2 and iy_away == 0 and home == 2 and away == 1:
        bahis_sayacı[203] += 1
    if iy_home == 2 and iy_away == 0 and home == 3 and away == 0:
        bahis_sayacı[204] += 1
    if iy_home == 2 and iy_away == 0 and ((home + away) > 4):
        bahis_sayacı[205] += 1
    if iy_home == 1 and iy_away == 1 and home == 1 and away == 1:
        bahis_sayacı[206] += 1
    if iy_home == 1 and iy_away == 1 and home == 1 and away == 2:
        bahis_sayacı[207] += 1
    if iy_home == 1 and iy_away == 1 and home == 2 and away == 1:
        bahis_sayacı[208] += 1
    if iy_home == 1 and iy_away == 1 and ((home + away) > 4):
        bahis_sayacı[209] += 1
    if iy_home == 0 and iy_away == 2 and home == 0 and away == 2:
        bahis_sayacı[210] += 1
    if iy_home == 0 and iy_away == 2 and home == 0 and away == 3:
        bahis_sayacı[211] += 1
    if iy_home == 0 and iy_away == 2 and home == 1 and away == 2:
        bahis_sayacı[212] += 1
    if iy_home == 0 and iy_away == 2 and ((home + away) > 4):
        bahis_sayacı[213] += 1
    if iy_home == 3 and iy_away == 0 and home == 3 and away == 0:
        bahis_sayacı[214] += 1
    if iy_home == 3 and iy_away == 0 and ((home + away) > 4):
        bahis_sayacı[215] += 1
    if iy_home == 2 and iy_away == 1 and home == 2 and away == 1:
        bahis_sayacı[216] += 1
    if iy_home == 2 and iy_away == 1 and ((home + away) > 4):
        bahis_sayacı[217] += 1
    if iy_home == 1 and iy_away == 2 and home == 1 and away == 2:
        bahis_sayacı[218] += 1
    if iy_home == 1 and iy_away == 2 and ((home + away) > 4):
        bahis_sayacı[219] += 1
    if iy_home == 0 and iy_away == 3 and home == 0 and away == 3:
        bahis_sayacı[220] += 1
    if iy_home == 0 and iy_away == 3 and ((home + away) > 4):
        bahis_sayacı[221] += 1
    if ((iy_home + iy_away) > 4) and ((home + away) > 4):
        bahis_sayacı[222] += 1

    # İlk Yarı Tek/Çift
    iy_toplam_gol = iy_home + iy_away
    if iy_toplam_gol % 2 != 0:
        bahis_sayacı[223] += 1 # İY Tek
    else:
        bahis_sayacı[224] += 1 # İY Çift

    # İlk Yarı Karşılıklı Gol
    if iy_home != 0 and iy_away != 0:
        bahis_sayacı[225] += 1
    else:
        bahis_sayacı[226] += 1

    # Ev Sahibi İlk Yarı Altı/Üstü
    if iy_home > 0.5:
        bahis_sayacı[227] += 1
    else:
        bahis_sayacı[228] += 1
    
    # Deplasman İlk Yarı Altı/Üstü
    if iy_away > 0.5:
        bahis_sayacı[229] += 1
    else:
        bahis_sayacı[230] += 1

    # İlk Yarı Sonucu ve İlk Yarı Altı/Üstü
    if iy_home > iy_away and (iy_home + iy_away) < 2:
        bahis_sayacı[231] += 1  # 1 ve Alt 1.5
    elif iy_home == iy_away and (iy_home + iy_away) < 2:
        bahis_sayacı[232] += 1  # 0 ve Alt 1.5
    elif iy_home < iy_away and (iy_home + iy_away) < 2:
        bahis_sayacı[233] += 1  # 2 ve Alt 1.5
    elif iy_home > iy_away and (iy_home + iy_away) >= 2:
        bahis_sayacı[234] += 1  # 1 ve Üst 1.5
    elif iy_home == iy_away and (iy_home + iy_away) >= 2:
        bahis_sayacı[235] += 1  # 0 ve Üst 1.5
    elif iy_home < iy_away and (iy_home + iy_away) >= 2:
        bahis_sayacı[236] += 1  # 2 ve Üst 1.5

    if iy_home > iy_away or home_second_half > away_second_half:
        bahis_sayacı[237] += 1  # Ev Sahibi Yarı Kazanır, Evet
    else:
        bahis_sayacı[238] += 1  # Ev Sahibi Yarı Kazanır, Hayır

    if iy_away > iy_home or away_second_half > home_second_half:
        bahis_sayacı[239] += 1  # Deplasman Yarı Kazanır, Evet
    else:
        bahis_sayacı[240] += 1  # Deplasman Yarı Kazanır, Hayır

    # İlk yarı ve maç sonu golleri 1.5 alt kontrolü
    if (iy_home + iy_away < 1.5) and (home_second_half + away_second_half < 1.5):
        bahis_sayacı[241] += 1  # İki Yarı da Alt 1.5, Evet
    else:
        bahis_sayacı[242] += 1  # İki Yarı da Alt 1.5, Hayır

    # İlk yarı ve maç sonu golleri 1.5 üst kontrolü
    if (iy_home + iy_away > 1.5) and (home_second_half + away_second_half > 1.5):
        bahis_sayacı[243] += 1  # İki Yarı da Üst 1.5, Evet
    else:
        bahis_sayacı[244] += 1  # İki Yarı da Üst 1.5, Hayır

    # İkinci Yarı Karşılıklı Gol var
    if (home_second_half != 0 and away_second_half != 0):
        bahis_sayacı[245] += 1
    else:
        bahis_sayacı[246] += 1

current_df = bahis_tipleri.copy()
current_df["Olasılıklar"] = bahis_sayacı
current_df.to_csv("current.csv")
exit()

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