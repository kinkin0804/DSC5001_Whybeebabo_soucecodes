#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv("apartment_rent.csv", encoding="utf-8-sig")

print("first five rows of the original data：")
print(df.head())


# In[2]:


print("data shape (rows, columns):")
print(df.shape)

print("\nfield types")
print(df.dtypes)

print("\nnumber of missing fields")
print(df.isnull().sum())


# In[5]:


before_filter_cnt = df.shape[0]

cond_monthly = df["price_type"] == "Monthly"
cond_category = df["category"] == "housing/rent/apartment"

df_removed = df[~(cond_monthly & cond_category)]
removed_count = df_removed.shape[0]

df = df[cond_monthly & cond_category].copy()
after_filter_cnt = df.shape[0]

print(f"Total number before filtering:{before_filter_cnt}")
print(f"Number of rows excluded (not for Monthly/Not for Apartment category)：{removed_count}")
print(f"Number of rows to retain after filtering:{after_filter_cnt}")

print("\nDistribution of the excluded sample price_type:")
print(df_removed["price_type"].value_counts())
print("\nDistribution of excluded sample categories:")
print(df_removed["category"].value_counts())


# In[7]:


df[["bedrooms", "bathrooms"]] = df[["bedrooms", "bathrooms"]].fillna(0)


# In[9]:


duplicate_num = df.duplicated().sum()
print(f"Number of repeated lines:{duplicate_num}")
df = df.drop_duplicates()
print("data shape:", df.shape)


# In[11]:


col_price = "price"
Q1 = df[col_price].quantile(0.25)
Q3 = df[col_price].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df["is_outlier_price"] = np.where(
    (df[col_price] < lower) | (df[col_price] > upper),
    1,
    0
)

outlier_count = df["is_outlier_price"].sum()
normal_count = (df["is_outlier_price"] == 0).sum()

print(f"price abnormal samples：{outlier_count}")
print(f"price normal samples：{normal_count}")
print(f"total number of rows：{df.shape[0]}")
print(f"price abnormal lower limit：{lower:.2f}，pricea bnormal upper limit：{upper:.2f}")


# In[13]:


import pandas as pd
import numpy as np

fake_lat = 39.8163
fake_lon = -98.5576

df["is_geo_valid"] = 1

cond1 = (df["latitude"] == fake_lat) & (df["longitude"] == fake_lon)
cond2 = (df["latitude"] < -90) | (df["latitude"] > 90) | (df["longitude"] < -180) | (df["longitude"] > 180)

df.loc[cond1 | cond2, "is_geo_valid"] = 0
df.loc[cond1 | cond2, ["latitude", "longitude"]] = np.nan

print("number of valid coordinates：", df["is_geo_valid"].sum())
print("number of invalid coordinates：", (df["is_geo_valid"] == 0).sum())


# In[15]:


check_cols = ["price", "square_feet"]

for col in check_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    flag_name = f"is_outlier_{col}"
    df[flag_name] = 0
    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), flag_name] = 1
    
    print(f"{col} number of outliers：", df[flag_name].sum())


# In[17]:


df["has_amenity"] = 0
df.loc[df["amenities"].notna() & (df["amenities"] != ""), "has_amenity"] = 1

df["has_pet_rule"] = 0
df.loc[df["pets_allowed"].notna(), "has_pet_rule"] = 1

df["is_studio"] = 0
df.loc[df["bedrooms"] == 0, "is_studio"] = 1

print("Number of studio apartments：", df["is_studio"].sum())


# In[19]:


df["body_len"] = df["body"].fillna("").apply(lambda x: len(str(x)))


# In[21]:


dup_match_cols = ["title", "cityname", "price", "square_feet", "bedrooms", "bathrooms"]

df["property_key"] = df[dup_match_cols].astype(str).agg("|".join, axis=1)

group_count = df["property_key"].value_counts()

df["is_dup_listing"] = df["property_key"].map(lambda x: 1 if group_count[x] > 1 else 0)

dup_total_rows = (df["is_dup_listing"] == 1).sum()
unique_dup_group = (group_count[group_count > 1]).shape[0]

print(f"total number of records for repeated listing announcements：{dup_total_rows}")
print(f"number of different repeated property listings groups：{unique_dup_group}")
print(f"number of unique non-repeated housing listings：{(df['is_dup_listing'] == 0).sum()}")


print("\n sample:")
dup_keys = group_count[group_count > 1].index[:3]
for k in dup_keys:
    print(f"\n>>> property_key：{k}")
    print(df[df["property_key"] == k][["id", "title", "cityname", "price", "square_feet"]])


# In[23]:


err_sqft = (df["square_feet"] <= 0) & df["square_feet"].notna()
err_bed = (df["bedrooms"] < 0) & df["bedrooms"].notna()
err_bath = (df["bathrooms"] < 0) & df["bathrooms"].notna()

df["is_business_error"] = 0
df.loc[err_sqft | err_bed | err_bath, "is_business_error"] = 1

error_count = df["is_business_error"].sum()
log_text = f"""
===== 清洗日志 =====
总数据行数：{len(df)}
筛选前总记录数：{before_filter_cnt}
剔除记录数（非Monthly月租 / 非公寓类别）：{removed_count}
筛选后保留记录数：{after_filter_cnt}
说明：仅保留price_type='Monthly'纯月租，category='housing/rent/apartment'公寓房源；周租、混合租期、住宅/短租房源全部剔除。
面积<=0错误行数：{err_sqft.sum()}
卧室负数错误行数：{err_bed.sum()}
浴室负数错误行数：{err_bath.sum()}
业务逻辑错误总行数：{error_count}
坐标无效行数：{(df["is_geo_valid"] == 0).sum()}
price异常数量：{df["is_outlier_price"].sum()}
square_feet异常数量：{df["is_outlier_square_feet"].sum()}
【重复挂牌识别统计】
存在重复发布的房源分组数量：{unique_dup_group}
属于重复挂牌的记录总行数：{dup_total_rows}
仅出现一次的独立挂牌记录行数：{(df['is_dup_listing'] == 0).sum()}
"""
print(log_text)

with open("clean_log.txt", "w", encoding="utf-8") as f:
    f.write(log_text)


# In[25]:


df.to_csv("apartment_rent_preprocessed.csv", index=False, encoding="utf-8-sig")
print("apartment_rent_preprocessed.csv，log file:：clean_log.txt")


# In[ ]:




