import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

df = pd.read_csv("../final_dataset.csv")


# Mapping categorical data

subtype_mapping = {
    'flat_studio': 1,
    'kot': 2,
    'apartment': 3,
    'service_flat': 4,
    'ground_floor': 5,
    'loft': 6,
    'duplex': 7,
    'house': 8,
    'bungalow': 9,
    'country_cottage': 10,
    'town_house': 11,
    'chalet': 12,
    'villa': 13,
    'penthouse': 14,
    'triplex': 15,
    'mixed_use_building': 16,
    'apartment_block': 17,
    'farmhouse': 18,
    'pavilion': 19,
    'show_house': 20,
    'mansion': 21,
    'exceptional_property': 22,
    'manor_house': 23,
    'castle': 24,
    'other_property': 25
}
df['SubtypeOfProperty_num'] = df['SubtypeOfProperty'].map(subtype_mapping)

type_of_sale_mapping = {
    'residential_monthly_rent' : 1,
    'annuity_monthly_amount' : 2,  
    'annuity_without_lump_sum' : 3,       
    'annuity_lump_sum' : 4,                 
    'homes_to_build' : 5,
    'residential_sale' : 6     
}
df['TypeOfSale_num'] = df['TypeOfSale'].map(type_of_sale_mapping)

PEB_mapping = {
    'NO_DATA': 0,
    'G': 1, 
    'G_C': 1,
    'G_F': 1,  
    'F': 2,
    'F_C': 2,
    'F_D': 2,
    'F_E': 2,
    'E': 3,
    'E_C': 3,
    'E_D': 3,
    'D': 4,
    'C': 5,
    'B': 6,
    'B_A': 6,
    'A': 7,
    'A_A+': 7,
    'A+': 8,
    'A++': 9     
}
df['PEB'] = df['PEB'].fillna('NO_DATA')
df['PEB_num'] = df['PEB'].map(PEB_mapping)

data = pd.DataFrame(df["LivingArea"])
living_area_imputer = IterativeImputer(random_state=0)
data_imputed = pd.DataFrame(living_area_imputer.fit_transform(data), columns=["LivingArea"])
df["LivingArea"] = data_imputed["LivingArea"].astype("int32")

region_mapping = {
    'Wallonie' : 1,
    'Flanders' : 2,  
    'Brussels' : 3   
}

province_mapping = {
    'Hainaut' : 1,
    'Liège' : 2,
    'Luxembourg' : 3,
    'Namur' : 4,
    'Walloon Brabant' : 5,
    'Limburg' : 6,
    'East Flanders': 7,
    'Flemish Brabant': 8,
    'Antwerp': 9,
    'West Flanders' : 10,
    'Brussels' : 11
}

df = df.dropna(subset=['Region','Province']) 
df['Region_num'] = df['Region'].map(region_mapping)   
df['Province_num'] = df['Province'].map(province_mapping)

kitchen_mapping = {
    'NO_DATA' : 0,
    'NOT_INSTALLED': 1,
    'USA_UNINSTALLED': 2,
    'SEMI_EQUIPPED': 3,
    'USA_SEMI_EQUIPPED': 4,
    'INSTALLED': 5,
    'USA_INSTALLED': 6,
    'HYPER_EQUIPPED': 7,
    'USA_HYPER_EQUIPPED': 8
}
df['Kitchen'] = df['Kitchen'].fillna('NO_DATA')
df['Kitchen_num'] = df['Kitchen'].map(kitchen_mapping)

state_of_building_mapping = {
    'NO_DATA' : 0,
    'TO_RESTORE' : 1,
    'TO_RENOVATE' : 2,
    'TO_BE_DONE_UP' : 3,
    'GOOD' : 4,
    'AS_NEW' : 5,
    'JUST_RENOVATED' : 6
}
df['StateOfBuilding'] = df['StateOfBuilding'].fillna('NO_DATA')
df['StateOfBuilding_num'] = df['StateOfBuilding'].map(state_of_building_mapping)


# Replacing NaN values with 0

fill_with_zeros = ['Furnished','Garden','Fireplace','SwimmingPool','Terrace','ToiletCount','ShowerCount','SurfaceOfPlot']
for i in fill_with_zeros:
    df[i] = df[i].fillna(0)

df.loc[df['Garden'] == 0, 'GardenArea'] = 0         


# Locality 

locality_dict = df.groupby('Locality')['Price'].mean().to_dict()
sorted_locality_dict = dict(sorted(locality_dict.items(), key=lambda item: item[1]))
value_mapping = 1
for locality in sorted_locality_dict:
    sorted_locality_dict[locality] = value_mapping
    value_mapping += 1
df['Locality_num'] = df['Locality'].map(sorted_locality_dict)
df = df.dropna(subset=['Locality_num'])

# Removing columns  
clean_data = df.drop(columns=['Url','Country','BathroomCount','Garden','Kitchen','Locality','PEB','PostalCode','PropertyId','RoomCount','ShowerCount','StateOfBuilding','SubtypeOfProperty','TypeOfSale','Province','District','ConstructionYear','Fireplace','MonthlyCharges','ToiletCount','Terrace','Furnished','SwimmingPool','FloodingZone','NumberOfFacades','Region'])
clean_data.to_csv("clean_data.csv")               