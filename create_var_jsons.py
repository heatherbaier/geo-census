import pandas as pd
import json


df = pd.read_csv("./data/mexico2010.csv")
df = df.fillna(0)
df.head()


socio = ['total_pop', 'weighted_avg_income', 'weighted_avg_earned_income', 'perc_rural', 'perc_owned', 'perc_married_with_children_hhtype', 'perc_single_parent_hhtype', 'perc_extended_family_hhtype', 'avg_nfams', 'perc_single', 
         'perc_married', 'avg_chborn', 'avg_chsurv', 'avg_num_years_from_last_birth', 'avg_chdead', 'perc_foreign_born_nativity', 'perc_employed', 'perc_disabled', 'perc_no_coverage_hlthcov', 'avg_age', 'sum_num_intmig', 'perc_no_onemeal', 
         'avg_famsize', 'avg_nchild', 'avg_nchlt5', 'avg_agedeadyr']
infra = ['perc_yes_electricity', 'perc_no_piped_water', 'perc_sewage_system', 'perc_electricity_fuelcook', 'perc_yes_phone', 'perc_yes_cell', 'perc_yes_internet', 'perc_trash_burned', 'perc_yes_autos', 'perc_yes_hotwater', 'perc_yes_computer', 
         'perc_yes_washer', 'perc_yes_refrig', 'perc_yes_tv', 'perc_yes_radio', 'avg_room_num', 'perc_no_toilet', 'perc_yes_school', 'perc_yes_literacy', 'perc_less_than_primary_edu', 'perc_primary_edu', 'perc_secondary_edu', 'perc_university_edu', 
         'perc_wood_fuelcook', 'perc_trash_collected_directly', 'avg_YRSCHOOL']


for col in socio:
    cur_dict = dict(zip(df['GEO2_MX'], df[col]))
    json_name = "./data/socio_vars/" + col + ".json"
    with open(json_name, 'w') as outfile:
        json.dump(cur_dict, outfile)
        

for col in infra:
    cur_dict = dict(zip(df['GEO2_MX'], df[col]))
    json_name = "./data/infra_vars/" + col + ".json"
    with open(json_name, 'w') as outfile:
        json.dump(cur_dict, outfile)
