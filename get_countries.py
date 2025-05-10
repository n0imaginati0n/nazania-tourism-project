#!/usr/bin/env python3

import pandas as pd
from geopy.distance import geodesic

'''
    main idea of the trick:

    I need to know country capitals coordinate to replace the country with the distance.
    Country name will tell me not so much, but the distance will show a ticket price and
    probably, how reach the people there

    I found free dataset with country capitals, but the countrries there are not exactly 
    as they are in our dataset. IMPORTANT! I've took the country names from both train 
    and test data. this will not make a data leakage, as it is just an independent fact.
    but will solve the problem with gettin the distance for all countries in the test set
    as well

    as some names aren't the same, the plan is:

    1.
        to create a list of countries needed in our data:
            make a set, where all the countries from train and test set will be added

            train data: 105 countries
            test data: 87 countries
            all together: 118 countries

        all the names should be stripped and lowered

    2.
        load the dataset with coordinates, convert the country names: strip, lowercase
        find the names, which aren't found in this dataset and map them manually

        just out of teh start 93 out of the 118 countries where found

        25 left (

        most of these countries written with mistakes. i didn't check, but it is very 
        possbile, that in the data set correct and incorrect country names are present.
        records should eb duplicated for both.

        korea - i think, it is south korea, because north korea hasn't much tourists

        scotland - still Great Britain
    
'''

def get_travel_countries() -> set:
    train = pd.read_csv('data/Train.csv')
    test = pd.read_csv('data/Test.csv')
    train_c = train['country'].unique()
    test_c = test['country'].unique()
    travel_countries = set(train_c)
    travel_countries.update(test_c)
    print(f'Test have {len(test_c)} countries, Train have {len(train_c)} countries, merged set have {len(travel_countries)} of them')
    return { cnt.strip().upper() for cnt in travel_countries }

def get_available_countries() -> pd.DataFrame:
    df = pd.read_csv('countries/country-capital-lat-long-population.csv')
    df['Country'] = df['Country'].str.strip().str.upper()
    return df

def calc_distance(row: pd.Series, tanzania_latd: float, tanzania_longd: float):
    return round(
        geodesic(
            (row['Latitude'], row['Longitude']), (tanzania_latd, tanzania_longd)
        ).kilometers,
        3
    )


def main():

    travel_countries = get_travel_countries()
    df_countries = get_available_countries()
    
    df_found = df_countries[ df_countries['Country'].isin( travel_countries ) ]
    print(f'found { df_found.shape[0] } countries out of { len( travel_countries )}')

    not_found = { cnt for cnt in travel_countries if cnt not in df_found['Country'].unique() }

    df_countries_mapped = pd.DataFrame(df_countries)
    df_countries_mapped['Country'] = df_countries_mapped['Country'].map({
        'CABO VERDE': 'CAPE VERDE',
        'DJIBOUTI': 'DJIBOUT',
        'CÔTE D\'IVOIRE': 'IVORY COAST',
        'SOMALIA': 'SOMALI',
        'CHINA, TAIWAN PROVINCE OF CHINA': 'TAIWAN',
        'BOSNIA AND HERZEGOVINA': 'BOSNIA',
        'COSTA RICA': 'COSTARICA',
        'CZECHIA': 'CZECH REPUBLIC',
        'DEMOCRATIC REPUBLIC OF THE CONGO': 'DRC',
        'IRAN (ISLAMIC REPUBLIC OF)': 'IRAN',
        'COMOROS': 'COMORO',
        'MOROCCO': 'MORROCO',
        'REPUBLIC OF KOREA': 'KOREA',
        'RUSSIAN FEDERATION': 'RUSSIA',
        'SWITZERLAND': 'SWIZERLAND',
        'TRINIDAD AND TOBAGO': 'TRINIDAD TOBACCO',
        'MALTA': 'MALT',
        'UNITED ARAB EMIRATES': 'UAE',
        'BULGARIA':'BURGARIA',
        'PHILIPPINES': 'PHILIPINES',
        'VIET NAM': 'VIETNAM',
        'UNITED KINGDOM': 'SCOTLAND',
        'SAUDI ARABIA': 'SAUD ARABIA',
        'UKRAINE': 'UKRAIN',
        'TFYR MACEDONIA': 'MACEDONIA'
    })

    df_found_2 = df_countries_mapped[ df_countries_mapped['Country'].isin(not_found) ]
    print(f'found { df_found_2.shape[0] } countries out of { len(not_found )}')
    
    df_all = pd.concat([df_found, df_found_2], axis = 0) \
        .reset_index() \
        .drop(['index', 'Capital City', 'Population', 'Capital Type'], axis = 1)


    tanzania_latd  = df_countries[ df_countries['Country'] == 'UNITED REPUBLIC OF TANZANIA'].iloc[:, 2].iloc[0]
    tanzania_longd = df_countries[ df_countries['Country'] == 'UNITED REPUBLIC OF TANZANIA'].iloc[:, 3].iloc[0]

    df_all['distance_to_tanzania_km'] = df_all.apply(
        lambda row: calc_distance(row, tanzania_latd, tanzania_longd),
        axis=1)

    df_all = df_all.drop(['Longitude', 'Latitude'], axis = 1)
    df_all.columns = [c.lower() for c in df_all.columns]

    df_all.to_csv('data/distances.csv', index=False, encoding='utf-8', sep = ',')


    print(df_all)

if __name__ == '__main__':
    main()
