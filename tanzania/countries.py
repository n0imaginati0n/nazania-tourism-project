#!/usr/bin/env python3

import os
import shutil
import kagglehub

import pandas as pd
from geopy.distance import geodesic


def load_kagglehub_data(
    dataset_url: str,
    dataset_file_name: str,
    destination_path: str):
    
    if not os.path.exists(destination_path):
        path = kagglehub.dataset_download(dataset_url)
        shutil.copyfile(
            os.path.join(path, dataset_file_name),
            destination_path)

def load_extra_data(dest_dir: str = 'data'):

    '''
    https://www.kaggle.com/datasets/zedataweaver/global-salary-data

    Currency Conversion:
        All salary figures have been converted to USD using real-time
        exchange rates to ensure consistency and facilitate cross-country
        comparisons and analysis.

    Monthly Basis: Please note that all salary values in this dataset
        are provided on a monthly basis, making it ideal for analyzing
        and comparing monthly income trends

    Reliable Source: The data is sourced from SalaryExplorer, a reputable
        platform known for its accurate and up-to-date salary information.
        It is a valuable resource for job seekers, employers, and researchers
    '''

    '''
    https://www.kaggle.com/datasets/dataanalyst001/all-capital-cities-in-the-world?select=all+capital+cities+in+the+world.csv
    
    Country:
        This column specifies the country in which each capital city is located.
    Continent:
        This column indicates the continent on which each capital city is situated.
    Latitude:
        This column provides the latitude coordinates of each capital city.
    Longitude:
        This column provides the longitude coordinates of each capital city
    '''

    datafiles = [
        ("zedataweaver/global-salary-data", 'salary_data.csv', 'data/salary_data_ext.csv'),
        # ("dataanalyst001/all-capital-cities-in-the-world", 'all capital cities in the world.csv', 'data/capitals_data_ext.csv')
    ]

    for data in datafiles:
        load_kagglehub_data(data[0], data[1], data[2])

def calc_distance(row: pd.Series, tanzania_latd: float, tanzania_longd: float):
    return round(
        geodesic(
            (row['latitude'], row['longitude']), (tanzania_latd, tanzania_longd)
        ).kilometers,
        3
    )

def get_countries_data() -> pd.DataFrame:
    load_extra_data('data')

    df_countries = pd.read_csv('data/country_data_ext.csv')
    df_salary = pd.read_csv('data/salary_data_ext.csv')

    df_salary.drop(
        ['wage_span', 'average_salary', 'lowest_salary', 'highest_salary'],
        axis = 1,
        inplace = True)
    df_salary.rename(columns = {
        'country_name': 'country',
        'continent_name': 'continent'
    }, inplace = True)
    df_salary['country'] = df_salary['country'].str.strip().str.upper()

    df_countries.drop(
        ['Capital City', 'Population', 'Capital Type'],
        axis = 1,
        inplace = True
    )
    df_countries.rename(columns = {
        'Country': 'country',
        'Latitude': 'latitude',
        'Longitude': 'longitude'
    }, inplace = True)
    df_countries['country'] = df_countries['country'].str.strip().str.upper()

    # I need to prepare result data set in form
    # 'country', 'salary', 'distance to tanzania, km'
    #
    # first country - salary. 
    # for each country I need one salary. i.e all the countries 
    # should be available in 'salary' data frame to maked it 
    # possible to join

    # need a names mapping. all the country names will be converted 
    # to uppercase for better matching
    #

    missed_salaries_countries = (
        ('ANGUILLA', 1750, 'Caribbean'),
        ('CHANNEL ISLANDS', 4600, 'Europe'),
        ('SAINT HELENA', 1040, 'Africa'),
        ('TOKELAU', 150, 'Oceania'),
        ('WALLIS AND FUTUNA ISLANDS', 3200, 'Oceania'),
        ('SAINT PIERRE AND MIQUELON', 1839, "Northern America"),
        ('NAURU', 613, 'Oceania'),
        ('ISLE OF MAN', 3965, 'Europe'),
        ('ISRAEL', 3657, 'Asia'),
        ('HOLY SEE', 3500, 'Europe'),
        ('CARIBBEAN NETHERLANDS', 1750, 'Caribbean'),
        ('CURAÇAO', 1000, 'South America'),
        ('NIUE', 4333, 'Oceania'),
        ('FALKLAND ISLANDS (MALVINAS)', 3755, 'South America'),
        ('KUWAIT', 4057, 'Asia'),
        ('SINT MAARTEN (DUTCH PART)', 3380, 'Caribbean'),
        ('TUVALU', 450, 'Oceania'),
        ('SOUTH SUDAN', 793, 'Africa'),
    )

    df_missed_salaries_countries = pd.DataFrame({
        'country': [ itm[0] for itm in missed_salaries_countries ],
        'continent': [ itm[1] for itm in missed_salaries_countries ],
        'median_salary': [ itm[2] for itm in missed_salaries_countries ]
    })

    df_salary = pd.concat([df_salary, df_missed_salaries_countries], ignore_index = True)

    map_countries = {
        'FAROE ISLANDS': 'FAEROE ISLANDS',
        'CAPE VERDE':'CABO VERDE',
        'MICRONESIA': 'MICRONESIA (FED. STATES OF)',
        'RUSSIA': 'RUSSIAN FEDERATION',
        'IRAN': 'IRAN (ISLAMIC REPUBLIC OF)',
        'PALESTINE': 'STATE OF PALESTINE',
        'TANZANIA': 'UNITED REPUBLIC OF TANZANIA',
        'KOREA (NORTH)': 'DEM. PEOPLE\'S REPUBLIC OF KOREA',
        'REUNION': 'RÉUNION',
        'BOLIVIA': 'BOLIVIA (PLURINATIONAL STATE OF)',
        'COTE DIVOIRE': 'CÔTE D\'IVOIRE',
        'HONG KONG': 'CHINA, HONG KONG SAR',
        'VIRGIN ISLANDS (US)': 'UNITED STATES VIRGIN ISLANDS',
        'SYRIA': 'SYRIAN ARAB REPUBLIC',
        'TAIWAN': 'CHINA, TAIWAN PROVINCE OF CHINA',
        'LAOS': 'LAO PEOPLE\'S DEMOCRATIC REPUBLIC',
        'EAST TIMOR': 'TIMOR-LESTE',
        'VIETNAM': 'VIET NAM',
        'BRUNEI': 'BRUNEI DARUSSALAM',
        'CZECH REPUBLIC': 'CZECHIA',
        'VENEZUELA': 'VENEZUELA (BOLIVARIAN REPUBLIC OF)',
        'UNITED STATES': 'UNITED STATES OF AMERICA',
        'CONGO DEMOCRATIC REPUBLIC': 'DEMOCRATIC REPUBLIC OF THE CONGO',
        'KOREA (SOUTH)': 'REPUBLIC OF KOREA',
        'MACAO': 'CHINA, MACAO SAR',
        'VIRGIN ISLANDS (BRITISH)': 'BRITISH VIRGIN ISLANDS',
        'MACEDONIA': 'TFYR MACEDONIA',
        'MOLDOVA': 'REPUBLIC OF MOLDOVA'
    }
    df_salary.replace(map_countries, inplace=True)

    df_all = pd.merge(df_countries, df_salary, on='country')
    
    tanzania_latd  = df_all[ df_all['country'] == 'UNITED REPUBLIC OF TANZANIA'].loc[:, 'latitude'].iloc[0]
    tanzania_longd = df_all[ df_all['country'] == 'UNITED REPUBLIC OF TANZANIA'].loc[:, 'longitude'].iloc[0]

    df_all['distance'] = df_all.apply(
        lambda row: calc_distance(row, tanzania_latd, tanzania_longd),
        axis=1)

    df_all.drop(['latitude', 'longitude'], axis = 1, inplace=True)
    return df_all

def extend_train_test_with_countries(df: pd.DataFrame) -> pd.DataFrame:
    df_countries = get_countries_data()

    df['country'] = df['country'].str.strip().str.upper()

    # the country names used in test/train data should be rteplaced with the 
    # country names from our list. in the imput data they can be broken,
    # mistaken, dialectial

    input_countries_to_official = {
        'BURGARIA': 'BULGARIA',
        'COMORO': 'COMOROS',
        'IVORY COAST': 'CÔTE D\'IVOIRE',
        'MORROCO': 'MOROCCO',
        'PHILIPINES': 'PHILIPPINES',
        'UAE': 'UNITED ARAB EMIRATES',
        'SCOTLAND': 'UNITED KINGDOM',
        'IRAN': 'IRAN (ISLAMIC REPUBLIC OF)',
        'KOREA': 'REPUBLIC OF KOREA',
        'SWIZERLAND': 'SWITZERLAND',
        'COSTARICA': 'COSTA RICA',
        'RUSSIA': 'RUSSIAN FEDERATION',
        'DJIBOUT': 'DJIBOUTI',
        'DRC': 'DEMOCRATIC REPUBLIC OF THE CONGO',
        'TAIWAN': 'CHINA, TAIWAN PROVINCE OF CHINA',
        'UKRAIN': 'UKRAINE',
        'MALT': 'MALTA',
        'CZECH REPUBLIC': 'CZECHIA',
        'TRINIDAD TOBACCO': 'TRINIDAD AND TOBAGO',
        'CAPE VERDE': 'CABO VERDE',
        'SAUD ARABIA': 'SAUDI ARABIA',
        'MACEDONIA': 'TFYR MACEDONIA',
        'BOSNIA': 'BOSNIA AND HERZEGOVINA',
        'VIETNAM': 'VIET NAM',
        'SOMALI': 'SOMALIA'
    }

    df.replace(input_countries_to_official, inplace=True)

    df_working = pd.merge(df, df_countries, on='country', how='left')
    return df_working


if __name__ == '__main__':
    df = pd.read_csv('data/Train.csv')
    df = pd.read_csv('data/Test.csv')
    extend_train_test_with_countries(df)

