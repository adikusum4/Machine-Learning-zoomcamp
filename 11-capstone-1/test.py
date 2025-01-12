#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

home = {
 'subdivision': 'The Hills at Southpoint',
 'total_living_area_sqft': 3100,
 'year_built': 2010,
 'bedrooms': 6,
 'full_baths': 3,
 'half_baths': 1,
 'property_type': 'Detached',
 'acres': '.26-.5 Acres',
 'approx_lot_sqft': 12500.52, 
 'approximate_acres': 0.280,
 'basement': 'Yes',
 'construction_type': 'Site Built',
 'days_on_market': 50,
 'fireplace': 2,
 'garage': 3,
 'hoa_1_fees_required': 'Yes',
 'internet_listing': 'Yes',
 'master_bedroom_1st_floor': 'Yes',
 'new_construction': 'No',
 'total_baths': 4,
 'zip': '27713',
 'inside_city': 'Yes',
 'elementary_school_1': 'Durham - Creekside',
 'high_school_1': 'Durham - Jordan',
 'middle_school_1': 'Durham - Githens',
 'restrictive_covenants': 'Yes',
 'age_house': 10,
 'closing_month': 9,
 'closing_day': 2,
 #'discount_price': ,
}


response = requests.post(url, json=home).json()
print()

print('The estimated value of the home is ${:,.2f}'.format(response['home_price']))
