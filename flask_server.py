from flask import Flask, request, render_template, jsonify, make_response, request
import pandas as pd
import os
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		df = pd.read_csv(request.files.get('file'))
		
		df = df.sample(frac=0.05)
		removable_features = ['SOURCE', 'CENSUS_TRACT', 'CENSUS_BLOCK', 'SQUARE', 'USECODE', 'Unnamed: 0', 'X', 'Y']
		df = df.drop(removable_features, axis=1)
		if 'PRICE' in df:
			df = df.drop('PRICE', axis=1)
		df['SALEDATE'] = pd.to_datetime(df['SALEDATE'])
		df['SALE_YR'] = df['SALEDATE'].dt.year; df['SALE_MONTH'] = df['SALEDATE'].dt.month
		df = df.drop('SALEDATE', axis=1)
		df = df.drop('GIS_LAST_MOD_DTTM', axis=1)

		
		cols_missing_morethan40 = ['LIVING_GBA','CMPLX_NUM' ,'YR_RMDL' ,'FULLADDRESS' ,'STATE' ,'NATIONALGRID' ,
                           'CITY' ,'STORIES' ,'KITCHENS' ,'ROOF' ,'INTWALL' ,'STRUCT' ,'STYLE' ,'EXTWALL' ,
                           'CNDTN' ,'GBA' ,'NUM_UNITS' ,'GRADE']
		df = df.drop(cols_missing_morethan40, axis=1)
		if 'ASSESSMENT_SUBNBHD' in df:
			df = df.drop('ASSESSMENT_SUBNBHD', axis=1)
		df = df.dropna(how='any')
		df = df.drop(['LATITUDE','LONGITUDE','ZIPCODE'], axis=1)

		df = df[df.FIREPLACES<=11]
		df = df[df.AYB>=1825]
		df = df[df.EYB>=1900]
		df = df[df.LANDAREA<=50000]

		df = pd.concat([df, pd.get_dummies(df[['AC','QUALIFIED','WARD','QUADRANT']])], axis=1)
		df = df.drop(['AC','QUALIFIED','WARD','QUADRANT'], axis=1)
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		df['HEAT']=le.fit_transform(df['HEAT'])
		df['ASSESSMENT_NBHD']=le.fit_transform(df['ASSESSMENT_NBHD'])
		
		loaded_model = joblib.load('model.pkl')
		predictions = loaded_model.predict(df)
		serial = np.arange(1,len(predictions)+1,1)
		result = dict(zip(serial, predictions))
		#result = df.shape
		
		Table = []
		for key, value in result.items():
			temp = []
			temp.extend([key,value])  #Note that this will change depending on the structure of your dictionary
			Table.append(temp)
		
		return render_template('upload.html', table=Table)
	return render_template('upload.html')

if __name__ == '__main__':
	app.run(debug=True)