# dash and plotly
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, callback, State
import plotly.graph_objects as go

# dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# sklearn and model
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import catboost as cb
import shap

# ------------------------------------------------------------------------------

class AddBMI(BaseEstimator, TransformerMixin):
    # add BMI
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        meter = X['height']/100
        X['bmi'] = round(X['weight'] / (meter ** 2),2)
        return X


class AddBMILevel(BaseEstimator, TransformerMixin):
    # add BMI level
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['bmi_level'] = np.nan
        X.loc[X['bmi']<18.50, 'bmi_level'] =  1 #'underweight'
        X.loc[(X['bmi']>=18.50) & (X['bmi'] <25.00), 'bmi_level'] = 2 #'healthy'
        X.loc[(X['bmi']>=25.00) & (X['bmi'] <30), 'bmi_level'] = 3 #'overweight'
        X.loc[X['bmi']>=30, 'bmi_level'] = 4 #'obesity'
        return X

# load model
model = joblib.load('cat_pipeline_colab.pkl')

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# allows callbacks to reference components that may be dynamically created and added later. 
app.config.suppress_callback_exceptions = True

sidebar_config = {
	'background-color': '#DCDCDC'
}

####################################################################
# For debugging
# base_value = 64
# preds = 97
# new_features = np.array(['other drugs', 'weight', 'alcoholic', 'heart disease', 'addiction',
#        'medications', 'surgery', 'blood pressure', 'smoker', 'gender',
#        'occupation danger', 'height', 'family cancer', 'lifestyle danger',
#        'immune', 'bmi', 'family choles', 'family heart disease',
#        'opioids', 'diabetes', 'cannabis', 'nicotin', 'cholesterol',
#        'asthma', 'bmi level'])
# values = np.array([-6.95882916, -6.89249988,  6.33020007,  5.0088834 ,  4.91509902,
#         3.99006289,  3.59020566,  3.58493215, -2.70542553,  2.26298549,
#         1.54600785,  1.47765055,  1.35097317,  0.75876114,  0.67341692,
#         0.53442746, -0.45865878,  0.42839709,  0.4239629 ,  0.36946292,
#         0.33057127,  0.15416676,  0.11876897,  0.02453146,  0.        ])

# fig = go.Figure()

# for i in range(len(new_features)):
#         fig.add_trace(go.Bar(
#             x=[new_features[i]],
#             y=[values[i]],
#             marker_color='red' if values[i] < 0 else 'blue',
#             name=f"{new_features[i]} ({values[i]:.3f})"
#         ))

# # Layout settings
# fig.update_layout(
#     title='SHAP Waterfall Plot',
#     xaxis_title="Feature",
#     yaxis_title="SHAP Value Impact",
#     showlegend=False
# )

# text = ['-------------------------------------------------',
# 		html.Br(),
# 		html.Span('Attention! How to read SHAP Value', style={'font-weight': 'bold'}),
# 		html.Br(),
# 		'1) If your age are longer than the average age then you should live longer than average people',
# 		html.Br(),
# 		'2) The number score of the factors indicate how much that factor contribute to your age if it is positive sign \
# 		then that factor contribute positive impact to your age and vice versa.',
# 		html.Br(),
# 		'-------------------------------------------------',
# 		html.Br(),
# 		html.Span(f'Your predicted age of death is {preds}.',style={'color':'red'}),
# 		html.Br(),
# 		html.Span('While the average age from the training data is 64.',style={'color':'blue'}),
# 		html.Br(),
# 		html.Br(),
# 		'The factors (most to least) which effect your age are:',
# 		html.Br(),
# 		html.Br(),
# 		*[html.P(f'{feature}, {value:.2f}') for feature,value in zip(new_features, values)]]

####################################################################

app.layout = dbc.Container(
	[
		dbc.Row(
			[
				html.H1('Predict Age of Death')
			]
		),
		dbc.Row(
			[
				html.A(
						children='Reference: Model Training in Kaggle: version 2 CATBOOST', 
						href='https://www.kaggle.com/code/natchaphonkamhaeng/predict-age-of-death',
						target='_blank'
					),
				html.A(
						children='Reference: Data from Kaggle',
						href='https://www.kaggle.com/datasets/joannpineda/individual-age-of-death-and-related-factors',
						target='_blank'
					),
				html.A(
						children='Reference: Github Code',
						href='https://github.com/Natcha-Phonkamhaeng/predict-age-of-death',
						target='_blank'
					)
			]
		),
		html.Br(),
		dbc.Row(
			[
				dbc.Col(
					[
						html.P('Weight'), # weight
						dcc.Input(
								id='input-weight',
								value='',
								type='number', 
								placeholder='weight in Kilogram', 
								style={'marginRight':'10px'}
							),
						html.Hr(),

						html.P('Gender'), # sex
						dcc.Dropdown(
								options=[
									{'label': 'Male', 'value': 'm'}, # label is the users see, value is what pass to callback
									{'label': 'Female', 'value': 'f'}
								], 
								value='', 
								id='dropdown-sex',
								style={'width':'59%'}
							),
						html.Hr(),

						html.P('Height'), # height
						dcc.Input(
								id='input-height',
								value='', 
								type='number', 
								placeholder='height in Centimeter'
							),
						html.Hr(),

						html.P('Systolic blood pressure (mmHg)'), # sys_bp
						dcc.Input(
								id='input-bp',
								value='',
								type='number', 
								placeholder='blood pressure in mmHG'
							),
						html.Hr(),

						html.P('Smoke'), # smoker
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='do you smoke?',
								id='dropdown-smoker',
								style={'width': '59%'}
							),
						html.Hr(),

						html.P('Nicotin Products'), # nic_other
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='do you use other nicotin products?',
								id='dropdown-nicotin',
								style={'width': '73%'}
							),
						html.Hr(),

						html.P('Number of Medications'), # num_meds
						dcc.Input(
								id='input-num-meds',
								value='', 
								type='number', 
								placeholder='How many medication pills you take?', 
								style={'width': '54%'}
							),
						html.Hr(),
						
						html.P('Occupation Danger'), # occup_danger
						dcc.Dropdown(
								options=[
									{'label': '1 - least danger', 'value': 1},
									{'label': '2 - moderate danger', 'value': 2},
									{'label': '3 - grave danger', 'value': 3}
								],
								value='',
								placeholder='Is your occupation risky for death?',
								id='dropdown-occup_danger',
								style={'width': '73%'}
							),
						html.Hr(),

						html.P('Lifestyle Danger'), # Is_danger
						dcc.Dropdown(
								options=[
									{'label': '1 - least danger', 'value': 1},
									{'label': '2 - moderate danger', 'value': 2},
									{'label': '3 - grave danger', 'value': 3}
								],
								value='',
								placeholder='Is your lifestyle danger?',
								id='dropdown-is_danger',
								style={'width': '70%'}
							),
						html.Hr(),

						html.P('Cannabis (กัญชา)'), # cannabis
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Do you use cannabis?',
								id='dropdown-cannabis',
								style={'width': '70%'}
							),
						html.Hr(),

						html.P('Opioids (ฝิ่น)'), # opioids
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Do you use opioids?',
								id='dropdown-opioids',
								style={'width': '70%'}
							),
						html.Hr(),

						html.P('Drugs'), # other_drugs
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Do you use other drugs?',
								id='dropdown-other_drugs',
								style={'width': '70%'}
							),
						html.Hr(),

						html.P('Alcoholic Drinks'), # drinks_aweek
						dcc.Input(
								id='input-drinks_aweek',
								value='', 
								type='number', 
								placeholder='How many alcohol you drink per week?', 
								style={'width': '58%'}
							),
						html.Hr(),

						html.P('Drugs Addition'), # addiction
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Have you ever had drug addition?',
								id='dropdown-addicion',
								style={'width': '72%'}
							),
						html.Hr(),

						html.P('Number of Surgery'), # major_surgery_num
						dcc.Input(
								id='input-major_surgery_num',
								value='', 
								type='number', 
								placeholder='How many serious surgeries performed?', 
								style={'width': '58%'}
							),
						html.Hr(),

						html.P('Diabetes (เบาหวาน)'), # diabetes
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Do you have diabetes?',
								id='dropdown-diabetes',
								style={'width': '70%'}
							),
						html.Hr(),

						html.P('Heart Disease'), # hds
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Have you ever had heart disease?',
								id='dropdown-hds',
								style={'width': '72%'}
							),
						html.Hr(),

						html.P('Cholesterol Level'), # cholesterol
						dcc.Input(
								id='input-cholesterol',
								value='', 
								type='number', 
								placeholder='Your cholesterol level', 
								style={'width': '40%'}
							),
						html.Hr(),

						html.P('Asthma Attack (หอบหืด)'), # asthma
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Have you ever had asthma attack?',
								id='dropdown-asthma',
								style={'width': '72%'}
							),
						html.Hr(),

						html.P('Immune Deficiency (ภูมิคุ้มกันบกพร่อง)'), # immune_defic
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Do you have immnological deficiency?',
								id='dropdown-immune_defic',
								style={'width': '77%'}
							),
						html.Hr(),

						html.P('Family Cancer'), # family_cancer
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Is member of your family have cancer?',
								id='dropdown-family_cancer',
								style={'width': '77%'}
							),
						html.Hr(),

						html.P('Family Heart Desease'), # family_heart_desease
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								# value='',
								placeholder='Is member of your family ever have heart desease?',
								id='dropdown-family_heart_desease',
								style={'width': '86%'}
							),
						html.Hr(),

						html.P('Family Cholesterol'), # family_cholesterol
						dcc.Dropdown(
								options=[
									{'label': 'No', 'value': 'n'},
									{'label': 'Yes', 'value': 'y'}
								],
								value='',
								placeholder='Is member of your family have high cholesterol?',
								id='dropdown-family_cholesterol',
								style={'width': '86%'}
							),
						html.Hr(),

						html.Button('Submit', id='button-submit', n_clicks=0),
						html.P(children='', id='submit-text')

					],
					style=sidebar_config
				),
				dbc.Col(
					[
						dcc.Loading(
							# id='loading',
							type='circle',
							children=[dcc.Graph(id='shap-waterfall-plot', style={'display': 'none'})],
							# children=[dcc.Graph(figure=fig, id='shap-waterfall-plot', style={'display': 'block'})] # only for debugging
							),
						html.H3(children='', id='shap-diag'),
						# html.H3(children='Diagnostics', id='shap-diag'), # only for debugging
						html.P(children='', id='shap-diag-text')
						# html.P(children=text, id='shap-diag-text') # only for debugging
					]
				)
			]
		)
	]
)

# Function to convert SHAP waterfall plot to Plotly figure
def shap_waterfall_to_plotly(shap_values):
    base_value = shap_values.base_values[0]
    features = shap_values.feature_names
    values = shap_values.values[0]
    
    # Sort features by absolute SHAP value
    sorted_indices = np.argsort(np.abs(values))[::-1]
    features = np.array(features)[sorted_indices]
    values = np.array(values)[sorted_indices]
    
    # Create feature name and input name from user
    feature_dict = {
	    'remainder__weight': 'weight',
	    'ohe__sex_m': 'gender',
	    'ohe__sex_f': 'gender',
	    'remainder__height': 'height',
	    'remainder__sys_bp': 'blood pressure',
	    'ohe__smoker_y': 'smoker',
	    'ohe__smoker_n': 'smoker',
	    'ohe__nic_other_y': 'nicotin',
	    'ohe__nic_other_n': 'nicotin',
	    'remainder__num_meds': 'medications',
	    'remainder__occup_danger': 'occupation danger',
	    'remainder__is_danger': 'lifestyle danger',
	    'ohe__cannabis_y': 'cannabis',
	    'ohe__cannabis_n': 'cannabis',
	    'ohe__opioids_y': 'opioids',
	    'ohe__opioids_n': 'opioids',
	    'ohe__other_drugs_y': 'other drugs',
	    'ohe__other_drugs_n': 'other drugs',
	    'remainder__drinks_aweek': 'alcoholic',
	    'ohe__addiction_y': 'drugs addiction',
	    'ohe__addiction_n': 'drugs addiction',
	    'remainder__major_surgery_num': 'surgery',
	    'ohe__diabetes_y': 'diabetes',
	    'ohe__diabetes_n': 'diabetes',
	    'ohe__hds_y': 'heart disease',
	    'ohe__hds_n': 'heart disease',
	    'remainder__cholesterol': 'cholesterol',
	    'ohe__asthma_y': 'asthma',
	    'ohe__asthma_n': 'asthma',
	    'ohe__immune_defic_y': 'immune',
	    'ohe__immune_defic_n': 'immune',
	    'ohe__family_cancer_y': 'family cancer',
	    'ohe__family_cancer_n': 'family cancer',
	    'ohe__family_heart_disease_y': 'family heart disease',
	    'ohe__family_heart_disease_n': 'family heart disease',
	    'ohe__family_cholesterol_y': 'family choles',
	    'ohe__family_cholesterol_n': 'family choles',
	    'remainder__bmi_level': 'bmi level',
	    'remainder__bmi': 'bmi'
    }
	
	# Mapping feature name and input name from user
	# get func will return value if value does not found it will not raise value error instead it will return key
    vec_replace = np.vectorize(lambda x: feature_dict.get(x,x))
    new_features = vec_replace(features)

    # Create Waterfall Plot
    fig = go.Figure()
    
    for i in range(len(new_features)):
        fig.add_trace(go.Bar(
            x=[new_features[i]],
            y=[values[i]],
            marker_color='red' if values[i] < 0 else 'blue',
            name=f"{new_features[i]} ({values[i]:.3f})"
        ))

    # Add base value as a reference line
    # fig.add_hline(y=base_value, line_dash="dot", annotation_text=f"Base Value: {base_value:.2f}")

    # Layout settings
    fig.update_layout(
        title='SHAP Waterfall Plot',
        xaxis_title="Feature",
        yaxis_title="SHAP Value Impact",
        showlegend=False
    )
    
    return fig, new_features, values


@callback(
	Output(component_id='shap-waterfall-plot', component_property='figure'),
	Output(component_id='shap-waterfall-plot', component_property='style'),
	Output(component_id='shap-diag', component_property='children'),
	Output(component_id='shap-diag-text', component_property='children'),
	Output(component_id='submit-text', component_property='children'),
	Input(component_id='button-submit', component_property='n_clicks'), # release the input when hit submit button
	State(component_id='input-weight', component_property='value'), # hold the input untill hit submit botton
	State(component_id='dropdown-sex', component_property='value'),
	State(component_id='input-height', component_property='value'),
	State(component_id='input-bp', component_property='value'),
	State(component_id='dropdown-smoker', component_property='value'),
	State(component_id='dropdown-nicotin', component_property='value'),
	State(component_id='input-num-meds', component_property='value'),
	State(component_id='dropdown-occup_danger', component_property='value'),
	State(component_id='dropdown-is_danger', component_property='value'),
	State(component_id='dropdown-cannabis', component_property='value'),
	State(component_id='dropdown-opioids', component_property='value'),
	State(component_id='dropdown-other_drugs', component_property='value'),
	State(component_id='input-drinks_aweek', component_property='value'),
	State(component_id='dropdown-addicion', component_property='value'),
	State(component_id='input-major_surgery_num', component_property='value'),
	State(component_id='dropdown-diabetes', component_property='value'),
	State(component_id='dropdown-hds', component_property='value'),
	State(component_id='input-cholesterol', component_property='value'),
	State(component_id='dropdown-asthma', component_property='value'),
	State(component_id='dropdown-immune_defic', component_property='value'),
	State(component_id='dropdown-family_cancer', component_property='value'),
	State(component_id='dropdown-family_heart_desease', component_property='value'),
	State(component_id='dropdown-family_cholesterol', component_property='value'),
	prevent_initial_call=True # Prevents callback from running on page load, will run only when user trigger
)
def update_output(n_clicks, weight, sex, height, bp, smoker, nicotin, num_meds, occup_danger,
				is_danger, cannabis, opioids, other_drugs, drinks_aweek, addiction, major_surgery,
				diabetes, hds, choles, asthma, immune, cancer, heart_desease, fam_choles):
	# 23 features
	my_list = [weight, sex, height, bp, smoker, nicotin, num_meds, occup_danger,
				is_danger, cannabis, opioids, other_drugs, drinks_aweek, addiction, major_surgery,
				diabetes, hds, choles, asthma, immune, cancer, heart_desease, fam_choles] 
	print(n_clicks)
	print(my_list)
	# checking if user select all dropdowns before click submit button
	if n_clicks > 0: # if user click submit button
		if None in my_list: # there's one dropdown missing
			return "","","","","Please fill all dropdown before submitting"
		elif not None in my_list: # user select all dropdown
			df = pd.DataFrame.from_dict(
					{
						'weight': [weight],
						'sex': [sex],
						'height': [height],
						'sys_bp': [bp],
						'smoker': [smoker],
						'nic_other': [nicotin],
						'num_meds': [num_meds],
						'occup_danger': [occup_danger],
						'is_danger': [is_danger],
						'cannabis': [cannabis],
						'opioids': [opioids],
						'other_drugs': [other_drugs],
						'drinks_aweek': [drinks_aweek],
						'addiction': [addiction],
						'major_surgery_num': [major_surgery],
						'diabetes': [diabetes],
						'hds': [hds],
						'cholesterol': [choles],
						'asthma': [asthma],
						'immune_defic': [immune],
						'family_cancer': [cancer],
						'family_heart_disease': [heart_desease],
						'family_cholesterol': [fam_choles]
					}
				)
			
			y_preds = model.predict(df)
			y_preds = int(y_preds[0])

			# preprocess step before SHAP
			# extract preprocess step from pipeline
			pipeline_preprocess = model.named_steps['pipeline'][2]
			
			# get feature name from OHE
			feature_names = pipeline_preprocess.get_feature_names_out() 
			
			# perform OHE on df
			df_OHE = pipeline_preprocess.transform(df) 
			
			# create dataframe from df OHE
			df_transformed = pd.DataFrame(df_OHE, columns=feature_names) 

			# SHAP
			# Extract only model from Pipeline
			cat_model = model.named_steps['cat']

			# Initialize the SHAP explainer
			explainer = shap.Explainer(cat_model)

			# Compute SHAP values
			shap_values = explainer(df_transformed)

			new_feature_name = shap_waterfall_to_plotly(shap_values)[1] # return array of new feature names base on user input
			new_values = shap_waterfall_to_plotly(shap_values)[2] # return array of new values after sorted descending 

			text = ['-------------------------------------------------',
				html.Br(),
				html.Span('Attention! How to read SHAP Value', style={'font-weight': 'bold'}),
				html.Br(),
				'1) If your age are longer than the average age then you should live longer than average people',
				html.Br(),
				'2) The number score of the factors indicate how much that factor contribute to your age if it is positive sign \
				then that factor contribute positive impact to your age and vice versa.',
				html.Br(),
				'-------------------------------------------------',
				html.Br(),
				html.Span(f'Your predicted age of death is {y_preds}.',style={'color':'red'}),
				html.Br(),
				html.Span('While the average age from the training data is 64.',style={'color':'blue'}),
				html.Br(),
				html.Br(),
				'The factors (most to least) which effect your age are:',
				html.Br(),
				html.Br(),
				*[html.P(f'{feature}, {value:.2f}') for feature,value in zip(new_feature_name, new_values)]]
			
			return shap_waterfall_to_plotly(shap_values)[0], \
					{'display': 'block'}, \
					'Diagnostics', \
					text,\
					'Submitted successfully, please scroll up to see result'
	return "","","","","please select all dropdown"


if __name__ == '__main__':
    app.run(debug=True)

