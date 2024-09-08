# ======================== | Imports | ========================

import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import joblib
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
import json
import requests # for get requests on url
import warnings
warnings.filterwarnings("ignore")
import os
import shap
import streamlit.components.v1 as components # to visualize shap plots

# Imports from Home page
# from Home import X_test, FLASK_URL

# URL parent du serveur Flask
FLASK_URL = "http://127.0.0.1:8500/"

# ======================== | Data Import | ========================

current_folder = os.getcwd()
basepath = os.path.join(current_folder, "Models")
y_test = pd.read_csv(os.path.join(basepath, "y_test_sample.csv"))
shap_values = pd.read_csv(os.path.join(basepath, "shap_values_sample.csv"))
columns = joblib.load('Models/columns.pkl')
data = pd.DataFrame(y_test, index=y_test.index).reset_index()

# ======================== | Initializations | ========================

# initialize variable viz for use in selection menu
viz = ""
# ======================== | Page title & sidebar | ========================

st.markdown("# Observation Group \U00002696")
st.sidebar.markdown("# Observation Group \U00002696")


# ======================== | Interactions, API calls and decode | ========================


#### API CALLS ####

# API call | GET X_test and cache it (heavy)
@st.cache_data
def load_X_test():
    url_X_test = FLASK_URL + "load_X_test/"
    response = requests.get(url_X_test)
    content = json.loads(response.content.decode('utf-8'))
    dict_X_test = content["X_test"]
    X_test = pd.DataFrame.from_dict(eval(dict_X_test), orient='columns')
    return X_test
X_test = load_X_test()

# API call | GET data (used to select customer idx)
@st.cache_data
def load_data():
    url_data = FLASK_URL + "load_data/"
    response = requests.get(url_data)
    content = json.loads(response.content.decode('utf-8'))
    dict_data = content["data"]
    data = pd.DataFrame.from_dict(eval(dict_data), orient='columns')
    return data
data = load_data()

# Retrieve previously selected value from Home page
if 'idx' in st.session_state:
    idx = st.session_state['idx']
    idx1 = st.sidebar.selectbox(
    "Select Credit File", 
    data.SK_ID_CURR, key = "idx2", index = int(data.loc[data["SK_ID_CURR"]==int(idx)].index[0]))
else:
    # Select Customer number SK_ID_CURR in data
    idx1 = st.sidebar.selectbox(
        "Select Credit File", 
        data.SK_ID_CURR, key = "idx2")
st.session_state['idx'] = idx1
idx = idx1

# Customer index in the corresponding array
data_idx = data.loc[data["SK_ID_CURR"]==int(idx)].index[0]

# API call | GET top_20
url_top_20 = FLASK_URL + "load_top_20/"
response = requests.get(url_top_20)
content = json.loads(response.content.decode('utf-8'))
top_20 = content["top_20"]
feat_top = content["feat_top"]
feat_tot = content["feat_tot"]

# API call | GET data for shap plots : type / index / shap_values :
url_cust_vs_group = FLASK_URL + "cust_vs_group/" + str(idx)
response = requests.get(url_cust_vs_group)
content = json.loads(response.content.decode('utf-8'))
decision = content["decision"]
base_value = content["base_value"]
shap_values1_idx = np.array(content["shap_values1_idx"])
dict_ID_to_predict = content["ID_to_predict"]
ID_to_predict = pd.DataFrame.from_dict(eval(dict_ID_to_predict), orient='columns')


# ======================== | Interactions Streamlit & Plots | ========================

# recall customer number
st.write(f"Customer number : {str(idx)} | Credit is " + decision)
st.write("Code_Gender : 1 = Female | 0 = Male")
# recall customer data
st.write(ID_to_predict)

if st.sidebar.checkbox('Show Feature Importance Analysis'):
    # Select type of feature explanation visualization
    viz = st.sidebar.selectbox("Select feature importance visualization", options=['global | feature importance', 'global | impact by feature', 'client | expected to predicted', 'client | probability of default', 'client | odds of default'])
    st.header(viz + ' :')
else : st.write(f"""To visualise total feature importance and specific risk analysis for customer {str(idx)}\n
    Please select "Select Feature importance visualization" on the sidebar.
    """)

# display Shap plots in Streamlit
def st_shap(plot, height=None):
    """ to display in Streamlit plots from Shap"""
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
def sample_feature_importance(index, type=viz):

    fig1 = plt.figure()

    if type=='client | probability of default':
        return st.write(st_shap(shap.force_plot(base_value, shap_values1_idx, ID_to_predict, link='logit'))) # choose between 'logit' or 'identity'
    elif type=='client | odds of default':
        return st.write(st_shap(shap.force_plot(base_value, shap_values1_idx, ID_to_predict)))
    elif type=='client | expected to predicted':
        shap.waterfall_plot(shap_values[int(index)], max_display=10)
        st.pyplot(fig1)    
    elif type=='global | impact by feature':
        shap.summary_plot(shap_values, X_test, max_display=10)
        st.pyplot(fig1)
    elif type=='global | feature importance':
        shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=columns)
        st.pyplot(fig1)
sample_feature_importance(data_idx)


if st.sidebar.checkbox('Show Observation vs Group'):
    # @st.cache(allow_output_mutation=True, show_spinner=True, suppress_st_warning=True)
    def obs_vs_group():
        st.header(f'20 first features = {round((feat_top/feat_tot)*100, 2)} % total importance :')

        # Show boxplot for each feature with original units
        # selection of 20 most explicative features
        
        sel = top_20
        width = 20
        height = ((len(sel)+1)/2)*2

        fig2 = plt.figure(figsize=(width, height))

        # fig2 = plt.subplot(2,1,2,figsize=(width, height))
        for i, c in enumerate(sel,1):
            chaine = 'Distribution de : ' + c
            ax = fig2.add_subplot((len(sel)+2)//2, 2, i)
            plt.title(chaine)
            sns.boxplot(x=X_test[c],
                        orient='h',
                        color='lightgrey',
                        notch=True,
                        flierprops={"marker": "o"},
                        boxprops={"facecolor": (.4, .6, .8, .5)},
                        medianprops={"color": "coral"},
                        ax=ax)

        # show customer ID values for each feature
            plt.scatter(ID_to_predict[c], c, marker = 'D', c='r', s=200)

        # scaling automatique ('notebook', 'paper', 'talk', 'poster')
        sns.set_context("talk")
        fig2.tight_layout()

        st.pyplot(fig2)

    obs_vs_group()
else : st.write(f"""To visualize customer {str(idx)} vs Group :\n
    Please select "Show Observation vs Group" on the sidebar.
    """)