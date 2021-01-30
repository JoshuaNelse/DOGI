import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.controller.nav_controller import Nav
from modules.service.image_service import ImageProcessor as ImgP
from modules.service.prediction_service import DOGI
from modules.UI.metric_ui import get_whole_metrics_page
from modules.util import br, md


####
# Side bar code for UI
####
st.sidebar.markdown('''
    # \- DOGI -
    ----
''')
st.sidebar.markdown('## Navigate:')
Nav.set_nav_control(st.sidebar.radio('', [
    Nav.BREED_CLASSIFIER,
    Nav.METRICS,
    Nav.ABOUT
]))


####
# Can probably move all Non UI function out of this file
####
@st.cache
def get_unique_breeds():
    labels_csv = pd.read_csv('data/labels.csv')
    return np.unique(labels_csv.breed.to_numpy())


####
# Main page code of UI
####
st.title(Nav.get_nav_control())
if Nav.get_nav_control() == Nav.BREED_CLASSIFIER:
    uploaded_file = st.file_uploader('What\'s that dog? Upload a picture here to find out!')
    if uploaded_file is not None:
        col1, col2 = st.beta_columns(2)
        col1.image(uploaded_file, use_column_width=True, use_column_height=True)
        prediction = DOGI.get_model().predict(
            DOGI.batch_for_prediction(ImgP.process_image(uploaded_file)))
        n = st.slider(value=10, label=f"Select number of top matches to view:",
                      min_value=5, max_value=20)
        plot_indexes = prediction.argsort().T[-n:][::-1].T[0]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set(title=f"Top {n} Predictions")
        ax.plot(get_unique_breeds()[plot_indexes], prediction.T[plot_indexes].T[0])
        plt.xticks(rotation='80', fontsize=14)
        st.pyplot(fig)

        col2.markdown(
            f"## {get_unique_breeds()[prediction.argmax()]}\n"
            f"#### {round(prediction.T[prediction.argmax()][0]*100, 2)}% Confidence\n"
        )
        col2.write('')
        if col2.button("Not right?", key='correction_button'):
            col2.write('Thanks for letting us know!')
#####
# Metrics
####
if Nav.get_nav_control() == Nav.METRICS:
    get_whole_metrics_page()

####
# About ?
####
if Nav.get_nav_control() == Nav.ABOUT:
    '''
    DOGI (Dog Idenitifier) was created to help you find out want breed that cute puppy in the window might be!
    '''
    br(2)
    md('No more are the days where you find yourself looking '
       'into a pair of heart melting eyes wondering, "What is that dog, '
       'and why is it so darn cute?!')
    st.image(
        'https://images.unsplash.com/photo-1564864963038-9789ab2438d7?ixlib=rb-1.2.1&ixid'
        '=MXwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHw%3D&auto=format&fit=crop&w=1500&q=80',
        use_column_width=True)
    br(1)
    md('Now (using DOGI) you can simply snap a picture of those adorable ball of fluff and quickly find out its breed!')

