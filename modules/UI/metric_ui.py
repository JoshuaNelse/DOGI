import streamlit as st
import pandas as pd
import numpy as np
import os
from modules.util import br, md


def get_whole_metrics_page():
    br(2)
    md('''
        ### Table of Contents
        * The Dataset
        * Data features
        * Model Selection
        * Model Evaluation
        ''')
    md('''## The Dataset''')
    md('Data sourced from kaggle: https://www.kaggle.com/c/dog-breed-identification')
    md('The dataset has roughly 10,000 training images '
       '(having associated labels) and 10,000 test images (with no labels).')

    st.code('''
train_filenames = [f"data/train/{fname}" for fname in os.listdir('data/train')]
test_filenames = [f"data/test/{fname}" for fname in os.listdir('data/test')]
st.write((len(train_filenames), len(test_filenames)))''')

    md('`(10222, 10357)`')

    with st.echo():
        labels_csv = pd.read_csv('data/labels.csv')
        st.write(labels_csv[:5])

    st.code('''
# example of an image (random index 45)
st.image(train_filenames[45])
    ''')
    st.image('resources/random_image_metric.png', use_column_width=True)
    md('''## Data features''')
    md('''There are 120 unique breeds (labels) and a reasonably even distribution amongst breeds.''')

    with st.echo():
        unique_breeds = np.unique(labels_csv.breed.to_numpy())
        st.write(unique_breeds.size)

    with st.echo():
        # Plotting the count for each dog breed available in this dataset (all 120)
        st.bar_chart(labels_csv.breed.value_counts())
    md('Note: hover over or expand the above graph to see specific breeds more clearly.')

    with st.echo():
        # Checking the mean count of individual breed samples in the training data.
        st.write(labels_csv.breed.value_counts().mean())

    md('''## Model Selection''')
    md('Considering that the features (images) are unstructured data and we want to '
       'have a model that maps an image to one of 120 categories (labels), '
       'I will go with a transfer learning approach to create a sequential model '
       'for classification of these images')
    md('''
        Documentation/References:\n
        - Sequential Model (w/Keras API): [Read more](https://www.tensorflow.org/guide/keras/sequential_model)
        - mobilenet_v2 (transfer learning): [Read more](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4)
        ''')
    md('''## Model Evaluation''')
    md('''
            The first fit of the model will be on a subset of the total training set. 
            This will be done utilizing rough 1/10th or 1,000 training images.
        ''')
    br()
    md('''
        Below are the results of training the model with the above mentioned 1000 training images.
        This model training output shows that while the accuracy of the model on the testing data 
        reached 100% the accuracy on the validation data reached ~69%.
        ''')
    st.code('''
    Building model with: https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4
    Epoch 1/100
    25/25 [==============================] - 274s 11s/step - loss: 4.6283 - accuracy: 0.0850 - val_loss: 3.2848 - val_accuracy: 0.3150
    Epoch 2/100
    25/25 [==============================] - 4s 174ms/step - loss: 1.6449 - accuracy: 0.6862 - val_loss: 2.0161 - val_accuracy: 0.5500
    Epoch 3/100
    25/25 [==============================] - 4s 175ms/step - loss: 0.5749 - accuracy: 0.9312 - val_loss: 1.5843 - val_accuracy: 0.6000
    Epoch 4/100
    25/25 [==============================] - 4s 172ms/step - loss: 0.2573 - accuracy: 0.9850 - val_loss: 1.4001 - val_accuracy: 0.6550
    Epoch 5/100
    25/25 [==============================] - 4s 172ms/step - loss: 0.1493 - accuracy: 0.9937 - val_loss: 1.3168 - val_accuracy: 0.6650
    Epoch 6/100
    25/25 [==============================] - 4s 173ms/step - loss: 0.1019 - accuracy: 1.0000 - val_loss: 1.2755 - val_accuracy: 0.6650
    Epoch 7/100
    25/25 [==============================] - 4s 173ms/step - loss: 0.0761 - accuracy: 1.0000 - val_loss: 1.2505 - val_accuracy: 0.6750
    Epoch 8/100
    25/25 [==============================] - 4s 173ms/step - loss: 0.0601 - accuracy: 1.0000 - val_loss: 1.2281 - val_accuracy: 0.6800
    Epoch 9/100
    25/25 [==============================] - 4s 171ms/step - loss: 0.0495 - accuracy: 1.0000 - val_loss: 1.2156 - val_accuracy: 0.6800
    Epoch 10/100
    25/25 [==============================] - 4s 174ms/step - loss: 0.0417 - accuracy: 1.0000 - val_loss: 1.2009 - val_accuracy: 0.6850
    Epoch 11/100
    25/25 [==============================] - 4s 172ms/step - loss: 0.0358 - accuracy: 1.0000 - val_loss: 1.1932 - val_accuracy: 0.6900
    Epoch 12/100
    25/25 [==============================] - 4s 175ms/step - loss: 0.0311 - accuracy: 1.0000 - val_loss: 1.1802 - val_accuracy: 0.6850
    Epoch 13/100
    25/25 [==============================] - 4s 178ms/step - loss: 0.0275 - accuracy: 1.0000 - val_loss: 1.1739 - val_accuracy: 0.6900
    Epoch 14/100
    25/25 [==============================] - 4s 173ms/step - loss: 0.0246 - accuracy: 1.0000 - val_loss: 1.1638 - val_accuracy: 0.6850''')
    br(2)
    md('''
        Here are some specific predictions performed by this model. 
        You can see that while some of the predictions have a decent amount of confidence, 
        a few of the examples are only "correct" by a small margin (if not guess incorrectly altogether).
        \nNote: the `green` bar denotes the "correct" label.
        ''')
    st.image('resources/sample_results1.png')
    br(2)
    md('''
        Now to fully train the model (train on all 10,000 images). 
        Notice that there is not an accuracy value listed for a validation set. 
        This is because the previously used validation set was actually 
        a subset of the training data. Since we will use the training data in its entirety,
         This model will be tested against images with no `labels` 
        ''')
    br()
    md('''
        From the output below we can see that the  model never quite reached 100% accuracy on the training set. 
        This is because the training was stopped to prevent over fitting. 
        Hopefully this leads to a model with an understanding than can be used in a more generalized sense. 
        ''')
    st.code('''
    Epoch 1/100
    320/320 [==============================] - 43s 127ms/step - loss: 2.3794 - accuracy: 0.4820
    Epoch 2/100
    320/320 [==============================] - 39s 122ms/step - loss: 0.3915 - accuracy: 0.8868
    Epoch 3/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.2313 - accuracy: 0.9430
    Epoch 4/100
    320/320 [==============================] - 38s 119ms/step - loss: 0.1494 - accuracy: 0.9652
    Epoch 5/100
    320/320 [==============================] - 39s 121ms/step - loss: 0.0969 - accuracy: 0.9824
    Epoch 6/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.0717 - accuracy: 0.9895
    Epoch 7/100
    320/320 [==============================] - 38s 119ms/step - loss: 0.0555 - accuracy: 0.9928
    Epoch 8/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.0411 - accuracy: 0.9963
    Epoch 9/100
    320/320 [==============================] - 39s 120ms/step - loss: 0.0340 - accuracy: 0.9965
    Epoch 10/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.0272 - accuracy: 0.9986
    Epoch 11/100
    320/320 [==============================] - 38s 118ms/step - loss: 0.0253 - accuracy: 0.9979
    Epoch 12/100
    320/320 [==============================] - 38s 119ms/step - loss: 0.0227 - accuracy: 0.9976
    Epoch 13/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.0160 - accuracy: 0.9993
    Epoch 14/100
    320/320 [==============================] - 38s 117ms/step - loss: 0.0150 - accuracy: 0.9988
    Epoch 15/100
    320/320 [==============================] - 38s 119ms/step - loss: 0.0211 - accuracy: 0.9981
    Epoch 16/100
    320/320 [==============================] - 39s 121ms/step - loss: 0.0123 - accuracy: 0.9992
    Epoch 17/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.0094 - accuracy: 0.9996
    Epoch 18/100
    320/320 [==============================] - 38s 119ms/step - loss: 0.0119 - accuracy: 0.9989
    Epoch 19/100
    320/320 [==============================] - 38s 120ms/step - loss: 0.0094 - accuracy: 0.9992
    Epoch 20/100
    320/320 [==============================] - 39s 120ms/step - loss: 0.0107 - accuracy: 0.9991
        ''')
    br(2)
    md('''
        Now time to see how the model holds up against images of dogs from the internet. 
        One concern could be that these images from the internet might already be in this dataset. 
        Let's hope that is not the case! (to mitigate the odds of this being a problem 
        I select "newer" images of dogs from unsplash. Reference: https://unsplash.com/images/animals/dog
        ''')
    for image in os.listdir('resources/internet_dogs/'):
        st.image(f"resources/internet_dogs/{image}")

    md('''
        The model appears to be predicting in a more confident manner and 
        where it does get confused it seems to have a hard time between two or more similar breeds. 
        There is certainly an improvement in this model 
        that used 10,000 images over the initial one with only 1,000
        ''')
    md('''
        ## Yay for doggies!
        ''')
