import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import requests
import zipfile
import gdown
# import torch
import nltk
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dense, Dropout, concatenate
# from keras._tf_keras.keras.models import Model
# from keras._tf_keras.keras.optimizers import Adam
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Ensure matplotlib inline mode is active for Streamlit
plt.switch_backend('Agg')

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Function to clean text data
def clean_text(text):
    pattern = '|'.join([" et "," un ", " de ", " ce ", " le "," d'"," la "," ces "," cette ", " l' ", " les ", " d "])
    text = text.str.replace(r'<[^<>]*>', '', regex=True)
    text = text.str.replace('\d+', '', regex=True)
    text = text.str.replace('Ø', '', regex=True)
    text = text.str.replace("[\[\]\?!\(\)\/\-:]", ' ', regex=True)
    text = text.str.replace("n°", '', regex=True)
    text = text.str.replace(pattern, ' ', regex=True)
    return text

# Function to load and preprocess images
def load_images(image_folder, image_ids, dataset):
    images = []
    count = 0
    for image_id in image_ids:
        try:
            image_path = os.path.join(image_folder, f'image_{image_id}_product_{dataset[dataset["imageid"]==image_id]["productid"].values[0]}.jpg')
            image = Image.open(image_path)
            image = image.resize((100, 100))  # Resize image to a standard size
            image = np.array(image)
            images.append(image)
        except FileNotFoundError:
            print(os.path.join(image_folder, f'image_{image_id}_product_{dataset[dataset["imageid"]==image_id]["productid"].values[0]}.jpg'))
            count += 1
            print('COUNT: ', count)
    return np.array(images)

# Load data
@st.cache_data(ttl=7200, show_spinner=True)
def load_data():
    # X_train = pd.read_csv('/content/drive/MyDrive/Rakuten_Project/X_train_update.csv')
    # X_test = pd.read_csv('/content/drive/MyDrive/Rakuten_Project/X_test_update.csv')
    # Y_train = pd.read_csv('/content/drive/MyDrive/Rakuten_Project/Y_train_CVw08PX.csv')

    X_train = pd.read_csv('https://drive.google.com/uc?id=1VXeunM_edswUl7Z6a7W7TbomhZW5SqlL')
    X_test_processed = pd.read_csv('https://drive.google.com/uc?id=101Cs4bHNbKADAQQq8dY7A1y4F2nDZ4Bt')
    Y_train = pd.read_csv('https://drive.google.com/uc?id=1jiScN4oux-AD08UILaadHy4vatYCOXWW')

    # Remove specific image IDs from X_test
    # image_ids = [1275488821, 1269512078, 1302547286, 1276746131, 1167984102, 873809748, 1262465799,
    #              1254969012, 1289490482, 1234540665, 1089420674, 1023836168, 1106280607, 1204250136]
    # X_test_processed = X_test[~X_test['imageid'].isin(image_ids)]

    # Rename columns for consistency
    # X_train.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    Y_train.rename(columns={"Unnamed: 0": "ID"}, inplace=True)
    # X_test_processed.rename(columns={"Unnamed: 0": "ID"}, inplace=True)

    # Clean text data
    # X_train['designation'] = clean_text(X_train['designation'])
    # X_test_processed['designation'] = clean_text(X_test_processed['designation'])

    # images_zip = 'https://drive.google.com/uc?id=1v8VS-lvgruHsRX-CNbjUEawhzDfbXtoj'

    # with zipfile.ZipFile(images_zip, 'r') as zip_ref:
    #     zip_ref.extractall('./images')

    file_url = 'https://drive.google.com/uc?id=1v8VS-lvgruHsRX-CNbjUEawhzDfbXtoj'
    local_filename = 'images.zip'

    if not os.path.exists(local_filename):
        try:
            st.write('Starting downloading images zip file...')
            gdown.download(file_url, 'images.zip', quiet=False)
            st.write("Zip file downloaded successfully!")
            st.write("Extracting contents from the zip file...")

            with zipfile.ZipFile(local_filename, 'r') as zip_ref:
                zip_ref.extractall('./images')

            st.write("Zip file extracted successfully!")
            st.write("Contents are extracted and ready to use.")

        except requests.exceptions.RequestException as e:
            st.write(f"Error downloading file: {e}")

        except zipfile.BadZipFile:
            st.write(f"Bad Zip File: {local_filename} is not a valid zip file")
    else:
        st.write('Contents are extracted and ready to use')
    return X_train, X_test_processed, Y_train

# Function to preprocess image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((100, 100))  # Resize image to a standard size
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize
    return image_array

# @st.cache(allow_output_mutation=True)
def train_test_split_for_model(train_images, train_data, tfidf):

    train_data['designation'].fillna('', inplace=True)

    train_data_copy = train_data.copy()

    designation_train_tfidf = tfidf.fit_transform(train_data['designation']).toarray()

    train_data_copy.drop(['productid', 'description', 'designation', 'imageid', 'ID', 'Unnamed: 0'], axis=1, inplace=True)

    train_features = np.concatenate((train_images.reshape(-1, 100 * 100 * 3),
                                    designation_train_tfidf.reshape(-1, 1000)),
                                    axis=1)

    train_image_features = train_features[:, :100*100*3].reshape(-1, 100, 100, 3)

    train_tfidf_features_designation = train_features[:, 100*100*3:]

    features_image_training, features_image_testing, features_text_training, features_text_testing, y_train, y_test = train_test_split(
        train_image_features,
        train_tfidf_features_designation,
        train_data,
        test_size=0.2,
        random_state=42
    )
    return features_image_training, features_image_testing, features_text_training, features_text_testing, y_train, y_test

@st.cache_data(ttl=3600, show_spinner=False)
def compile_model(features_image_training, features_text_training, y_train):
    image_input = Input(shape=(100, 100, 3), name='image_input')
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    image_output = Flatten()(maxpool2)
    image_model = Model(inputs=image_input, outputs=image_output)

    designation_input = Input(shape=(1000,), name='designation_input')
    dense1_text = Dense(256, activation='relu')(designation_input)
    text_model = Model(inputs=designation_input, outputs=dense1_text)

    combined_input = concatenate([image_model.output, text_model.output])
    dropout = Dropout(0.5)(combined_input)
    final_output = Dense(27, activation='softmax')(dropout)

    # Define and compile the final model
    final_model = Model(inputs=[image_model.input, text_model.input], outputs=final_output)
    optimizer = Adam(learning_rate=0.006)
    final_model.compile(optimizer=optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    history = final_model.fit(
        {'image_input': features_image_training, 'designation_input': features_text_training},
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )

    # loss, accuracy = final_model.evaluate(
    #     {'image_input': features_image_testing, 'designation_input': features_text_testing},
    #     y_test
    # )

    # Define the model

    # image_input = Input(shape=(100, 100, 3), name='image_input')
    # conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
    # maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(maxpool1)
    # maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # image_output = Flatten()(maxpool2)

    # designation_input = Input(shape=(1000,), name='designation_input')

    # # Concatenate the outputs
    # concatenated = concatenate([image_output, designation_input])

    # dense1 = Dense(256, activation='relu')(concatenated)
    # dropout = Dropout(0.5)(dense1)
    # output = Dense(27, activation='softmax')(dropout)

    # optimizer = Adam(learning_rate=0.006)

    # # Define and compile the model
    # model = Model(inputs=[image_input, designation_input], outputs=output)
    # model.compile(optimizer=optimizer,
    #             loss='sparse_categorical_crossentropy',
    #             metrics=['accuracy'])

    # # st.write('FIT:  =>  ', features_image_training.shape, '\nFTT:  =>  ', features_text_training.shape, '\nYT :  =>  ', y_train.shape)

    # # Train the model
    # history = model.fit(
    #     {'image_input': features_image_training, 'designation_input': features_text_training},
    #     y_train,
    #     epochs=20,
    #     batch_size=64,
    #     validation_split=0.2
    # )
    return final_model, history

# Function to display countplot
def display_countplot(train_data):
    plt.figure(figsize=(10, 8))
    sns.countplot(y='prdtypecode', data=train_data)
    plt.title('Product Type Code Distribution')
    plt.xlabel('Count')
    plt.ylabel('Product Type Code')
    st.pyplot()

def classification_report_bar_plots(model, features_image_testing, features_text_testing, y_test, le):
    y_pred = model.predict({'image_input': features_image_testing, 'designation_input': features_text_testing})
    y_pred_classes = np.argmax(y_pred, axis=1)
    class_report = classification_report(y_test, y_pred_classes, target_names=le.classes_.astype(str), output_dict=True)

    # Convert to DataFrame
    class_report_df = pd.DataFrame(class_report).transpose()

    # Plot precision, recall, and f1-score
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.barplot(x=class_report_df.index, y=class_report_df['precision'], palette="viridis")
    plt.title('Precision by Class')
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 2)
    sns.barplot(x=class_report_df.index, y=class_report_df['recall'], palette="viridis")
    plt.title('Recall by Class')
    plt.xticks(rotation=90)

    plt.subplot(1, 3, 3)
    sns.barplot(x=class_report_df.index, y=class_report_df['f1-score'], palette="viridis")
    plt.title('F1-Score by Class')
    plt.xticks(rotation=90)

    plt.tight_layout()
    st.pyplot()

# Function to display sample image and designation
def display_sample_image_and_designation(train_data):
    idx = np.random.randint(0, len(train_data))
    sample_image_path = f'./images/images/image_train/image_{train_data["imageid"].iloc[idx]}_product_{train_data["productid"].iloc[idx]}.jpg'
    sample_image = Image.open(sample_image_path)
    sample_designation = train_data['designation'].iloc[idx]

    st.write('Sample Image :\n')
    plt.imshow(sample_image)
    plt.axis('off')
    plt.title('Product Type Code: ' + str(train_data["prdtypecode"].iloc[idx]))
    st.pyplot()
    
    # st.image(sample_image, caption=f'Sample Image - Product Type Code: {train_data["prdtypecode"].iloc[idx]}')
    st.write(f'Sample Image Designation: {sample_designation}')

# Function to display model accuracy
def display_model_accuracy(model, features_image_testing, features_text_testing, y_test):
    loss, accuracy = model.evaluate({'image_input': features_image_testing, 'designation_input': features_text_testing}, y_test)
    st.write(f'Test Accuracy: {accuracy:.2%}')

# Function to display confusion matrix
def display_confusion_matrix(model, features_image_testing, features_text_testing, y_test, le):
    y_pred = model.predict({'image_input': features_image_testing, 'designation_input': features_text_testing})
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot()

# Function to display classification report
def display_classification_report(y_test, y_pred_classes, le):
    class_report = classification_report(y_test, y_pred_classes, target_names=le.classes_.astype(str))
    st.text_area('Classification Report', class_report, height=400)

# Function to display sample predictions
def display_sample_predictions(model, Remerged_XY_train, tfidf, le):
    testing_idx = int(len(Remerged_XY_train) * 0.2)
    testing_sample_df = Remerged_XY_train.tail(testing_idx)

    random_indices = np.random.randint(0, len(testing_sample_df), size=5)
    
    for idx in random_indices:
        sample_image_path = f'./images/images/image_train/image_{Remerged_XY_train["imageid"].iloc[idx]}_product_{Remerged_XY_train["productid"].iloc[idx]}.jpg'
        sample_image = preprocess_image(sample_image_path)
        sample_image = np.expand_dims(sample_image, axis=0)
        
        designation_test = Remerged_XY_train['designation'].iloc[idx]
        designation_test_tfidf = tfidf.transform([designation_test]).toarray()
        
        x_test_predict = model.predict([sample_image, designation_test_tfidf])
        best_prediction_x_test = np.argmax(x_test_predict, axis=1)
        
        predicted_label = le.inverse_transform(best_prediction_x_test)[0]
        actual_label = Remerged_XY_train['prdtypecode'].iloc[idx]
        
        st.image(sample_image.squeeze(), caption=f'Sample Image - Predicted: {predicted_label} | Actual: {actual_label}')

# Streamlit app
def main():
    nltk.download('stopwords')
    french_stop_words = stopwords.words('french')

    # Load and preprocess data
    X_train, X_test_processed, Y_train = load_data()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Rakuten Product Classifier')
    # st.set_page_config(page_title='Rakuten Product Classifier', layout='wide')

    train_data = pd.merge(X_train, Y_train, on='ID')

    train_data = train_data.head(12000)
    Y_train = Y_train.head(12000)

    le = LabelEncoder()
    train_data['prdtypecode'] = le.fit_transform(train_data['prdtypecode'])
    tfidf = TfidfVectorizer(max_features=1000, stop_words=french_stop_words)

    st.sidebar.title('Navigation')
    options = st.sidebar.radio('Select Page:', ['Home', 'Visualizations', 'Model Training', 'Predictions'])

    if options == 'Home':
        st.title('Welcome to the Product Classification App')
        st.markdown('### This application demonstrates the integration of text and image data for product classification.')
        st.image('https://source.unsplash.com/800x400/?product')
        st.markdown('Use the sidebar to navigate through different sections of the app.')

    elif options == 'Visualizations':
        st.title('Data Visualizations')
        st.markdown('### Explore the distribution of product type codes.')

        display_countplot(Y_train)
        display_sample_image_and_designation(train_data)

    elif options == 'Model Training':
        st.title('Model Training')
        st.markdown('### Train a model to classify products based on text and image data.')
        train_images = load_images('./images/images/image_train/', X_train['imageid'], X_train)

        features_image_training, features_image_testing, features_text_training, features_text_testing, y_train, y_test = train_test_split_for_model(train_images, X_train, tfidf)

        le = LabelEncoder()
        y_train = le.fit_transform(y_train['prdtypecode'])
        y_test = le.transform(y_test['prdtypecode'])

        st.write('Training the model...')
        model, history = compile_model(features_image_training, features_text_training, y_train)
        st.write('Model training completed!')

        st.subheader('Model Accuracy')
        display_model_accuracy(model, features_image_testing, features_text_testing, y_test)

    elif options == 'Predictions':
        st.title('Predictions and Evaluations')
        st.markdown('### Evaluate the trained model and make predictions.')
        test_images = load_images('./images/images/image_test/', X_test_processed['imageid'], X_test_processed)

        st.write('### Confusion Matrix')
        display_confusion_matrix(model, features_image_testing, features_text_testing, y_test)

        y_pred = model.predict({'image_input': features_image_testing, 'designation_input': features_text_testing}).argmax(axis=1)
        st.write('### Bar Graphs of Accuracy metrics')
        display_classification_report(y_test, y_pred_classes=y_pred, le=le)

        st.write('### Classification Report')
        classification_report_bar_plots(model, features_image_testing, features_text_testing, y_test, le)

    # st.title('Rakuten Product Classifier')

    # # Train and test Split for images and text data
    # X_train_images = load_images('./images/images/image_train', train_data['imageid'].values, train_data)

    # features_image_training, features_image_testing, features_text_training, features_text_testing, y_train, y_test = train_test_split_for_model(
    #     train_images=X_train_images, train_data=train_data, tfidf=tfidf
    # )

    # # st.button('Show Sample Data Points')
    # if st.button('Show Sample Data Points'):
    #     st.subheader('SAMPLE DATA:')
    #     st.write(train_data.head(5))

    # # Display countplot
    # # st.button('Show Product Type Code Distributions Graph')/
    # if st.button('Show Product Type Code Distributions Graph'):
    #     st.subheader('Product Type Code Distribution')
    #     display_countplot(train_data)

    # # Display sample image and designation
    # # st.button('Show Sample Image')
    # if st.button('Show Sample Image'):
    #     st.subheader('Sample Image and Designation')
    #     display_sample_image_and_designation(train_data)

    # # st.button('Start Training')
    # if st.button('Start Training'):

    #     # Train the model
    #     st.subheader('Training the Model')
    #     model, history = compile_model(
    #         features_image_training=features_image_training,
    #         features_text_training=features_text_training,
    #         y_train=y_train,
    #     )
    #     st.write(f'Model History: {history}')

    # # Display model accuracy
    # # st.button('Show Model Accuracy')
    # if st.button('Show Model Accuracy'):
    #     st.subheader('Model Accuracy')
    #     display_model_accuracy(model, features_image_testing, features_text_testing, y_test)

    # # Display confusion matrix
    # # st.button('Show Confusion Matrix')
    # if st.button('Show Confusion Matrix'):
    #     st.subheader('Confusion Matrix')
    #     display_confusion_matrix(model, features_image_testing, features_text_testing, y_test, le)

    # # Display classification report
    # # st.button('Show classification Report')
    # if st.button('Show classification Report'):
    #     st.subheader('Classification Report')
    #     display_classification_report(y_test, model.predict({'image_input': features_image_testing, 'designation_input': features_text_testing}).argmax(axis=1), le)

    # # Display sample predictions
    # # st.subheader('Sample Predictions')
    # # display_sample_predictions(model, Remerged_XY_train, tfidf, le)

    # if st.button('Show classification Report Bar Graphs'):
    #     st.subheader('Bar Graphs of Precision, Recall and F1-Score')
    #     classification_report_bar_plots(model, features_image_testing, features_text_testing, y_test, le)

if __name__ == '__main__':
    main()
