import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="JOB INDUSTRY PREDICTION", page_icon="📈", layout="wide")

def main():
    
    url ="https://raw.githubusercontent.com/rafsunsheikh/alumni_employability/master/datasets/alumni_dataset_business.csv"
    df = pd.read_csv(url)

    # Filter rows where 'Education Subject' column is 'Business'
    df = df[df['Education Subject'] == 'Business']

    st.title("JOB INDUSTRY PREDICTION")
    st.markdown("### This page predicts the job industry of the alumni based on their current Education Major, Degree and Location.")
    st.divider()

    # # Data preprocessing
    # # Handle missing values by filling them with 'Unknown' for categorical columns
    # categorical_columns = ['Education Degree', 'Education Major', 'Location City', 'Location State', 'Location Country', 'Company Industry Name']
    # df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # # Select features and target variable
    # features = ['Education Degree', 'Education Major', 'Location City', 'Location State', 'Location Country']
    # target = 'Company Industry Name'

    # # Encode categorical variables
    # label_encoders = {}
    # for feature in features:
    #     label_encoders[feature] = LabelEncoder()
    #     df[feature] = label_encoders[feature].fit_transform(df[feature])

    # # Split data into training and testing sets
    # X = df[features]
    # y = df[target]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Assuming you have already filled NaN values in categorical columns
    categorical_columns = ['Education Degree', 'Education Major', 'Location City', 'Location State', 'Location Country', 'Company Industry Name']
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # Select features and target variable
    features = ['Education Degree', 'Education Major',   'Location Country', 'Location State', 'Location City']
    target = 'Company Industry Name'

    # Concatenate text features into a single column for each row
    df['text_features'] = df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to convert the text features into numerical format
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(df['text_features'])

    # Split data into training and testing sets
    y = df[target]
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)


    # Choose and train a classification model (Random Forest in this example)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_tfidf, y_train)

    # Make predictions on the testing data
    y_pred = clf.predict(X_test_tfidf)

    # Evaluate the model's performance
    # classification_rep = classification_report(y_test, y_pred)
    # print('Classification Report:\n', classification_rep)





    ################ Education Degree ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> Education Degree </h2>", unsafe_allow_html=True)
    education_degree_options = ["Bachelor\'s Degree", "Master\'s Degree", "PhD", "Graduate Degree", "Graduate Diploma", "Exchange Student"]
    education_degree = st.selectbox("Select an option:", education_degree_options)

    # Display the user input and selected option
    st.write("You selected:", education_degree)
    st.divider()

    ################ Education Major ################
    st.markdown("<h2 style='text-align: center;'> Education Major </h2>", unsafe_allow_html=True)
    st.write("### Education Major")
    education_major_options = ['BBA', 'Human Resources Management', 'MBA',
       'Business Administration and Management, General',
       'Business, Management, Marketing, and Related Support Services',
       'Business Psychology', 'Business Administration Marketing',
       'International Business, Sociology, Philosophy',
       'International Business', 'Business and Environmental Managment',
       'Environmental and Business Management',
       'Marketing and Management', 'Industrial Relations',
       'Marketing and International Business',
       'Business Economics and Marketing', 'Strategic Management',
       'Marketing, Public Relations',
       'Information Technology and International Business',
       'Business Economics with Marketing and Psychology',
       'Business Management', 'Human Resource Management',
       'Graduate Certificate',
       'Accounting, Finance, & International Business',
       'International Business and Management',
       'Business/Commerce, General', 'Marketing & Management',
       'Environment and Business',
       'Business Administration and Management',
       'Doctor of Business Administration', 'International Busienss',
       'Entrepreneurship/Entrepreneurial Studies']
    
    education_major = st.selectbox("Select an option:", education_major_options)

    # Display the user input and selected option
    st.write("You selected:", education_major)
    st.divider()


    ################ Location Country ################
    # Dropdown menu for selecting options
    st.markdown("<h1 style='text-align: center;'> What is your location? </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Country </h2>", unsafe_allow_html=True)
    location_country_options = ['USA', 'Australia', 'Norway', 'United Kingdom', 'Singapore',
       'India', 'United Arab Emirates', 'Finland', 'Canada', 'China',
       'Netherlands']
    location_country = st.selectbox("Select an option:", location_country_options)

    # Display the user input and selected option
    st.write("You selected:", location_country)
    st.divider()


    ################ Location State ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> State </h2>", unsafe_allow_html=True)
    location_state_options = ['South Carolina', 'Australian Capital Territory', 'Agder',
       'New South Wales', 'Victoria', 'Queensland', 'England', 'Wales',
       'The Globe', 'Iowa', 'Maharashtra', 'New York', 'Dubai',
       'Western Australia', 'District of Columbia', 'Uusimaa',
       'California', 'Minnesota', 'Washington', 'Texas', 'Tasmania',
       'Ontario', 'Hong Kong', 'North Carolina', 'South Australia',
       'Scotland']
    location_state = st.selectbox("Select an option:", location_state_options)

    # Display the user input and selected option
    st.write("You selected:", location_state)
    st.divider()


    ################ Location City ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> City </h2>", unsafe_allow_html=True)
    location_city_options = ['Summerville', 'Canberra', 'Kristiansand', 'Newcastle', 'Wantirna',
       'Sydney', 'Brisbane', 'Waratah', 'Bristol', 'Newland', 'Pontypool',
       'New York City Metropolitan Area', 'Cecil St', 'Des Moines',
       'Melbourne', 'Kirrawee', 'Braemar', 'Mumbai', 'North Mackay',
       'Brooklyn', 'Newcastle East', 'Dubai', 'Gracetown', 'Perth',
       'Forresters Beach', "Saint Paul's", 'Ulan', 'Washington',
       'Columbia', 'Gold Coast', 'Helsinki', 'Del Mar',
       'Minneapolis-St. Paul', 'Sunshine Coast', 'Olympia', 'Austin',
       'Nambucca Heads', 'Glasgow', 'London', 'Central Coast',
       'South Hobart', 'Halls Head', 'The Junction', 'Toronto', 'Lithgow',
       'Port Macquarie', 'Erskineville', 'New York', 'Charlotte',
       'Gosford', 'Manchester', 'Adelaide', 'The Randstad',
       'San Francisco', 'Edinburgh']
    location_city = st.selectbox("Select an option:", location_city_options)

    # Display the user input and selected option
    st.write("You selected:", location_city)
    st.divider()



    # # Example: Predict the Company Type for a single data point
    # single_data_point = {
    #     'Education Degree': education_degree,
    #     'Education Major': education_major,
    #     'Location City': location_city,
    #     'Location State': location_state,
    #     'Location Country': location_country,
    # }
    # single_data_df = pd.DataFrame([single_data_point])
    # for feature in features:
    #     single_data_df[feature] = label_encoders[feature].transform(single_data_df[feature])

    # predicted_industry = clf.predict(single_data_df[features])
    # st.markdown("<h2 style='text-align: center;'> Predicted Industry </h2>", unsafe_allow_html=True)
    # st.markdown(f"<h3 style='text-align: center;'> {predicted_industry[0]} </h3>", unsafe_allow_html=True)
    # # st.write(f"Predicted Industry: {predicted_industry[0]}")


    # Example: Predict the Company Type for a single data point
    single_data_point = {
        'Education Degree': education_degree,
        'Education Major': education_major,
        'Location Country': location_country,
        'Location State': location_state,
        'Location City': location_city,
    }

    # Create a DataFrame for the single data point
    single_data_df = pd.DataFrame([single_data_point])

    # Concatenate text features into a single column for the single data point
    single_data_df['text_features'] = single_data_df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to transform the text features of the single data point
    single_data_tfidf = vectorizer.transform(single_data_df['text_features'])

    # Predict the Company Type for the single data point
    predicted_industry = clf.predict(single_data_tfidf)

    # Display the prediction
    st.markdown("<h2 style='text-align: center;'> Predicted Industry </h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'> {predicted_industry[0]} </h3>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()



