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
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RECOMMENDED JOBS PREDICTION", page_icon="📈", layout="wide")

def main():
    
    df = pd.read_csv('../datasets/business_dataset.csv')

    st.title("RECOMMENDED JOBS PREDICTION")
    st.markdown("### This page predicts the job Title of the alumni based on their current Education Subject, Degree, Industry and Location.")
    st.divider()

    # Assuming you have already filled NaN values in categorical columns
    categorical_columns = ['Education Degree', 'Education Subject', 'Location City', 'Location State', 'Location Country', 'Company Industry Name', 'Employment Title']
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # Select features and target variable
    features = ['Education Degree', 'Education Subject', 'Company Industry Name',   'Location Country', 'Location State', 'Location City',]
    target = 'Employment Title'

    # Concatenate text features into a single column for each row
    df['text_features'] = df[features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to convert the text features into numerical format
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text_features'])


    ################ Education Degree ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> Education Degree </h2>", unsafe_allow_html=True)
    education_degree_options = ["Bachelor\'s Degree", "Master\'s Degree", "PhD", "Graduate Degree", "Graduate Diploma", "Exchange Student"]
    education_degree = st.selectbox("Select an option:", education_degree_options)

    # Display the user input and selected option
    st.write("You selected:", education_degree)
    st.divider()

    ################ Education Subject ################
    st.markdown("<h2 style='text-align: center;'> Education Subject </h2>", unsafe_allow_html=True)
    st.write("### Education Subject")
    education_subject_options = ['Business', 'Business Psychology', 'Business Administration',
                               'Icon Business Bootcamp','International Business',
                               'Information Technology and Business', 'Media Arts/Business']

    
    education_subject = st.selectbox("Select an option:", education_subject_options)

    # Display the user input and selected option
    st.write("You selected:", education_subject)
    st.divider()

    ################ Job Industry ################
    # Dropdown menu for selecting options
    # st.markdown("<h1 style='text-align: center;'> Where do you want to Work? </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Job Industry </h2>", unsafe_allow_html=True)
    industry_options = ['Software Development', 'Primary/Secondary Education',
       'Higher Education', 'Law Practice', 'Hospital & Health Care',
       'Chemicals', 'Business Consulting and Services',
       'Defense and Space Manufacturing', 'IT Services and IT Consulting',
       'Human Resources', 'Utilities', 'Banking', 'Telecommunications',
       'Government Administration', 'Food and Beverage Services',
       'Management Consulting', 'Nonprofit Organization Management',
       'Research Services', 'Education Management', 'Construction',
       'Education', 'Financial Services', 'Environmental Services',
       'steel and building materials manufacturer',
       'Transportation/Trucking/Railroad', 'Consultation',
       'Mining & Metals', 'Newspaper Publishing', 'Mining',
       'Marketing and Advertising', 'Hospitality', 'Retail', 'Research',
       'Building Materials', 'Internet Publishing',
       'Non-profit Organizations', 'Accounting',
       'Museums and Institutions', 'Packaging and Containers',
       'Computer Software', 'Machinery', 'Machinery Manufacturing',
       'Information Technology and Services', 'Investment Management',
       'Real Estate', 'Health, Wellness and Fitness',
       'Pharmaceutical Manufacturing', 'Design Services', 'Sports',
       'Oil & Energy', 'Retail Apparel and Fashion', 'Insurance',
       'IT Service', 'Staffing and Recruiting', 'Plastics',
       'Industrial Machinery Manufacturing']
    indusrty = st.selectbox("Select an option:", industry_options)

    # Display the user input and selected option
    st.write("You selected:", indusrty)
    st.divider()


    ################ Location Country ################
    # Dropdown menu for selecting options
    st.markdown("<h1 style='text-align: center;'> Where do you want to Work? </h1>", unsafe_allow_html=True)
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



    # Example: Predict the Company Type for a single data point
    single_data_point = {
        'Education Degree': education_degree,
        'Education Subject': education_subject,
        'Company Industry Name': indusrty,
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

    # Calculate cosine similarity between user profile and job profiles
    cosine_sim = cosine_similarity(single_data_tfidf, tfidf_matrix)

    # Get recommended job indices based on similarity scores
    similar_jobs = list(enumerate(cosine_sim[0]))
    sorted_jobs = sorted(similar_jobs, key=lambda x: x[1], reverse=True)

    # Display the top recommended jobs (you can adjust the number of recommendations)
    st.markdown("<h1 style='text-align: center;'> Recommended Jobs </h1>", unsafe_allow_html=True)
    top_recommendations = sorted_jobs[:5]  # Here, we recommend the top 5 jobs
    for job_idx, similarity_score in top_recommendations:
        st.write(f'Job Title: {df["Employment Title"][job_idx]}')
        st.write(f'Company: {df["Employment Company Name"][job_idx]}')
        st.write(f'Similarity Score: {similarity_score:.2f}')
        st.write('-' * 40)



if __name__ == "__main__":
    main()



