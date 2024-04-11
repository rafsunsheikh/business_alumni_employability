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

st.set_page_config(page_title="EDUCATION DEGREE AND MAJOR PREDICTION", page_icon="📈", layout="wide")

def main():
    
    url = "https://raw.githubusercontent.com/rafsunsheikh/alumni_employability/master/datasets/alumni_dataset_business.csv"
    df = pd.read_csv(url)

    # Filter rows where 'Education Subject' column is 'Business'
    df = df[df['Education Subject'] == 'Business']

    st.title("EDUCATION DEGREE AND MAJOR PREDICTION")
    st.markdown("### This page predicts the Education Degree and Major of the alumni based on their Desired Job TItle, Company, Industry and Location.")
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
    categorical_columns = ['Employment Title', 'Employment Company Name', 'Company Industry Name','Company Details Size', 'Company Type', 'Location City', 'Location State', 'Location Country', 'Education Major', 'Education Degree']
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

   ############################# Classifier for Education Degree ##################################

    # Select features and target variable
    features_for_degree = ['Employment Title', 'Employment Company Name', 'Company Industry Name','Company Details Size', 'Company Type', 'Location Country', 'Location State', 'Location City',]
    target_for_degree = 'Education Degree'

    # Concatenate text features into a single column for each row
    df['text_features_degree'] = df[features_for_degree].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to convert the text features into numerical format
    vectorizer = TfidfVectorizer()
    X_tfidf_degree = vectorizer.fit_transform(df['text_features_degree'])

    # Split data into training and testing sets
    y_degree = df[target_for_degree]
    X_train_degree_tfidf, X_test_degree_tfidf, y_train_degree, y_test_degree = train_test_split(X_tfidf_degree, y_degree, test_size=0.2, random_state=42)


    # Choose and train a classification model (Random Forest in this example)
    clf_degree = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_degree.fit(X_train_degree_tfidf, y_train_degree)

    # Make predictions on the testing data
    y_pred_degree = clf_degree.predict(X_test_degree_tfidf)

    # Evaluate the model's performance
    # classification_rep = classification_report(y_test, y_pred)
    # print('Classification Report:\n', classification_rep)


       ############################# Classifier for Education Major ##################################

    # Select features and target variable
    features_for_major = ['Employment Title', 'Employment Company Name', 'Company Industry Name','Company Details Size', 'Company Type', 'Location Country', 'Location State', 'Location City',]
    target_for_major = 'Education Major'

    # Concatenate text features into a single column for each row
    df['text_features_major'] = df[features_for_major].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to convert the text features into numerical format
    vectorizer = TfidfVectorizer()
    X_tfidf_major = vectorizer.fit_transform(df['text_features_major'])

    # Split data into training and testing sets
    y_major = df[target_for_major]
    X_train_major_tfidf, X_test_major_tfidf, y_train_major, y_test_major = train_test_split(X_tfidf_major, y_major, test_size=0.2, random_state=42)


    # Choose and train a classification model (Random Forest in this example)
    clf_major = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_major.fit(X_train_major_tfidf, y_train_major)

    # Make predictions on the testing data
    y_pred_major = clf_major.predict(X_test_major_tfidf)





    ################ Employment Title ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> Employment Title </h2>", unsafe_allow_html=True)
    employment_title_options = ['A/g Assistant Director Workplace Relations', 'Acting Service Manager',
         'Architectural Design Manager',
         'Asia Treasurer & Singapore Country Manager',
         'Assistant Director, Entry Level Program Management',
         'Assistant Exhibitions Producer', 'Board Treasurer',
         'Board member and Chair of the Audit and Risk Committee',
         'Brand & Marketing Manager - Cardiology, Sleep & Respiratory Medicine',
         'Business Analyst', 'Business Excellence and Improvement Lead',
         'Business Specialist', 'CS Training Manager',
         'Centre for Leadership Excellence Principal Consultant, Culture and Capability',
         'Chief Customer Officer',
         'Chief Responsible Investment and Signatory Relations Officer',
         'Client Data Governance Lead', 'Client Sales Manager', 'Co-Founder',
         'Communications Advisor (Marketing)', 'Consultant',
         'Deputy Vice-Chancellor Future Growth', 'Design Manager - City Projects',
         'Digital Marketing Coordinator', 'Digital Transformation Specialist',
         'Director Corporate Communications', 'Director, Executive Education',
         'Diversity and Inclusion Manager', 'Dual Brand Bank Manager',
         'Engineering Project Manager', 'Environmental Support Technician',
         'Executive Administration', 'Founder & Principal Lawyer',
         'GM Sales & Marketing', 'GRC Consultant', 'General Manager',
         'General Manager Asset Maintenance', 'Global Head of People & Culture',
         'HR Solutions Specialist','HR Support Consultant',
         'Head of Composites Program', 'Head of People and Governance',
         'Head of Product, Partnerships & Customer Excellence - Australia & New Zealand',
         'Human Resources Business Partner, Global Functions',
         'Human Resources Director', 'Human Resources Manager',
         'Independent Director', 'Interim Chief Executive Officer',
         'Interim Corporate Services Director', 'Intermediate Consultant',
         'Key Account Management', 'L&OD Lead',
         'Learning and Transformation Partner', 'Lecturer/Course Coordinator',
         'MBA Coordinator', 'Manager - IP and Technology & Innovation Governance',
         'Manager - People Consulting, HR Transformation',
         'Manager Contestable Network Solutions',
         'Manager Learning and Development', 'Manager Sustainability',
         'Manager, Line 1 Risk, IfNSW & HBCF', 'Managing Partner',
         'Mandarin Language Teacher', 'Marketing Assistant',
         'Marketing and Communications Manager', 'National MDU Operations Manager',
         'National Risk Systems Lead', 'National Sales Manager (Interim)',
         'Operations Analyst II', 'Operations Engineer (Marine Manager)',
         'Outside Sales Representative', 'Packaging Consultant',
         'Paraplanner and Research Officer', 'Partner' 'Planner',
         'Planning and Delivery Specialist', 'Planning and Performance Manager',
         'Practice Engagement & Growth Manager', 'Principal', 'Principal Analyst',
         'Principal Psychologist', 'Product Designer', 'Product Manager',
         'Product Owner [Via Capgemini]', 'Project Director', 'Project Manager',
         'Project Manager - Sydney', 'Project Manager, Interoperability',
         'Property Advisor', 'React Practice Lead', 'Recover@Work Partner',
         'Relationship Manager - Commercial (Financial Markets)',
         'Research Officer - ARC DECRA Project', 'SAED Project Manager',
         'SAIT Business School Instructor', 'Safety Quality Environment Manager',
         'Senior Associate', 'Senior Associate Consultant (Healthcare)',
         'Senior Business Analyst, Enterprise & Strategic Data Technology',
         'Senior Coast and Estuaries Officer',
         'Senior Manager, Global Enterprise Marketing',
         'Senior Marketing Executive, Audience Data', 'Senior Research Audiologist',
         'Senior Software Engineer', 'Seniorrdgiver HR', 'Solicitor',
         'Special Education Teacher (Mild/Moderate)',
         'Staff Officer - US Requirements, Force Posture Initiatives',
         'Strategic Coordinator', 'Supply Chain Finance Sr Manager - M&W',
         'Systems Manager', 'technology program',]

    employment_title = st.selectbox("Select an option:", employment_title_options)

    # Display the user input and selected option
    st.write("You selected:", employment_title)
    st.divider()

    ################ Employment Company Name ################
    st.markdown("<h2 style='text-align: center;'> Employment Company Name </h2>", unsafe_allow_html=True)
    st.write("### Employment Company Name")
    employment_company_name_options = ['ACN Pacific', 'ACT Education Directorate', 'ARTC' 'AVet Health Ltd',
         'Advitech', 'Alcoa of Australia Limited', 'Alexandria Moulding',
         'Andrew Sparks', 'AngloGold Ashanti',
         'Australian College of Perioperative Nurses (ACORN)',
         'Australian Digital Health Agency',
         'Australian Doctors International (ADI)',
         'Australian Government Department of Finance', 'Australian Museum',
         'Australian Psychological Services', 'Australian Red Cross',
         'Barnardos Australia', 'Bradken', 'CPS Technology & Infrastructure Pty Ltd',
         'Campbelltown City Council', 'Catherine Henry Lawyers',
         'Centre for Children and Young People',
         'Chamberlain Brown Property Acquisitions', 'Citadel', 'City of Melbourne',
         'CityFibre', 'CoAct', 'Commonwealth Bank', 'Defence Australia',
         'Dolphins NRL', 'EXYTE U.S., INC.', 'Element Materials Technology',
         'Emirates Steel Arkan', 'Employment Hero', 'Energy One Limited',
         'Essential Energy', 'FT Consulting', 'Financial Times',
         'Foresight Consulting', 'Forest Ave Outreach', 'Fortnum', 'GE HealthCare',
         'GenesisCare', 'Glencore Australia', 'Goodstart Early Learning', 'Google',
         'Hilton', 'Hollier Law', 'Inaburra School', 'InfoSecAssure',
         'Innovations Academy', 'Insight IT & Engineering Recruitment',
         'Iplex Pipelines', 'Iress', 'KPMG UK', 'La Trobe University',
         'Larsen & Toubro', 'Loom Decor', 'Lumea', 'Macquarie University',
         'Metso Outotec', 'Microsoft', 'Monash Health', 'NAB', 'NGH',
         'NSW Department of Planning and Environment',
         'National Acoustic Laboratories', 'National Disability Insurance Agency',
         'Neuberger Berman', 'Nigel J Barling Pty Ltd',
         'Novartis, Australia & New Zealand', 'Oak Hill Asset Management Inc',
         'Onsite Rental Group', 'Orica', 'OsloMet storbyuniversitetet', 'PepsiCo',
         'Principles for Responsible Investment', 'ProsperOps', 'PwC Australia',
         'Queensland Health', 'Relocity, Inc.', 'Roche Group', 'Rugby India',
         'SARA LEE HOLDINGS PTY. LTD', 'SPARC Group LLC',
         'SW Accountants & Advisors', 'Samsara', 'Securus Consulting Group',
         'Southern Alberta Institute of Technology (SAIT)',
         'Southern Cross University', 'Tactix-Sener Group',
         'The University of Sydney Business School', 'Toast',
         'Torfaen County Borough Council', 'Town of Port Hedland', 'Travelex',
         'TricorBraun', 'Tweddle Child and Family Health Service', 'UWV',
         'Uniting Communities', 'University of Newcastle', 'Visy',
         'Water Treatment Services', 'Wessex Water', 'Westpac', 'Westpac Group',
         'Zenviron', 'iFin Advisory', 'icare NSW', 'nbn Australia']
    
    employment_company_name = st.selectbox("Select an option:", employment_company_name_options)

    # Display the user input and selected option
    st.write("You selected:", employment_company_name)
    st.divider()

    ################ Job Industry ################
    # Dropdown menu for selecting options
    # st.markdown("<h1 style='text-align: center;'> Where do you want to Work? </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Job Industry </h2>", unsafe_allow_html=True)
    industry_options = ['Accounting', 'Banking', 'Building Materials',
      'Business Consulting and Services', 'Chemicals', 'Computer Software',
      'Construction', 'Consultation', 'Defense and Space Manufacturing',
      'Design Services', 'Education', 'Education Management',
      'Environmental Services', 'Financial Services',
      'Food and Beverage Services', 'Government Administration',
      'Health, Wellness and Fitness', 'Higher Education',
      'Hospital & Health Care', 'Hospitality', 'Human Resources', 'IT Service',
      'IT Services and IT Consulting', 'Industrial Machinery Manufacturing',
      'Information Technology and Services', 'Insurance', 'Internet Publishing',
      'Investment Management', 'Law Practice', 'Machinery',
      'Machinery Manufacturing', 'Management Consulting',
      'Marketing and Advertising', 'Mining', 'Mining & Metals',
      'Museums and Institutions', 'Newspaper Publishing',
      'Non-profit Organizations', 'Nonprofit Organization Management',
      'Oil & Energy', 'Packaging and Containers', 'Pharmaceutical Manufacturing',
      'Plastics', 'Primary/Secondary Education', 'Real Estate', 'Research',
      'Research Services', 'Retail', 'Retail Apparel and Fashion',
      'Software Development', 'Sports', 'Staffing and Recruiting',
      'Telecommunications', 'Transportation/Trucking/Railroad', 'Utilities',
      'steel and building materials manufacturer']
         
    indusrty = st.selectbox("Select an option:", industry_options)

    # Display the user input and selected option
    st.write("You selected:", indusrty)
    st.divider()


     ################ Company Details Size ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> Company Details Size </h2>", unsafe_allow_html=True)
    company_details_size_options = ['1-10 employees', '10001+ employees', '1001-5000 employees',
      '11-50 employees', '2-10 employees', '201-500 employees',
      '5001-10000 employees', '501-1000 employees', '51-200 employees']
    
    company_details_size = st.selectbox("Select an option:", company_details_size_options)

    # Display the user input and selected option
    st.write("You selected:", company_details_size)
    st.divider()



     ################ Company Type ################
    # Dropdown menu for selecting options
    st.markdown("<h2 style='text-align: center;'> Company Type </h2>", unsafe_allow_html=True)
    company_type_options = ['Educational', 'Educational Institution', 'Government Agency', 'Non Profit',
      'Nonprofit', 'Partnership', 'Privately Held', 'Public Company',
      'Sole Proprietorship']
    company_type = st.selectbox("Select an option:", company_type_options)

    # Display the user input and selected option
    st.write("You selected:", company_type)
    st.divider()


    ################ Location Country ################
    # Dropdown menu for selecting options
    st.markdown("<h1 style='text-align: center;'> What is your location? </h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'> Country </h2>", unsafe_allow_html=True)
    location_country_options = ['Australia', 'Canada', 'China', 'Finland', 'India', 'Netherlands', 'Norway',
      'Singapore', 'USA', 'United Arab Emirates', 'United Kingdom',]
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
        'Employment Title': employment_title,
        'Employment Company Name': employment_company_name,
        'Company Industry Name': indusrty,
        'Company Details Size': company_details_size,
        'Company Type': company_type,
        'Location Country': location_country,
        'Location State': location_state,
        'Location City': location_city,
    }

   ################### Classification Education Degree ####################################
    # Create a DataFrame for the single data point
    single_data_df_degree = pd.DataFrame([single_data_point])

    # Concatenate text features into a single column for the single data point
    single_data_df_degree['text_features_degree'] = single_data_df_degree[features_for_degree].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to transform the text features of the single data point
    single_data_tfidf_degree = vectorizer.transform(single_data_df_degree['text_features_degree'])

    # Predict the Company Type for the single data point
    predicted_education_degree = clf_degree.predict(single_data_tfidf_degree)

    # Display the prediction
    st.markdown("<h2 style='text-align: center;'> Predicted Education Degree </h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'> {predicted_education_degree[0]} </h3>", unsafe_allow_html=True)


################### Classification Education Major ####################################
    # Create a DataFrame for the single data point
    single_data_df_major = pd.DataFrame([single_data_point])

    # Concatenate text features into a single column for the single data point
    single_data_df_major['text_features_major'] = single_data_df_major[features_for_major].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    # Use TfidfVectorizer to transform the text features of the single data point
    single_data_tfidf_major = vectorizer.transform(single_data_df_major['text_features_major'])

    # Predict the Company Type for the single data point
    predicted_education_major = clf_major.predict(single_data_tfidf_major)

    # Display the prediction
    st.markdown("<h2 style='text-align: center;'> Predicted Education Major </h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'> {predicted_education_major[0]} </h3>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()



