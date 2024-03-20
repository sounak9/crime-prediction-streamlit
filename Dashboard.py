import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
import base64
import io
from io import BytesIO

# Load your data
uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read the uploaded file into a pandas DataFrame
    crimes_data = pd.read_csv(uploaded_file)
    
    crimes_data.columns = crimes_data.columns.str.strip()
    crimes_data.columns = crimes_data.columns.str.replace(',', '')
    crimes_data.columns = crimes_data.columns.str.replace(' ', '_')
    crimes_data.columns = crimes_data.columns.str.lower()

    #Checking the data contents
    st.write(crimes_data.head())

    #Checking the data for any null values and its datatypes
    st.write(crimes_data.info())

    #Check the data for any duplicates
    st.write(crimes_data[crimes_data.duplicated(keep=False)])

    # Removing Primary key type attributes as they are of no use for any type of analysis, Location columns is just a combination of Latitude and Longitude
    crimes_data.drop(['id','case_number','location'],axis=1,inplace=True)

    msno.heatmap(crimes_data,figsize=(15,5))

    msno.dendrogram(crimes_data,figsize=(20,5))

    crimes_data.isnull().sum()

    #Dropping observations where latitude is null/Nan
    crimes_data.dropna(subset=['latitude'],inplace=True)
    crimes_data.reset_index(drop=True,inplace=True)

    crimes_data.isnull().sum()

    crimes_data.dropna(inplace=True)
    crimes_data.reset_index(drop=True,inplace=True)

    crimes_data.info()

    """Removed the unnecessary entries."""

    #Converting the data column to datetime object so we can get better results of our analysis
    #Get the day of the week, month, and time of the crimes
    crimes_data.date = pd.to_datetime(crimes_data.date)
    crimes_data['day_of_week'] = crimes_data.date.dt.day_name()
    crimes_data['month'] = crimes_data.date.dt.month_name()
    crimes_data['time'] = crimes_data.date.dt.hour

    # Display the DataFrame with the added columns
    st.write(crimes_data.head())

    st.set_option('deprecation.showPyplotGlobalUse', False)# Plot 1: Countplot for crimes by month and year
    plt.figure(figsize=(20, 5))
    zone_plot = sns.countplot(data=crimes_data, x='month', hue='year', order=crimes_data['month'].value_counts().index, palette='Set2')
    
    st.pyplot()

    # Plot 2: Heatmap of missing values
    plt.figure(figsize=(15, 5))
    msno.heatmap(crimes_data, figsize=(15,5))
    st.pyplot()

    # Plot 3: Dendrogram of missing values
    plt.figure(figsize=(20, 5))
    msno.dendrogram(crimes_data, figsize=(20,5))
    st.pyplot()

    primary_type_map = {
    ('BURGLARY','MOTOR VEHICLE THEFT','THEFT','ROBBERY') : 'THEFT',
    ('BATTERY','ASSAULT','NON-CRIMINAL','NON-CRIMINAL (SUBJECT SPECIFIED)') : 'NON-CRIMINAL_ASSAULT',
    ('CRIM SEXUAL ASSAULT','SEX OFFENSE','STALKING','PROSTITUTION') : 'SEXUAL_OFFENSE',
    ('WEAPONS VIOLATION','CONCEALED CARRY LICENSE VIOLATION') :  'WEAPONS_OFFENSE',
    ('HOMICIDE','CRIMINAL DAMAGE','DECEPTIVE PRACTICE','CRIMINAL TRESPASS') : 'CRIMINAL_OFFENSE',
    ('KIDNAPPING','HUMAN TRAFFICKING','OFFENSE INVOLVING CHILDREN') : 'HUMAN_TRAFFICKING_OFFENSE',
    ('NARCOTICS','OTHER NARCOTIC VIOLATION') : 'NARCOTIC_OFFENSE',
    ('OTHER OFFENSE','ARSON','GAMBLING','PUBLIC PEACE VIOLATION','INTIMIDATION','INTERFERENCE WITH PUBLIC OFFICER','LIQUOR LAW VIOLATION','OBSCENITY','PUBLIC INDECENCY') : 'OTHER_OFFENSE'
}
    primary_type_mapping = {}
    for keys, values in primary_type_map.items():
        for key in keys:
            primary_type_mapping[key] = values
    crimes_data['primary_type_grouped'] = crimes_data.primary_type.map(primary_type_mapping)

    zone_mapping = {
    'N' : 'North',
    'S' : 'South',
    'E' : 'East',
    'W' : 'West'
}
    crimes_data['zone'] = crimes_data.block.str.split(" ", n = 2, expand = True)[1].map(zone_mapping)

    #Mapping seasons from month of crime
    season_map = {
        ('March','April','May') : 'Spring',
        ('June','July','August') : 'Summer',
        ('September','October','November') : 'Fall',
        ('December','January','February') : 'Winter'
    }
    season_mapping = {}
    for keys, values in season_map.items():
        for key in keys:
            season_mapping[key] = values
    crimes_data['season'] = crimes_data.month.map(season_mapping)

    #Mapping similar locations of crime under one group.
    loc_map = {
        ('RESIDENCE', 'APARTMENT', 'CHA APARTMENT', 'RESIDENCE PORCH/HALLWAY', 'RESIDENCE-GARAGE',
        'RESIDENTIAL YARD (FRONT/BACK)', 'DRIVEWAY - RESIDENTIAL', 'HOUSE') : 'RESIDENCE',

        ('BARBERSHOP', 'COMMERCIAL / BUSINESS OFFICE', 'CURRENCY EXCHANGE', 'DEPARTMENT STORE', 'RESTAURANT',
        'ATHLETIC CLUB', 'TAVERN/LIQUOR STORE', 'SMALL RETAIL STORE', 'HOTEL/MOTEL', 'GAS STATION',
        'AUTO / BOAT / RV DEALERSHIP', 'CONVENIENCE STORE', 'BANK', 'BAR OR TAVERN', 'DRUG STORE',
        'GROCERY FOOD STORE', 'CAR WASH', 'SPORTS ARENA/STADIUM', 'DAY CARE CENTER', 'MOVIE HOUSE/THEATER',
        'APPLIANCE STORE', 'CLEANING STORE', 'PAWN SHOP', 'FACTORY/MANUFACTURING BUILDING', 'ANIMAL HOSPITAL',
        'BOWLING ALLEY', 'SAVINGS AND LOAN', 'CREDIT UNION', 'KENNEL', 'GARAGE/AUTO REPAIR', 'LIQUOR STORE',
        'GAS STATION DRIVE/PROP.', 'OFFICE', 'BARBER SHOP/BEAUTY SALON') : 'BUSINESS',

        ('VEHICLE NON-COMMERCIAL', 'AUTO', 'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)', 'TAXICAB',
        'VEHICLE-COMMERCIAL', 'VEHICLE - DELIVERY TRUCK', 'VEHICLE-COMMERCIAL - TROLLEY BUS',
        'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS') : 'VEHICLE',

        ('AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA', 'CTA PLATFORM', 'CTA STATION', 'CTA BUS STOP',
        'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA', 'CTA TRAIN', 'CTA BUS', 'CTA GARAGE / OTHER PROPERTY',
        'OTHER RAILROAD PROP / TRAIN DEPOT', 'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA',
        'AIRPORT BUILDING NON-TERMINAL - SECURE AREA', 'AIRPORT EXTERIOR - NON-SECURE AREA', 'AIRCRAFT',
        'AIRPORT PARKING LOT', 'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA', 'OTHER COMMERCIAL TRANSPORTATION',
        'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA', 'AIRPORT VENDING ESTABLISHMENT',
        'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA', 'AIRPORT EXTERIOR - SECURE AREA', 'AIRPORT TRANSPORTATION SYSTEM (ATS)',
        'CTA TRACKS - RIGHT OF WAY', 'AIRPORT/AIRCRAFT', 'BOAT/WATERCRAFT', 'CTA PROPERTY', 'CTA "L" PLATFORM',
        'RAILROAD PROPERTY') : 'PUBLIC_TRANSPORTATION',

        ('HOSPITAL BUILDING/GROUNDS', 'NURSING HOME/RETIREMENT HOME', 'SCHOOL, PUBLIC, BUILDING',
        'CHURCH/SYNAGOGUE/PLACE OF WORSHIP', 'SCHOOL, PUBLIC, GROUNDS', 'SCHOOL, PRIVATE, BUILDING',
        'MEDICAL/DENTAL OFFICE', 'LIBRARY', 'COLLEGE/UNIVERSITY RESIDENCE HALL', 'YMCA', 'HOSPITAL') : 'PUBLIC_BUILDING',

        ('STREET', 'PARKING LOT/GARAGE(NON.RESID.)', 'SIDEWALK', 'PARK PROPERTY', 'ALLEY', 'CEMETARY',
        'CHA HALLWAY/STAIRWELL/ELEVATOR', 'CHA PARKING LOT/GROUNDS', 'COLLEGE/UNIVERSITY GROUNDS', 'BRIDGE',
        'SCHOOL, PRIVATE, GROUNDS', 'FOREST PRESERVE', 'LAKEFRONT/WATERFRONT/RIVERBANK', 'PARKING LOT', 'DRIVEWAY',
        'HALLWAY', 'YARD', 'CHA GROUNDS', 'RIVER BANK', 'STAIRWELL', 'CHA PARKING LOT') : 'PUBLIC_AREA',

        ('POLICE FACILITY/VEH PARKING LOT', 'GOVERNMENT BUILDING/PROPERTY', 'FEDERAL BUILDING', 'JAIL / LOCK-UP FACILITY',
        'FIRE STATION', 'GOVERNMENT BUILDING') : 'GOVERNMENT',

        ('OTHER', 'ABANDONED BUILDING', 'WAREHOUSE', 'ATM (AUTOMATIC TELLER MACHINE)', 'VACANT LOT/LAND',
        'CONSTRUCTION SITE', 'POOL ROOM', 'NEWSSTAND', 'HIGHWAY/EXPRESSWAY', 'COIN OPERATED MACHINE', 'HORSE STABLE',
        'FARM', 'GARAGE', 'WOODED AREA', 'GANGWAY', 'TRAILER', 'BASEMENT', 'CHA PLAY LOT') : 'OTHER'
    }

    loc_mapping = {}
    for keys, values in loc_map.items():
        for key in keys:
            loc_mapping[key] = values
    crimes_data['loc_grouped'] = crimes_data.location_description.map(loc_mapping)

    #Mapping crimes to ints to get better information from plots
    crimes_data.arrest = crimes_data.arrest.astype(int)
    crimes_data.domestic = crimes_data.domestic.astype(int)
#pie plot1
    plt.figure(figsize=(10, 10))
    crimes_data_primary_type_pie = plt.pie(crimes_data['primary_type_grouped'].value_counts(), labels=crimes_data['primary_type_grouped'].value_counts().index, autopct='%1.1f%%', radius=2.5)
    plt.legend(loc='best')
    st.pyplot()
#pie plot2
    plt.figure(figsize=(10, 10))
    crimes_data_loc_pie = plt.pie(crimes_data['loc_grouped'].value_counts(), labels=crimes_data['loc_grouped'].value_counts().index, autopct='%1.1f%%', shadow=True, radius=2.5)
    plt.legend(loc='best')
    st.pyplot()
#Bar chart
    plt.figure(figsize=(20, 6))
    top_20_primary_types = crimes_data.primary_type.value_counts().index[:20]
    top_20_counts = crimes_data.primary_type.value_counts().values[:20]
    primary_type_plot = sns.barplot(x=top_20_primary_types, y=top_20_counts, palette='Set2')
    plt.xticks(rotation=45) 
    st.pyplot()

    zone_plot = sns.countplot(data=crimes_data,x='zone',hue='year',order=crimes_data.zone.value_counts().index,palette='Set2')
    st.pyplot()

    zone_plot = sns.countplot(data=crimes_data,x='season',hue='year',palette='Set2')
    st.pyplot()

    arrest_plot = sns.countplot(data=crimes_data,x='year',hue='arrest',palette='Set2')
    st.pyplot()

    plt.figure(figsize=(20, 6))  # Adjust the figure size as needed
    # Extracting the top 20 location descriptions and their counts
    top_20_location_desc = crimes_data.location_description.value_counts().index[:20]
    top_20_counts = crimes_data.location_description.value_counts().values[:20]
    # Creating the bar plot
    location_description_plot_2018 = sns.barplot(x=top_20_location_desc, y=top_20_counts, palette='Set2')
    plt.xticks(rotation=45) 
    st.pyplot()

    crimes_data_primary_type_pie = plt.pie(crimes_data.primary_type_grouped.value_counts(),labels=crimes_data.primary_type_grouped.value_counts().index,autopct='%1.1f%%',shadow=True,radius=2.5)
    plt.legend(loc = 'best')
    st.pyplot()

#Creating a scatter plot for
    district_crime_rates = pd.DataFrame(columns=['theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count','human_trafficking_count', 'narcotic_offense_count',
    'other_offense_count'])
    district_crime_rates = district_crime_rates.astype(int)

    for i in range(1, 32):
        temp_district_df = crimes_data[crimes_data['district'] == i]

        temp_district_theft = temp_district_df[temp_district_df['primary_type_grouped'] == 'THEFT']
        num_theft = temp_district_theft.primary_type_grouped.count()

        temp_district_assault = temp_district_df[temp_district_df['primary_type_grouped'] == 'NON-CRIMINAL_ASSAULT']
        num_assault = temp_district_assault.primary_type_grouped.count()

        temp_district_sexual_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'SEXUAL_OFFENSE']
        num_sexual_offense = temp_district_sexual_offense.primary_type_grouped.count()

        temp_district_weapons_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'WEAPONS_OFFENSE']
        num_weapons_offense = temp_district_weapons_offense.primary_type_grouped.count()

        temp_district_criminal_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'CRIMINAL_OFFENSE']
        num_criminal_offense = temp_district_criminal_offense.primary_type_grouped.count()

        temp_district_human_trafficking = temp_district_df[temp_district_df['primary_type_grouped'] == 'HUMAN_TRAFFICKING_OFFENSE']
        num_human_trafficking = temp_district_human_trafficking.primary_type_grouped.count()

        temp_district_narcotic_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'NARCOTIC_OFFENSE']
        num_narcotic_offense = temp_district_narcotic_offense.primary_type_grouped.count()

        temp_district_other_offense = temp_district_df[temp_district_df['primary_type_grouped'] == 'OTHER_OFFENSE']
        num_other_offense = temp_district_other_offense.primary_type_grouped.count()

        district_crime_rates.loc[i] = [num_theft, num_assault, num_sexual_offense, num_weapons_offense, num_criminal_offense, num_human_trafficking, num_narcotic_offense, num_other_offense]

    district_crime_rates.head()

    # Standardize the data
    district_crime_rates_standardized = preprocessing.scale(district_crime_rates)
    district_crime_rates_standardized = pd.DataFrame(district_crime_rates_standardized)
    district_crime_rates_standardized.head()

    # Clustering with K-Means
    kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(district_crime_rates_standardized)
    #y_kmeans

    #beginning of  the cluster numbering with 1 instead of 0
    y_kmeans1=y_kmeans+1

    # New list called cluster
    kmeans_clusters = list(y_kmeans1)
    # Adding cluster to our data set
    district_crime_rates['kmeans_cluster'] = kmeans_clusters

    #Mean of clusters 1 to 4
    kmeans_mean_cluster = pd.DataFrame(round(district_crime_rates.groupby('kmeans_cluster').mean(),1))
    #kmeans_mean_cluster

    district_crime_rates.head()

    # Clustering with DBSCAN
    clustering = DBSCAN(eps = 1, min_samples = 3, metric = "euclidean").fit(district_crime_rates_standardized)

    # Show clusters
    dbscan_clusters = clustering.labels_
    # print(clusters)

    district_crime_rates['dbscan_clusters'] = dbscan_clusters + 2
    district_crime_rates.head()

    # Clustering with Hierarchical Clustering with average linkage
    clustering = linkage(district_crime_rates_standardized, method = "average", metric = "euclidean")

    # Plot dendrogram




    # Form clusters
    hierarchical_clusters = fcluster(clustering, 4, criterion = 'maxclust')
    #print(clusters)

    district_crime_rates['hierarchical_clusters'] = hierarchical_clusters
    district_crime_rates.head()

    # Add 'district' column
    district_crime_rates['district'] = district_crime_rates.index
    district_crime_rates = district_crime_rates[['district', 'kmeans_cluster', 'dbscan_clusters', 'hierarchical_clusters', 'theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count', 'human_trafficking_count', 'narcotic_offense_count', 'other_offense_count']]

    # Remove all columns but 'district' & each method's cluster
    district_crime_rates = district_crime_rates.drop(['theft_count', 'assault_count', 'sexual_offense_count', 'weapons_offense_count', 'criminal_offense_count', 'human_trafficking_count', 'narcotic_offense_count', 'other_offense_count'], axis=1)
    district_crime_rates.head(31)

    # Merge each district's clusters for each method into a single dataframe
    crimes_data_clustered = pd.merge(crimes_data, district_crime_rates, on='district', how='inner')
    st.write(crimes_data.head())
   

    # Read the uploaded file into a pandas DataFrame
crimes_data = pd.read_csv(uploaded_file)
st.write("Column Names:", crimes_data.columns.tolist())
st.write("First few rows of the DataFrame:")
st.write(crimes_data.head())

# Filter out rows where X_COORDINATE is not 0
new_crimes_data = crimes_data.loc[crimes_data['X_COORDINATE'] != 0]

# Perform KMeans clustering
kmeans = KMeans(n_clusters=5)  # You can adjust the number of clusters as needed
new_crimes_data['kmeans_cluster'] = kmeans.fit_predict(new_crimes_data[['X_COORDINATE', 'Y_COORDINATE']])

# Plot the clustered data
fig, ax = plt.subplots(figsize=(10, 8))
sns.scatterplot(x='X_COORDINATE', y='Y_COORDINATE', data=new_crimes_data, hue='kmeans_cluster', palette='Dark2', legend='full', alpha=0.7, ax=ax)

ax.set_title("KMeans Clustering of Crimes by District")
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")

# Display the plot in Streamlit
st.pyplot(fig)

# Read the crime data
