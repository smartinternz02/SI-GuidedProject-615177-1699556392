import re
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objs as go
import seaborn as sns
import warnings
# Filter warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

import flask
from flask import Flask, render_template, request
import pickle


app = Flask(__name__) #Initialize the flask App

zomato_df=pd.read_csv('zomato.csv')

# Dropping the columns "dish liked", "phone", "url"
zomato_df = zomato_df.drop(['phone', 'dish_liked', 'url'], axis=1)

# Remove NaN values from the dataset
zomato_df.dropna(how="any", inplace=True)

# Removing duplicates and displaying the sum of duplicated rows
zomato_df.duplicated().sum()
zomato_df.drop_duplicates(inplace=True)

# Changing the column names
zomato_df = zomato_df.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})
print("Column names after renaming:", zomato_df.columns)

# Removing '/5' from Rates
zomato_df = zomato_df.loc[zomato_df['rate'] != 'NEW']
zomato_df = zomato_df.loc[zomato_df['rate'] != '-'].reset_index(drop=True)

remove_slash = lambda x: x.replace('/5', '') if isinstance(x, str) else x
zomato_df['rate'] = zomato_df['rate'].apply(remove_slash).str.strip().astype('float')


# Changing the 'cost' to string, replacing commas with hyphens, and converting to float
zomato_df['cost'] = zomato_df['cost'].astype(str)
zomato_df['cost'] = zomato_df['cost'].apply(lambda x: x.replace(',', '-'))

# Splitting values based on hyphen and taking the average
zomato_df['cost'] = zomato_df['cost'].apply(lambda x: np.mean(list(map(float, x.split('-')))) if '-' in x else float(x))

# Now, converting the 'cost' column to float
zomato_df['cost'] = zomato_df['cost'].astype(float)
zomato_df['Mean Rating'] = zomato_df.groupby('name')['rate'].transform('mean')

# Scaling the mean rating values
scaler = MinMaxScaler(feature_range=(1, 5))
zomato_df['Mean Rating'] = scaler.fit_transform(zomato_df[['Mean Rating']]).round(2)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

        print("Accessed predict route")
        restaurant_name = request.form.get('restaurant_name')
        print("Restaurant Name:", restaurant_name)

         #type = request.form['type']
        # output = request.form['output']
        # if type=="text":
        #     output = re.sub('[^a-zA-Z.,]','',output)
        df_percent=zomato_df.head(5000) 
        print(df_percent.head() )
        # df_percent.head()             
        print(df_percent.shape) 
        print(df_percent.columns)
        df_percent.set_index('name',inplace=True)
        indices=pd.Series(df_percent.index)

        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.1, stop_words="english",max_features=1000)
        tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'].fillna(' '))
        cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
        # # # Save necessary objects using pickle
        # # with open('tfidf_model.pkl', 'wb') as file:
        # #     pickle.dump(tfidf, file)

        with open('cosine_similarities.pkl', 'wb') as file:
            pickle.dump(cosine_similarities, file)
        with open('cosine_similarities.pkl', 'rb') as file:
            cosine_similarities = pickle.load(file)
         
        def recommend(name, cosine_similarities, indices, subset_df):
            # Check if the name is in the DataFrame
            if name not in subset_df.index:
                print(f"Restaurant '{name}' not found in the dataset.")
                return None  # You can choose to return something else or handle it as needed
            
            # Create a list to put top restaurants
            recommend_restaurant = []
            
            # Find the index of the restaurant entered
            idx = subset_df.index.get_loc(name)
            
            # Find the restaurants with a similar cosine-sim value and order them from biggest to smallest
            score_series = cosine_similarities[idx]
            
            # Extract top 30 restaurant indexes with a similar cosine-sim value
            top10_indexes = np.argsort(score_series)[::-1][1:11]  # Starting from 1 to exclude the entered restaurant itself, and taking only the top 10
            
            # Names of the top 30 restaurants
            # Names of the top 10 restaurants
            for each in top10_indexes:
                recommend_restaurant.append(indices[each])

            
            # Creating the new data set to show similar restaurants
            df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
            
            # Create the top 30 similar restaurants with some of their columns
            for each in recommend_restaurant:
                df_new = pd.concat([df_new, subset_df[['cuisines', 'Mean Rating', 'cost']].loc[each]])
            
            # Drop the same-named restaurants and sort only the top 10 by the highest rating
            df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
            df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
            
            print(f'TOP {len(df_new)} RESTAURANTS LIKE {name} WITH SIMILAR REVIEWS:')
            
            return df_new


        # Example usage
        # recommend('Addhuri Udupi Bhojana', cosine_similarities, indices, df_percent)
            


        # Example usage
        # print("Before recommend function")
        # Check if restaurant_name is empty
        if not restaurant_name:
        # Set a default restaurant name or handle it as per your requirement
            restaurant_name = "Addhuri Udupi Bhojana"  # Example default name
        print("Restaurant Name:  1", restaurant_name)
        # result = recommend("Addhuri Udupi Bhojana", cosine_similarities, indices, df_percent)
        result = recommend(str(restaurant_name), cosine_similarities, indices, df_percent)
        # print("Restaurant Name:  2 Taaza Thindi" )
        # print(recommend('Taaza Thindi ', cosine_similarities, indices, df_percent))

        # print("After recommend function")
        # print(result)
        # print(type(result)) 
 
        # Pass the recommendations to the template
        return render_template('predict.html',recommendations=result) 
   

    



if __name__ == "__main__":
    app.run(debug=True, port=5000)  
