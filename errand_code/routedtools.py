# Primary functions for automated errand router
import psycopg2
import codecs
import pandas as pd
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import permutations, repeat
from config import *
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_stop_words(stop_file_path):
    '''Load up a set of stopwords to perform text preprocessing'''
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
    return frozenset(stop_set)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def build_word_vector_matrix(vector_file, n_words):
    '''Read a GloVe array from sys.argv[1] and return its vectors and labels as arrays'''
    np_arrays = []
    word_list = []
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for i, line in enumerate(f):
            sr = line.split()
            word_list.append(sr[0])
            np_arrays.append(np.array([float(j) for j in sr[1:]]))
            if i>n_words:
                break
    word_vectors = (np.array(np_arrays)).astype('float32')
    glove_dict = dict(zip(word_list,word_vectors))
    return glove_dict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_salient_terms(conn, table_name, stopwords, glove_dict):
    sql_query = f"""
    SELECT * FROM {table_name};
    """
    nearby_stores = pd.read_sql_query(sql_query,conn)
    print("Number of stores found ", nearby_stores.shape[0])
    
    # Pull reviews that match my selected business IDs
    all_ids = nearby_stores['business_id'].values.tolist()
    id_string= '\', \''.join(all_ids)
    id_string = '\''+ id_string+'\''
    query= f"""
    SELECT * FROM t_review_phx WHERE business_id IN ({id_string})
    """
    nearby_reviews = pd.read_sql_query(query,conn)

    # Sort both dfs by business_id.
    nearby_reviews = nearby_reviews.sort_values(by=['business_id'])
    nearby_stores = nearby_stores.sort_values(by=['business_id'])
    print(nearby_reviews.shape[0],'matched reviews found.')

    # Perform TF-IDF normalization to identify most salient/characteristic terms
    # Preprocessing:
    # - lemmatize
    # - ignore words that appear in 85% of documents, 
    # - eliminate stop words
    lemmatizer = WordNetLemmatizer()
    def preprocess1(token):
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        return token 

    # Get TF-IDF-reweighted vectors for each document + vocab
    # [Note: may want to specify 1 or 2 n-gram things w/ arg: ngram_range=(1,2)]
    vectorizer = TfidfVectorizer(max_df=0.85,stop_words=stopwords,preprocessor=preprocess1)
    X = vectorizer.fit_transform(nearby_reviews['text'].tolist())
    vocab = vectorizer.get_feature_names()
    tfidf_array = X.toarray()

    # Drop any vocab terms that are not in my dict
    keepers = [(glove_dict.get(x,None) is not None) for x in vocab]
    vocab = (np.array(vocab))[keepers]
    tfidf_array = tfidf_array[:, keepers]

    # Of remaining words, rerank to get most salient words -> match to vocab entries and save as dataframe we can match
    top_n = 50
    ranked1 = (-tfidf_array).argsort(axis=-1)[:,0:top_n]
    def vocab_lookup(row_in):
        return np.take(vocab,row_in)
    vocab_list = np.apply_along_axis(vocab_lookup, axis=1, arr=ranked1)
    # (NOTE: if we have time, may also be helpful to save scores.)

    # Now convert this to a dataframe with business_id
    topwords_df = pd.DataFrame(vocab_list).set_index(nearby_stores['business_id'])
    return topwords_df

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_nearby(start_lat, start_long, radius, names_table, reviews_table, conn, min_length):
    '''Get all businesses within specified radius of a starting location. Filter for those w/ reasonable review depth'''
    # Pull businesses
    sql_query = f"""
    SELECT * FROM {names_table} WHERE ( 
    ((latitude - {start_lat}) * (latitude - {start_lat})) + 
    ((longitude - {start_long}) * (longitude - {start_long}))) < {np.power(radius,2)}
    AND ( NOT (categories LIKE '%Restaurant%' OR categories LIKE '%Bar%' 
    OR categories LIKE '%Festivals%' OR categories LIKE '%Hair Salons%'
    OR categories LIKE '%Food Truck%' OR categories LIKE '%Hotel%'
    OR categories LIKE '%Contractor%' OR categories LIKE '%Apartment%'
    OR categories LIKE '%Cater%' OR categories LIKE '%Local Flavor%'
    OR categories LIKE '%Coffee%' OR categories ILIKE '%ice cream%')
    OR (categories LIKE '%Grocery%') );
    """
    nearby_stores = pd.read_sql_query(sql_query,conn)

    # Pull reviews that match my selected business IDs
    all_ids = nearby_stores['business_id'].values.tolist()
    id_string= '\', \''.join(all_ids)
    id_string = '\''+ id_string+'\''
    query= f"""
    SELECT * FROM {reviews_table} WHERE business_id IN ({id_string})
    """
    nearby_reviews = pd.read_sql_query(query,conn)
    # Filter out short reviews (and corresponding stores)
    nearby_reviews = nearby_reviews[nearby_reviews['text'].str.len()>min_length];
    nearby_stores = nearby_stores[np.in1d(nearby_stores['business_id'],
                                          nearby_reviews['business_id'])];
    # Sort dfs by business_id.
    nearby_stores = nearby_stores.sort_values(by=['business_id'])
    nearby_reviews = nearby_reviews.sort_values(by=['business_id'])
    
    return nearby_stores, nearby_reviews

# MAIN MATCHING FUNCTION - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_matches(shopping_list, glove_dict, nearby_stores, topwords_df):
    '''Matching function: identify stores liely to carry items in question.'''
    match_df = pd.DataFrame(np.zeros([len(shopping_list),5]),
                        columns=['item','names','business_ids','scores','item_copy'])
    match_df = match_df.astype('object')
    shopping_list_out = []
    drops = []
    for idx2, query in enumerate(shopping_list):
        print('processing', query)
        try:
            query_scores = np.zeros(topwords_df.shape[0])
            for entry in range(topwords_df.shape[0]):
                query_list = query.split();
                query_vect = np.asarray([glove_dict[x] for x in query_list])
                review_vects = np.asarray([glove_dict[x] for x in topwords_df.iloc[entry].values])
                all_cos = np.sort(cosine_similarity(query_vect,review_vects))
                query_scores[entry] = (np.max(all_cos) + np.mean(all_cos[-3:])) / 2
            top_scorers = np.argsort(-query_scores)
            top_scores = -np.sort(-query_scores)
            num_good = np.sum([top_scores>0.5]) # add a thresholding step to prevent terrible matches
            n = np.max([4, num_good])
            n = np.min([n,20])
            # Store results
            shopping_list_out.append(query)
            match_df.at[idx2, 'item'] = query
            match_df.at[idx2, 'names'] = (nearby_stores.iloc[top_scorers[0:(n-1)]])['name'].values
            match_df.at[idx2, 'business_ids'] = (nearby_stores.iloc[top_scorers[0:(n-1)]])['business_id'].values
            match_df.at[idx2, 'scores'] = top_scores[0:(n-1)]
            match_df.at[idx2, 'item_copy'] = list(repeat(query,(n-1)))
        except:
            print('(skipped this item.)')
            drops.append(idx2)
    match_df.drop(drops,inplace=True)
    return match_df, shopping_list_out

# QUADRATIC KNAPSACK OPTIMIZATION  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def knapsack_optimize(match_df, shopping_list, nearby_stores, start_lat,start_long, weights):  
    
    # A) Build candidate list - *every* store with a score (on some item) above a threshold
    full_ids = np.array([element for list_ in match_df['business_ids'].values for element in list_])
    full_items = np.array([element for list_ in match_df['item_copy'].values for element in list_])
    id_counts = (pd.Series(full_ids)).value_counts()
    
    # Get unique stores, and their matching information
    unique_ids = np.unique(full_ids)
    unique_info = (nearby_stores.set_index('business_id')).loc[unique_ids]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # B) Get weights -> first, disparate elements of store weights

    # i) Rating cost is just (5 - ['stars']) - generally going to be 0-3
    # ii) Dist cost is distance to store, in miles (will be 0-5)
    # iii) Match cost is avg of all scores for every time that store appeared in list.
    unique_info['dist_cost'] = np.sqrt((unique_info['latitude']-start_lat)**2 
                                       + (unique_info['longitude']-start_long)**2)*69
    full_scores = np.array([element for list_ in match_df['scores'].values for element in list_])
    unique_info['match_cost'] = 1 - np.array([np.mean(full_scores[full_ids==x]) for x in unique_ids])

    w = (weights[0]*(1+unique_info['dist_cost']) + 
         weights[1]*(50*unique_info['match_cost']) + 
         weights[2]*(5.25-unique_info['stars']))

    # C) Calculate "profit" matrix (with adjustments for crossed items)
    P =np.zeros([len(unique_ids), len(unique_ids)],dtype=float)
    for i in range(0,len(unique_ids)):
        for j in range(i,len(unique_ids)):
            if i==j:
                P[i,i] = id_counts.loc[unique_ids[i]]
            else:
                val1 = len(np.unique(full_items[np.any([full_ids==unique_ids[i],
                                                          full_ids==unique_ids[j]],axis=0)]))
                P[i,j] = val1
                P[j,i] = val1

    # D) Greedy search: keep grabbing best profit-to-weight-ratio (pair 1st, then singles) until we span shopping list
    score = 0.0
    top_j = 0
    top_i = 0
    for i in range(0,len(unique_ids)):
        for j in range(i,len(unique_ids)):
            if i!=j:
                if P[i,j]/(w[i]+w[j]) > score:
                    top_i = i
                    top_j = j
                    score = P[i,j]/(w[i]+w[j])
    # Double-check that we don't need only one of pair

    if P[top_i, top_i] == P[top_i, top_j]:
        knapsack_items = np.unique(full_items[full_ids==unique_ids[top_i]])
        knapsack_idx = [top_i]
    elif P[top_j, top_j] == P[top_i, top_j]:
        knapsack_items = np.unique(full_items[full_ids==unique_ids[top_j]])
        knapsack_idx = [top_j]
    else:
        knapsack_items = np.unique(full_items[np.any([full_ids==unique_ids[top_i],
                                                full_ids==unique_ids[top_j]],axis=0)])
        knapsack_idx = [top_i,top_j]

    mult = 0
    while (len(knapsack_items) < len(shopping_list)) and (mult<100):
        mult+=1
        best_ratio = 0
        for i in range(0,len(unique_ids)):
            if (i not in knapsack_idx):
                tmp_net = sum(P[i,x] for x in knapsack_idx) - sum(P[x,x] for x in knapsack_idx) - (mult*P[i,i])
                tmp_ratio = float(tmp_net)/float(w[i])
                if tmp_ratio > best_ratio:
                    best_ratio = tmp_ratio
                    top_i = i
        if best_ratio==0:
            break
        else:
            knapsack_items = np.unique(np.concatenate((knapsack_items, full_items[full_ids==unique_ids[top_i]])))
            knapsack_idx.append(top_i)          

    # Collect info for top choices
    top_info = unique_info.loc[np.take(unique_ids,knapsack_idx)]
    top_info.insert(0,'items',np.ndarray(top_info.shape[0],dtype='object'))
    b_id = top_info.index.values
    for idx,i in enumerate(knapsack_idx):
        top_info.at[b_id[idx],'items'] = (full_items[full_ids==unique_ids[i]])
    return top_info


# TRAVELING SALESMAN/MAPPING FCNS - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def make_nodes(lats, longs, start_lat, start_long):
    lats.insert(0,start_lat)
    longs.insert(0,start_long)
    labels = list(range(1,1+len(lats)))
    nodes = {}
    for label in labels:
        nodes[label] = [lats[label-1], longs[label-1]]
    return nodes

def distance1(p1, p2):
    """Calculates distance between two points, memoizes result"""
    d = (((p2[0] - p1[0])**2) + ((p2[1] - p1[1])**2)) **.5
    return d

def find_shortest_route(nodes, route_length):
    """Find shortest route of length route_length from nodes."""   
    minimum_distance = None
    for route in permutations(range(2, route_length+1)):
        route = route + (1,)
        current_distance = 0
        prev = nodes[1]
        for next in route:
            current_distance += distance1(prev, nodes[next])
            prev = nodes[next]
        route = (1,) + route 
        #print('route total: ', route, ', dist=',current_distance)
        if not minimum_distance or current_distance < minimum_distance:
            minimum_distance = current_distance
            minimum_route = route
    return minimum_distance, minimum_route


def build_urls(start_address, route,top_info, start_lat,start_long):
    # Build my URL (in order of the TSP routing)
    addr_string = start_address.replace(' ','+')+'/'
    for place in route[1:-1]:
        print(top_info['name'][place-2])
        addr_string = addr_string + top_info['address'].values[place-2].replace(' ','+')+'+Phoenix,+AZ/'
    addr_string = addr_string + start_address.replace(' ','+')+'/'
    maps_url = 'https://www.google.com/maps/dir/'+addr_string
    
    # Format a Google Maps API embedded image URL
    img_url = "https://maps.googleapis.com/maps/api/staticmap?center=" + start_address.replace(' ', '+')
    img_url = img_url + "&zoom=11&size=300x300&maptype=roadmap"
    img_url = img_url + "&markers=color:blue%7Clabel:" + str((0)) + "%7C" + str(start_lat) + "," + str(start_long)
    for idx, place in enumerate(route[1:-1]):
        img_url = img_url + "&markers=label:" + str((idx + 1)) + "%7C" + str(top_info['latitude'].iloc[place - 2]) + ","
        img_url = img_url + str(top_info['longitude'].iloc[place - 2])
    img_url = img_url + "&key="+api_key
    return(maps_url, img_url)
    
