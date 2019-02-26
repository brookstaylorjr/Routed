from flask import request, render_template
from errand_code import app

import psycopg2
import requests
import pickle
import numpy as np
import pandas as pd

import routedtools

# A) Connect to appropriate Yelp database
print('Establishing database connection....')
conn = psycopg2.connect(database='yelp_small', user='brooks', password='davidson', host='localhost')


@app.route('/')
@app.route('/index')
@app.route('/input')
def errrand_input():
    return render_template("input.html")

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/output')
def errand_output():
    glove_dict = pickle.load(open("./errand_code/resources/glove_dict.p", "rb"))
    topwords_df = pickle.load(open("./errand_code/resources/bestwords.p", "rb"))

    shopping_list = request.args.get('shopping_list').split('\r\n')
    start_address = request.args.get('start_address')

    # 1) Pull address - geocode it and find nearby businesses
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address=' + start_address.replace(' ', '+') + \
          '&key=AIzaSyBiYgVNU_z-2EKbxNTLVS-N6LZKIxgViJc'
    r = requests.get(url)
    # Pull out latitude/longitude
    start_lat = r.json()['results'][0]['geometry']['location']['lat']
    start_long = r.json()['results'][0]['geometry']['location']['lng']

    # Step 2: Select for businesses/reviews within a given radius of starting address
    radius = (5 / 69)  # in miles, converted to degrees lat/long
    min_length = 6000  # Min number of characters for a given business, ~1000 words

    nearby_stores = routedtools.get_nearby(start_lat, start_long, radius,
                                           't_phoenix', 't_review_phx', conn, min_length)[0]
    topwords_df = topwords_df.loc[nearby_stores['business_id']]  # Filter for relevant word lists

    print("Number of stores found within", radius * 69,
          "mile radius of address:", nearby_stores.shape[0])

    # Step 3: Map my query term(s) to GloVe vectors - calculate cosine distances to corresponding points for businesses
    match_df, shopping_list = routedtools.get_matches(shopping_list, glove_dict, nearby_stores, topwords_df)
    print(match_df.head(5))
    # Steps 4 & 5: turn match list into a routing
    # Aggregate weights" - [SCALING SCORE MAY CHANGE AS MATCHING ALGORITHM CHANGES]
    a = [0.1, 100, 0]  # weight of distance penalty
    b = [10, 1, 1]  # weight of matching penalty
    c = [0, 1, 10]  # weight of rating penalty

    top_choices = []
    top_addresses = []
    top_items = []
    top_stars = []
    all_routes = []
    all_starts = [0]

    all_mapurls = []
    all_imgurls = []

    for i in range(len(a)):
        # Step 4: Build "best" list of stores -> reformulate my optimization as a quadratic knapsack problem
        weights = [a[i], b[i], c[i]]
        top_info = routedtools.knapsack_optimize(match_df, shopping_list,
                                                 nearby_stores, start_lat, start_long, weights)
        # Step 5: brute-force solve the resultant Traveling Salesman Problem.
        nodes = routedtools.make_nodes(top_info['latitude'].tolist(), top_info['longitude'].tolist(),
                                       start_lat, start_long)
        cost, route = routedtools.find_shortest_route(nodes, len(nodes))
        print('Shortest route: {}'.format(route))
        print('Travel cost   : {}'.format(cost * 69))
        # Build maps URLS
        maps_url, img_url = routedtools.build_urls(start_address, route, top_info, start_lat, start_long)

        # Append new results into output vars
        top_choices.extend(top_info['name'].values)
        top_addresses.extend(top_info['address'].values)
        top_items.extend(top_info['items'].values)
        top_stars.extend(top_info['stars'].values)
        all_routes.extend([route[1:-1]])
        all_starts.append(top_info.shape[0] + all_starts[-1])
        all_mapurls.append(maps_url)
        all_imgurls.append(img_url)




    return render_template("output.html",
                           shopping_list = match_df['item'].values,
                           all_choices=match_df['names'].values,
                           all_scores=match_df['scores'].values,
                           top_choices = top_choices,
                           top_addresses = top_addresses,
                           top_items=top_items,
                           top_stars=top_stars,
                           all_routes=all_routes,
                           all_starts=all_starts,
                           all_mapurls=all_mapurls,
                           all_imgurls=all_imgurls)
