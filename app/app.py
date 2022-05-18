import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from classes import SVD_recommender

# get the recommender object trained on the rating dataset
recommender = pickle.load(open("./data/recommender.pkl", "rb"))
print(recommender.recommend_with_itemid(2))

# get the books object containing the book information
books = pd.DataFrame(pickle.load(open("./data/books.pkl", "rb")))
books_subset = books.loc[books.book_id.isin(recommender.item_idx)]

# samples subset of ratings of 500 users
ratings_subset = pd.DataFrame(pickle.load(open("./data/ratings_subset.pkl", "rb")))

def recom_list(item):
    item_line = books_subset.loc[books_subset.original_title == item]
    id_ = item_line.book_id.values[0]
    res = recommender.recommend_with_itemid(item_id=id_)
    res = res.merge(books_subset, left_on="item_id", right_on="book_id").drop(columns="item_id").query("book_id != @id_")
    return res

########################### STREAM LIT CODE ################################

import streamlit as st
st.set_page_config(layout="wide")

st.title("Book recommender system")

with st.container():
    offset = 0
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        option = st.selectbox("List of all books ..", books_subset.original_title.values)
    with col2:
        top_k = st.selectbox("Top K recommendations", [4, 8, 16, 20, 24, 28, 32, 36, 40])
    with col3:
        k = st.selectbox("Hyperparameter", [5, 10, 25, 50, 80, 100, 150])

    if st.button("Show"):
        offset = 0
        recommender.fit(k)
        st.text(f"Number of singular values : {recommender.sigma.shape[0]}")

        results = recom_list(option)

        with st.expander("Readme"):
            st.text("This application makes use of SVD to give relevant recommendations for the book of your choice.")
            st.text("SVD is a mathematical technique for matrix factorization. It helps to map large matrices to smaller dimensions.")
            st.image("https://timmoti.github.io/img/svd_diagram.png", width=300)
            st.text("By mapping the items to latent space of lower dimensions helps to curb out the problems caused by higher dimensions.")
            st.text("In addition to giving recommendations, SVD can be used to predict the ratings of entire rating matrices and fill all missing values.")

        st.markdown("------")

        while offset < top_k:
            cols = st.columns(4)

            for i, col in enumerate(cols):
                with col:
                    with st.container():
                        st.text(results.original_title.iloc[i+offset])
                        st.caption(results.authors.iloc[i + offset])
                        st.caption(f"{results.scores.iloc[i+offset]:.2f}")
                        st.image(results.image_url.iloc[i+offset])
            offset += 4
            st.markdown("------")













