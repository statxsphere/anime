import pandas as pd
import streamlit as st
import plotly.express as pex
# import numpy as np
# import scipy as sp
# from sklearn.metrics.pairwise import cosine_similarity


@st.cache 
def load_animedata():
    return pd.read_csv("data1/finalanime.csv")

anime = load_animedata()
maxmem = int(anime['members'].max())

types = st.sidebar.selectbox('Select your medium:', anime['type'].drop_duplicates())
genres = st.sidebar.selectbox('Select your genre:', anime['genre1'].drop_duplicates())
ratings = st.sidebar.slider('Rating:',min_value=0,max_value=10,value=(0,10))
members = st.sidebar.slider('Members:',min_value=0,max_value=maxmem,value=(0,maxmem),step=1)

@st.cache 
def set_filters():
    df = anime.loc[(anime['type']==types) & (anime['genre1']==genres) & 
                   (anime['rating']==ratings) & (anime['members']==members)]
    return df

@st.cache(allow_output_mutation=True)
def figM():
    anime = set_filters()
    fig = pex.treemap(anime.dropna(how='any'), path=['type','genre1', 'name'], values='members',
                      color='rating', hover_data=['rating','episodes'],
                      color_continuous_scale='RdBu')
    fig.update_layout(title_text="Which Anime Should You Watch?",
                      font_size=10, autosize=False, width=800, height=500, margin=dict(l=20, r=40, t=100, b=20))
    return fig

fig = figM()

st.title("So, what anime should you watch?")
st.write('''I've come up with two complimentary methods to help you plan your anime journey out.
''')
st.write('''### Method 1: Interactive AniMap.

The AniMap is designed for people that are new to anime. It's an easy-to-use, interactive way to go through a plethora of anime to help you find the right one to start (or continue) your journey with. Here's how it works: 

* The map (below) is divided first by medium of consumption (outer most), and then by genre, and then the individual anime (inner most).
* The size of each box represents the number of members, i.e, the popularity of that label.
* The darkness of shade represents how highly the anime was rated.
* You can zoom in to a category by clicking on it. You can zoom out by clicking the outer label on the top end of the map.
* You can hover over any element to get details on it. 
* Navigate through this map to find the kind of anime you feel most like watching.
''')

st.plotly_chart(fig)

st.write('''### Method 2: The Anime Scout.

The Anime Scout is a simple tool, meant for more cultured Otaku that will give you a list of reccomendations based on an anime you already love. You're sure to hit it off with at least one of these.
''')

st.write('''#### How does the Anime Scout work?

* Type the name of an anime you like in the space below.
* Hit the 'Scount Anime!' button, and that's it!

So, what are you waiting for? Let's scout some anime!''')

x = st.text_input('Tell me an anime you like:')
n = st.number_input('How many anime should I scout?', min_value=1,max_value=25,value=10,step=1)
anime_name = anime[anime['name'].str.contains(x, case=False)].sort_values(by='members', ascending=False).reset_index()['name'][0]
@st.cache 
def load_itemsdata():
    # user_sub = pd.read_csv('data1/ratingSub.zip')
    # user_sub = user_sub.apply(pd.to_numeric,downcast='integer')
    # merged = user_sub.merge(anime, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])
    # del user_sub
    # merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)
    # ints = merged.select_dtypes(include=['int'])
    # floats = merged.select_dtypes(include=['float'])
    # converted_int = ints.apply(pd.to_numeric,downcast='integer')
    # convMem = merged.members.astype('int32')
    # converted_flt = floats.apply(pd.to_numeric,downcast='float')
    # merged[converted_int.columns] = converted_int
    # merged[converted_flt.columns] = converted_flt
    # merged['members'] = convMem
    # del ints,floats,converted_int,convMem,converted_flt
    # piv = merged.pivot_table(index=['user_id'], columns=['name'], values='user_rating').apply(pd.to_numeric,downcast='float')
    # del merged
    # piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    # del piv
    # piv_norm.fillna(0, inplace=True)
    # piv_norm = piv_norm.T
    # piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]
    # piv_sparse = sp.sparse.csr_matrix(piv_norm.values) 
    # items = pd.DataFrame(cosine_similarity(piv_sparse), index = piv_norm.index, columns = piv_norm.index)
    # del piv_norm, piv_sparse
    items = pd.read_csv('https://www.dropbox.com/s/29gp0edhsnr25nu/items.csv?dl=1',usecols=['name',anime_name])
    return items

def AnimeScout(x,n):
    items = load_itemsdata()
    anime_name = anime[anime['name'].str.contains(x, case=False)].sort_values(by='members', ascending=False).reset_index()['name'][0]
    count = 1
    st.write('If you like {}, you may also like:\n'.format(anime_name))
    for item in items.sort_values(by = anime_name, ascending = False).name[1:n+1]:
        st.write('No. {}: {}'.format(count, item))
        count +=1

if st.button('Scout Anime!'):
    AnimeScout(x,n)
