import pandas as pd
import streamlit as st
import plotly.express as pex
import numpy as np
# import scipy as sp
# from sklearn.metrics.pairwise import cosine_similarity


@st.cache 
def load_animedata():
    return pd.read_csv("data1/finalanime.csv")

anime = load_animedata()
n_rows = anime.shape[0]
minmem = int(anime['members'].min())
maxmem = int(anime['members'].max())
epmin = float(anime['episodes'].min())
epmax = float(anime['episodes'].max())

# types = st.sidebar.selectbox('Select your medium:', anime['type'].unique())
# genres = anime["genre1"].loc[anime["type"] == types]
# genre_choice = st.sidebar.selectbox('Select your genre:', genres)
# sliders = {
#     "rating": st.sidebar.slider(
#         "Rating:", min_value=0,max_value=10,value=(0,10)
#     ),
#     "members": st.sidebar.slider(
#         "Members", min_value=minmem,max_value=maxmem,value=(minmem,maxmem),step=1
#     ),
#     "episodes": st.sidebar.slider(
#         "Episodes", min_value=epmin,max_value=epmax,value=(epmin,epmax),step=1.0
#     ),
# }
st.sidebar.title('AniMap Filters:')
ratings = st.sidebar.slider("Rating:", min_value=1.0,max_value=10.0,value=(1.0,10.0), step=0.1)
members1 = st.sidebar.slider("Members", min_value=minmem,max_value=maxmem,value=(minmem,maxmem),step=1)
episodes1 = st.sidebar.slider("Episodes", min_value=epmin,max_value=epmax,value=(epmin,epmax),step=1.0)

# filter = np.full(n_rows, True)  # Initialize filter as only True

# for feature_name, slider in sliders.items():
#     # Here we update the filter to take into account the value of each slider
#     filter = (
#         filter
#         & (anime[feature_name] >= slider[0])
#         & (anime[feature_name] <= slider[1])
#     )


anime = anime[(anime.rating.between(ratings[0],ratings[1])) 
               & (anime.members.between(members1[0],members1[1]))
               & (anime.episodes.between(episodes1[0],episodes1[1]))]

@st.cache(allow_output_mutation=True)
def figM(anime):
    fig = pex.treemap(anime.dropna(how='any'), path=['type','genre1', 'name'], values='members',
                      color='rating', hover_data=['rating','episodes'],
                      color_continuous_scale='RdBu')
    fig.update_layout(title_text="Which Anime Should You Watch?",
                      font_size=10, autosize=False, width=800, height=500, margin=dict(l=20, r=40, t=100, b=20))
    return fig

fig = figM(anime)

# if st.sidebar.button('Apply filter.'):
#     anime = anime[filter]
#     fig = figM()

# if st.sidebar.button('Unfilter.'):
#     anime = load_animedata()
    

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

anime = load_animedata()
x = st.text_input('Tell me an anime you like:')
try:
    anime_name = anime[anime['name'].str.contains(x, case=False)].sort_values(by='members', ascending=False).reset_index()['name'][0]
except:
    st.write('This anime is not in the database. Sorry!')
n = st.number_input('How many anime should I scout?', min_value=1,max_value=25,value=10,step=1)

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
    try:
        items = load_itemsdata()
        count = 1
        st.write('If you like {}, you may also like:\n'.format(anime_name))
        for item in items.sort_values(by = anime_name, ascending = False).name[1:n+1]:
            st.write('No. {}: {}'.format(count, item))
            count +=1
    except:
        st.write('Anime named "{}" was not found. Please try again with a different name!'.format(x))
        st.write('''Some tips:
                 
        * Try using the Japanese name. Eg: Yu Yu Hakusho instead of Ghost Fighter.
        * Try using the full name. Eg: Ano Hi Mita Hana instead of AnoHana.
        * Try being mindful of spaces. Eg: Hunter X Hunter instead of HunterXHunter.
        * Try using full forms. Eg: Fullmetal Alchemist instead of FMA.
        * Lastly, not every anime is on the database yet.
                 
        Thank you. Have fun Scouting!''')
    

if st.button('Scout Anime!'):
    AnimeScout(x,n)
