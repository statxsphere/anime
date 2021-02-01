import pandas as pd
import streamlit as st
import plotly.express as pex

anime = pd.read_csv("data1/finalanime.csv")
items = pd.read_csv('data1/items.csv')

def AnimeScout(x):
    anime_name = anime[anime['name'].str.contains(x, case=False)].sort_values(by='members', ascending=False).reset_index()['name'][0]
    count = 1
    st.write('If you like {}, you may also like:\n'.format(anime_name))
    for item in items.sort_values(by = anime_name, ascending = False).name[1:11]:
        st.write('No. {}: {}'.format(count, item))
        count +=1

fig = pex.treemap(anime.dropna(how='any'), path=['type','genre1', 'name'], values='members',
                  color='rating', hover_data=['rating','episodes'],
                  color_continuous_scale='RdBu')
fig.update_layout(title_text="Which Anime Should You Watch?",
                  font_size=10, autosize=False, width=800, height=500, margin=dict(l=20, r=40, t=100, b=20))

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

if st.button('Scout Anime!'):
    AnimeScout(x)
