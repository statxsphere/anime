{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import re\n",
    "import numpy as np\n",
    "import scipy as sp\n",
    "import seaborn as sns\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import gc\n",
    "import operator\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "anime = pd.read_csv('data/finalanime.csv')\n",
    "user = pd.read_csv('data/rating.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>anime_id</th>\n",
       "      <th>name</th>\n",
       "      <th>genre</th>\n",
       "      <th>type</th>\n",
       "      <th>episodes</th>\n",
       "      <th>rating</th>\n",
       "      <th>members</th>\n",
       "      <th>genre1</th>\n",
       "      <th>count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>5863</th>\n",
       "      <td>21</td>\n",
       "      <td>One Piece</td>\n",
       "      <td>Action, Adventure, Comedy, Drama, Fantasy, Sho...</td>\n",
       "      <td>TV</td>\n",
       "      <td>35.977578</td>\n",
       "      <td>8.58</td>\n",
       "      <td>504862</td>\n",
       "      <td>Action</td>\n",
       "      <td>2768</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      anime_id       name                                              genre  \\\n",
       "5863        21  One Piece  Action, Adventure, Comedy, Drama, Fantasy, Sho...   \n",
       "\n",
       "     type   episodes  rating  members  genre1  count  \n",
       "5863   TV  35.977578    8.58   504862  Action   2768  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "anime[anime['name']=='One Piece']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 2065572 entries, 0 to 2065571\n",
      "Data columns (total 3 columns):\n",
      "user_id     int16\n",
      "anime_id    int32\n",
      "rating      int8\n",
      "dtypes: int16(1), int32(1), int8(1)\n",
      "memory usage: 29.5 MB\n"
     ]
    }
   ],
   "source": [
    "# def text_cleaning(text):\n",
    "#     text = re.sub(r'&quot;', '', text)\n",
    "#     text = re.sub(r'.hack//', '', text)\n",
    "#     text = re.sub(r'&#039;', '', text)\n",
    "#     text = re.sub(r'A&#039;s', '', text)\n",
    "#     text = re.sub(r'I&#039;', 'I\\'', text)\n",
    "#     text = re.sub(r'&amp;', 'and', text)\n",
    "    \n",
    "#     return text\n",
    "\n",
    "# anime['name'] = anime['name'].apply(text_cleaning)\n",
    "user['rating'] = user['rating'].apply(lambda x: 0 if x == -1 else x)\n",
    "user_sub = user[user['user_id']<20000]\n",
    "user_sub = user_sub.apply(pd.to_numeric,downcast='integer')\n",
    "user_sub.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "38"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "merged = user_sub.merge(anime, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])\n",
    "merged.rename(columns = {'rating_user':'user_rating'}, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 2065522 entries, 0 to 2065521\n",
      "Data columns (total 11 columns):\n",
      "user_id        int16\n",
      "anime_id       int32\n",
      "user_rating    int8\n",
      "name           object\n",
      "genre          object\n",
      "type           object\n",
      "episodes       float64\n",
      "rating         float64\n",
      "members        int64\n",
      "genre1         object\n",
      "count          int64\n",
      "dtypes: float64(2), int16(1), int32(1), int64(2), int8(1), object(4)\n",
      "memory usage: 155.6+ MB\n"
     ]
    }
   ],
   "source": [
    "merged.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "95"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 19999 entries, 1 to 19999\n",
      "Columns: 9243 entries, 0 to ◯\n",
      "dtypes: float32(9243)\n",
      "memory usage: 705.3 MB\n"
     ]
    }
   ],
   "source": [
    "piv = merged.pivot_table(index=['user_id'], columns=['name'], values='user_rating').apply(pd.to_numeric,downcast='float')\n",
    "piv.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Note: As we are subtracting the mean from each rating to standardize\n",
    "# all users with only one rating or who had rated everything the same will be dropped\n",
    "\n",
    "# Normalize the values\n",
    "piv_norm = piv.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)\n",
    "\n",
    "\n",
    "# Drop all columns containing only zeros representing users who did not rate\n",
    "piv_norm.fillna(0, inplace=True)\n",
    "piv_norm = piv_norm.T\n",
    "piv_norm = piv_norm.loc[:, (piv_norm != 0).any(axis=0)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "20"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "gc.collect()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "piv_sparse = sp.sparse.csr_matrix(piv_norm.values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "item_similarity = cosine_similarity(piv_sparse)\n",
    "user_similarity = cosine_similarity(piv_sparse.T)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "item_sim_df = pd.DataFrame(item_similarity, index = piv_norm.index, columns = piv_norm.index)\n",
    "user_sim_df = pd.DataFrame(user_similarity, index = piv_norm.columns, columns = piv_norm.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "def top_animes(anime_name):\n",
    "    count = 1\n",
    "    print('If you like {}, you may also like:\\n'.format(anime_name))\n",
    "    for item in item_sim_df.sort_values(by = anime_name, ascending = False).index[1:11]:\n",
    "        print('No. {}: {}'.format(count, item))\n",
    "        count +=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "def top_animes1():\n",
    "    x = input('Enter Anime Name:')\n",
    "    anime_name = anime[anime['name'].str.contains(x, case=False)].sort_values(by='members', ascending=False).reset_index()['name'][0]\n",
    "    count = 1\n",
    "    print('If you like {}, you may also like:\\n'.format(anime_name))\n",
    "    for item in item_sim_df.sort_values(by = anime_name, ascending = False).index[1:11]:\n",
    "        print('No. {}: {}'.format(count, item))\n",
    "        count +=1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter Anime Name:hunter\n"
     ]
    }
   ],
   "source": [
    "x = input('Enter Anime Name:')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "anime_n = anime[anime['name'].str.contains(x, case=False)].sort_values(by='members', ascending=False).reset_index()['name'][0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "If you like Hunter x Hunter (2011), you may also like:\n",
      "\n",
      "No. 1: Fullmetal Alchemist: Brotherhood\n",
      "No. 2: One Punch Man\n",
      "No. 3: Steins;Gate\n",
      "No. 4: Magi: The Kingdom of Magic\n",
      "No. 5: Kiseijuu: Sei no Kakuritsu\n",
      "No. 6: Haikyuu!!\n",
      "No. 7: Shigatsu wa Kimi no Uso\n",
      "No. 8: Shingeki no Kyojin\n",
      "No. 9: Haikyuu!! Second Season\n",
      "No. 10: Tengen Toppa Gurren Lagann\n"
     ]
    }
   ],
   "source": [
    "top_animes(anime_n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Enter Anime Name:fullmetal\n",
      "If you like Fullmetal Alchemist: Brotherhood, you may also like:\n",
      "\n",
      "No. 1: Steins;Gate\n",
      "No. 2: Code Geass: Hangyaku no Lelouch R2\n",
      "No. 3: Death Note\n",
      "No. 4: Tengen Toppa Gurren Lagann\n",
      "No. 5: Code Geass: Hangyaku no Lelouch\n",
      "No. 6: Hunter x Hunter (2011)\n",
      "No. 7: Clannad: After Story\n",
      "No. 8: Fullmetal Alchemist\n",
      "No. 9: Fate/Zero 2nd Season\n",
      "No. 10: Shingeki no Kyojin\n"
     ]
    }
   ],
   "source": [
    "top_animes1()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
