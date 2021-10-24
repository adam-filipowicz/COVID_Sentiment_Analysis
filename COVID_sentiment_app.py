import streamlit as st
import pandas as pd
import tweepy as tw
from wordcloud import WordCloud
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure

st.set_page_config(layout="wide")

matplotlib.use('agg')

_lock = RendererAgg.lock

sns.set_style('white')

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

row0_1.title('Analyzing Sentiments of COVID-19 Tweets')

with row0_2:
    st.write('')

row0_2.subheader(
    'A Streamlit web app by Adam Filipowicz'
)

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("""The global COVID-19 pandemic has ravaged our planet for almost two years, and the COVID-19 vaccines 
    continue to be our best bet at ending it. These vaccines may be the greatest scientific achievement of our time, but 
    anti-vaccine movements seem to be halting vaccination campaigns around the world. Understanding how these vaccines 
    and other COVID-related topics are discussed by the general public is thus an important part of the effort to end
    the pandemic. By understanding how the public talks about COVID, government agencies and pro-vaccine groups 
    can change their messaging to assuage fears, combat misinformation, and better reach people who remain vaccine
    hesitant and/or against public health measures that will slow the spread of COVID.""")
    st.markdown("""When COVID related news breaks, people take to social media to voice their opinions. Social media
    apps such as Twitter thus represent a promising avenue to collect public sentiment and analyze what language is 
    being used around COVID and related topics. By collecting and analyzing tweets as news breaks, we can get a
    snapshot of public sentiment during important events.""")
    st.markdown("""**This app scrapes Twitter for recent tweets about COVID-19 and related topics, and then
    analyzes whether those tweets have a positive, negative, or neutral sentiment. Some helpful data visualizations
    are then presented. You can choose from a variety of topics, and then choose how many tweets you'd like to analyze. 
    Thank you for trying it out!**""")
    st.markdown(
        "**To begin, select a topic and how many tweets to analyze:**")


def main():
    # Twitter API Connection #
    auth = tw.OAuthHandler(st.secrets['api_key'], st.secrets['api_key_secret'])
    auth.set_access_token(st.secrets['access_token'], st.secrets['access_token_secret'])
    api = tw.API(auth)

    df = pd.DataFrame(columns=["user_name", "user_followers", "user_verified", "text",
                               "favorites", "retweets"])

    def get_tweets(topic, count):
        i = 0
        for tweet in tw.Cursor(api.search_tweets, q=topic + ' -filter:retweets', count=100, lang='en').items():
            df.loc[i, "user_name"] = tweet.user.name
            df.loc[i, "user_followers"] = tweet.user.followers_count
            df.loc[i, "user_verified"] = tweet.user.verified
            df.loc[i, "text"] = tweet.text
            df.loc[i, "favorites"] = tweet.favorite_count
            df.loc[i, "retweets"] = tweet.retweet_count
            i = i+1
            if i > count:
                break
            else:
                pass

    def clean_tweet(text):
        texts = text
        remove_url = lambda x: re.sub(r'https\S+', '', str(x))
        texts_url = texts.apply(remove_url)
        lower = lambda x: x.lower()
        texts_lc = texts_url.apply(lower)
        rmv_punc = lambda x: x.translate(str.maketrans('', '', string.punctuation))
        texts_punc = texts_lc.apply(rmv_punc)
        update_words = ['covid', '#covid', 'coronavirus', '#coronavirus', 'covid19', '#covid19', 'corona', '#corona',
                        'sars-cov-2', '#sars-cov-2', 'vaccine', 'vaccines', '#vaccine', 'covidvaccine', '#covidvaccine',
                        'vaccinated', 'vaccination', topic, topic.lower()]
        stop_words = set(stopwords.words('english'))
        stop_words.update(update_words)
        remove_words = lambda x: ' '.join([word for word in x.split() if word not in stop_words])
        texts_clean = texts_punc.apply(remove_words)
        df.text = texts_clean

    def analyze_sentiment(text):
        sid = SentimentIntensityAnalyzer()
        ps = lambda x: sid.polarity_scores(x)
        sentiment_scores = text.apply(ps)
        sentiment_df = pd.DataFrame(data=list(sentiment_scores))
        labels = lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative')
        sentiment_df['label'] = sentiment_df.compound.apply(labels)
        return df.join(sentiment_df.label)

    # Collect Input from user :
    row2_spacer1, row2_1, row2_spacer2 = st.columns((.01, .1, .2))
    with row2_1:
        topic = st.selectbox("Topic", (
            'Booster', 'COVID', 'J&J', 'Mandate', 'Mask', 'Moderna', 'mRNA', 'Pfizer', 'Vaccine',
            'Virus', 'Wuhan'
        ), index=1)

        count = st.slider(
            'Number of tweets',
            100, 500
        ) - 1

    if st.button('Get Latest Tweets'):
        with st.spinner("Please wait, tweets are being collected"):
            get_tweets(topic, count)
        st.success('Tweets have been collected!')

        clean_tweet(text=df['text'])
        data = analyze_sentiment(text=df['text'])
        counts_df = data.label.value_counts().reset_index()
        st.write(counts_df)

        # Create visualizations
        st.write('')
        row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
            (.1, 1, .1, 1, .1))

        with row3_1, _lock:
            st.subheader('All Tweets')
            fig = Figure()
            ax = fig.subplots()
            sns.barplot(data=counts_df, x='index', y='label', ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Tweets')
            st.pyplot(fig)

        with row3_2, _lock:
            st.subheader('Verified Users Only')
            fig = Figure()
            ax = fig.subplots()
            verified_counts = data.loc[data['user_verified'] == True].label.value_counts().reset_index()
            if verified_counts.label.sum() == 0:
                st.markdown(
                    '**There were no verified users in the tweets analyzed**'
                )
            else:
                sns.barplot(data=verified_counts, x='index', y='label', ax=ax)
                ax.set_xlabel('Sentiment')
                ax.set_ylabel('Tweets')
                st.pyplot(fig)

        st.write('')
        row4_space1, row4_1, row4_space2, row4_2, row4_space3, row4_3, row4_space4 = st.columns(
            (.1, 1, .1, 1, .1, 1, .1))

        with row4_1, _lock:
            st.subheader('Number of Favorites')
            fig = Figure()
            ax = fig.subplots()
            data_fav = data.groupby(['label']).favorites.sum().reset_index()
            sns.barplot(data=data_fav, x='label', y='favorites', ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Favorites')
            st.pyplot(fig)

        with row4_2, _lock:
            st.subheader('Number of Retweets')
            fig = Figure()
            ax = fig.subplots()
            data_retwt = data.groupby(['label']).retweets.sum().reset_index()
            sns.barplot(data=data_retwt, x='label', y='retweets', ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Retweets')
            st.pyplot(fig)

        with row4_3, _lock:
            st.subheader('Number of Followers')
            fig = Figure()
            ax = fig.subplots()
            data_uf = data.groupby(['label']).user_followers.sum().reset_index()
            sns.barplot(data=data_uf, x='label', y='user_followers', ax=ax)
            ax.set_xlabel('Sentiment')
            ax.set_ylabel('Followers')
            st.pyplot(fig)

        st.write('')
        row5_space1, row5_1, row5_space2, row5_2, row5_space3, row5_3, row5_space4 = st.columns(
            (.1, 2, .05, 2, .05, 2, .1))

        with row5_1, _lock:
            st.subheader('Most Popular Words in All Tweets')
            fig, ax = plt.subplots()
            words = ' '.join([word for word in data['text']])
            word_cloud = WordCloud(width=1000, height=500, random_state=20, max_font_size=120).generate(words)
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)

        with row5_2, _lock:
            st.subheader("Most Popular Words in Positive Tweets")
            fig, ax = plt.subplots()
            positive = data[data['label'] == 'positive']
            words = ' '.join([word for word in positive['text']])
            word_cloud = WordCloud(width=1000, height=500, random_state=20, max_font_size=120).generate(words)
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)

        with row5_3, _lock:
            st.subheader("Most Popular Words in Negative Tweets")
            fig, ax = plt.subplots()
            negative = data[data['label'] == 'negative']
            words = ' '.join([word for word in negative['text']])
            word_cloud = WordCloud(width=1000, height=500, random_state=20, max_font_size=120).generate(words)
            plt.imshow(word_cloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(fig)


if __name__ == '__main__':
    main()

st.markdown('***')
st.markdown("""If you have any feedback on this, please reach out to me on 
[Twitter] (https://twitter.com/adamfilipowicz).""")
