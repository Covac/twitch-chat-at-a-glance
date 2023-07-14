from twitchio.ext import commands
import requests
import asyncio
import json
import emoji
import gc
from time import sleep
from collections import OrderedDict

import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pattern.en import sentiment

from nltk.corpus import stopwords,wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from nltk import pos_tag

#for tf-idf (The inverse document frequency) and LSA (Latent Semantic Analysis)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer #how much information the word provides
from sklearn.decomposition import TruncatedSVD #analyze relationships between a set of focuments and terms contained within
from operator import itemgetter

#server related
import json
from quart import Quart, render_template, send_from_directory
import config

app = Quart(__name__)
app.config.from_object(config.ServerConfig)

@app.route("/")
async def root():
    return await render_template("main.html")

@app.route("/api")
async def api():
    return JSON, 200, {"Content-Type":"application/json"}
"""
@app.route("/fonts/<path:path>")
async def font(path):
    return await send_from_directory('static', path)"""

@app.route("/images/<path:path>")
async def pin(path):
    return await send_from_directory('static', path)

@app.route("/api/data")
async def data_api():
    return DATA, 200, {"Content-Type":"application/json"}

'''
# Set the URL with query parameters
url = 'https://id.twitch.tv/oauth2/authorize'
params = {
    'response_type': 'token',
    'client_id': config.BotConfig.CLIENT_ID,
    'redirect_uri': 'http://localhost:3000',
    'scope': 'chat:read chat:edit'
}

# Send the GET request and follow the redirects
response = requests.get(url, params=params, allow_redirects=True)

# Get the final URL after redirection
final_url = response.url

# Print the final URL
print(final_url)'''

DATA = {'status': 'Server is online but only started recently!'}
JSON = {'status': 'Server is online but only started recently!'}#I think i might as well have 2 types of data to serve, if i want to make something intensive on frontend 

class Bot(commands.Bot):#myself

    def __init__(self):
        # Initialise our Bot with our access token, prefix and a list of channels to join on boot...
        # prefix can be a callable, which returns a list of strings or a string...
        # initial_channels can also be a callable which returns a list of strings...
        self.cnt = 0
        super().__init__(token=config.BotConfig.SECRET_KEY, prefix='!', initial_channels=['forsen','xqc','nymn','pokelawls','moonmoon','sodapoppin','hasanabi','39daph'])#'covac123','nymn','xqc','forsen','moonmoon','pokelawls','hasanabi','sodapoppin'
        self.ClientID = config.BotConfig.CLIENT_ID
        self.AccessToken = config.BotConfig.SECRET_KEY
        self.WURL = config.BotConfig.WEBHOOK_URL
        self.GlobalAndSubEmotes = []
        self.Channel3rdPartyEmotes = {} #OrderedDict()#hopefully no repeating emotes
        self.EmotesAndLinks = {}#{'EMOTE':{'SIZE':'LINK',}}
        self.Live = []
        self.Tracker = {}
        '''
        {channel:{"logs":[{"message":message,
                            "num":num_of_spam,
                            "who":set(who_spammed)
                            }],
                    "chatters":[{
                        "username":username,
                        "num":number_of_messages}]
                        }}
        #We then empty logs and chatters every sch
        '''
        self.Condensed = {}
        '''
        {'channels': [{string:{
                        'top_sentences': [string],
                        'topics': [{'topic':string,
                                    'times_used': int}],
                        'most_spammed_messages':[{
                            'message': string,
                            'num': int,
                            'who': [string]}],
                        'spammers':[{
                            'username': string,
                            'num': int}]
                        }}]
        }
        '''
                
        self.showchat = False
        self.sanitize = False
        self.loop.create_task(self.refresh_emotes_task(43200))#120 only for testing should be 12( 43200 sec ) or 6 ( 21600 sec ) hours
        self.loop.create_task(self.get_live(60))#these are about 10 sec out of sync
        self.loop.create_task(self.new_task(60))
        print("Starting internal api service")
        self.loop.create_task(app.run_task(host=config.ServerConfig.HOST, port=config.ServerConfig.PORT))#Hypercorn later?

        #tf-idf and LSA
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.lsa = TruncatedSVD(n_components=2,algorithm='arpack')
        


    async def event_ready(self):
        # Notify us when everything is ready!
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')
        print(f'Currently joined channels : {self.connected_channels}')
        for channel in self.connected_channels:
            if type(channel).__name__ == 'Channel':
                print(f'Starting logs for {channel.name}')
                self.Tracker[channel.name] = {'logs':[],'chatters':[]}
        self.start()
        self.send_health_check("Server successfully started!")

    def start(self):#and restart
        self.GlobalAndSubEmotes = []
        self.Channel3rdPartyEmotes = {}
        self.EmotesAndLinks = {}
        params = []
        print(self.connected_channels)
        for channel in self.connected_channels:
            if type(channel).__name__ == 'Channel':
                params.append(('login',channel.name))
        pairs = self.GetChannelID(params)#Returns Dictionaty
        
        for channelID in pairs.values():
            print("Getting Twitch emotes for: ",channelID)
            self.Emotes(channelID)

        print("Loading globals and 3rd party emotes!")
        self.TwitchGlobalEmotes()
        self.LoadBTTVandFFZGlobal()
        for username,userid in pairs.items():
            print(username,userid)
            self.BTTVandFFZChannel(username,userid)

        from sys import getsizeof
        print("Size of Emote&Link buffer in kB: ~", getsizeof(self.EmotesAndLinks)//1024)
                

    def LoadBTTVandFFZGlobal(self):#AND 7TV AND TWITCH
        bttv_response = requests.get('https://api.betterttv.net/3/cached/emotes/global')
        bttv_emotes = bttv_response.json()

        ffz_response = requests.get('https://api.frankerfacez.com/v1/set/global')
        ffz_emotes = ffz_response.json()['sets']['3']['emoticons']  # Set ID 3 is the global set

        seventv_response = requests.get('https://api.7tv.app/v2/emotes/global')
        seventv_emotes = seventv_response.json()

        # Add emote codes for both sets
        for emote in bttv_emotes:
            self.GlobalAndSubEmotes.append(emote['code'])
            self.EmotesAndLinks[emote['code']] = [f'https://cdn.betterttv.net/emote/{emote["id"]}/1x',f'https://cdn.betterttv.net/emote/{emote["id"]}/2x',f'https://cdn.betterttv.net/emote/{emote["id"]}/3x']
            
        for emote in ffz_emotes:
            urlList = []
            self.GlobalAndSubEmotes.append(emote['name'])
            for url in emote['urls'].values():
                urlList.append(url)
            self.EmotesAndLinks[emote['name']] = urlList.copy()
            
        for emote in seventv_emotes:
            self.GlobalAndSubEmotes.append(emote['name'])
            urlList = []
            for url in emote['urls']:
                urlList.append(url[1])
            self.EmotesAndLinks[emote['name']] = urlList.copy()

        print('BTTV, FFZ and 7TV globals added')

    def BTTVandFFZChannel(self,USERNAME,USERID):#AND FUCKING 7TV
        print(f'##### Getting 3rd party emotes for {USERNAME} id:{USERID} #####')
        if USERNAME in self.Channel3rdPartyEmotes.keys():
            pass
        else:
            self.Channel3rdPartyEmotes[USERNAME] = []
            
        response = requests.get(f'https://api.betterttv.net/3/cached/users/twitch/{USERID}')
        if response.status_code == 200:
            print(f'{USERNAME} BTTV EMOTES RETRIEVED SUCCESSFULLY!')
        bttv_emotes = response.json()

        ffz_response = requests.get(f'https://api.frankerfacez.com/v1/room/id/{USERID}')
        if ffz_response.status_code == 200:
            print(f'{USERNAME} FFZ EMOTES RETRIEVED SUCCESSFULLY!')
            #print(ffz_response.headers) #contains info on api limits (Leaky bucket)
        ffz_emotes = json.loads(ffz_response.content)#WHY YOU MIGHT BE WONDERING, because you would get some other kind of api from response.json() because idiots made it!

        response = requests.get(f'https://7tv.io/v3/users/twitch/{USERID}')
        if response.status_code == 200:
            print(f'{USERNAME} 7TV EMOTES RETRIEVED SUCCESSFULLY!')
            seventv_emotes = response.json()
        else:
            print(f'No 7TV emotes for channel: {USERNAME}')
                    
        for emote in bttv_emotes['channelEmotes']:
            #id,code,imageType,animated,userId
            self.Channel3rdPartyEmotes[USERNAME].append(emote['code'])
            self.EmotesAndLinks[emote['code']] = [f'https://cdn.betterttv.net/emote/{emote["id"]}/1x',f'https://cdn.betterttv.net/emote/{emote["id"]}/2x',f'https://cdn.betterttv.net/emote/{emote["id"]}/3x']

        for emote in bttv_emotes['sharedEmotes']:
            #id,code,umageType,animated,user{id,name,displayName,providerId}
            self.Channel3rdPartyEmotes[USERNAME].append(emote['code'])
            self.EmotesAndLinks[emote['code']] = [f'https://cdn.betterttv.net/emote/{emote["id"]}/1x',f'https://cdn.betterttv.net/emote/{emote["id"]}/2x',f'https://cdn.betterttv.net/emote/{emote["id"]}/3x']

        if ffz_response.status_code == 200:
            #try:
            for emoteSet in ffz_emotes["sets"]:
                for emote in ffz_emotes["sets"][str(emoteSet)]["emoticons"]:
                    urlList = []
                    self.Channel3rdPartyEmotes[USERNAME].append(emote['name'])
                    self.EmotesAndLinks[emote['name']] = emote['urls']
                    for url in emote['urls'].values():
                        urlList.append(url)
                    self.EmotesAndLinks[emote['name']] = urlList.copy()

        if response.status_code == 200:#i want it to look nice so i will just repeat it :))))
            try:
                for emote in seventv_emotes['emote_set']['emotes']:
                    #seems they
                    self.Channel3rdPartyEmotes[USERNAME].append(emote['name'])
                    links = []
                    host = 'https:' + emote['data']['host']['url'] + '/'
                    for file in emote['data']['host']['files']:
                        links.append(host+file['name'])
                    self.EmotesAndLinks[emote['name']] = links.copy()#references can go fts
            except:
                print(f"{USERNAME} has no 7TV emotes!")#probably had emotes before but has none now?
                
        print(f"Added 3rd party emotes for: {USERNAME} with id {USERID}")

    def generateParams(self,List,attached_string):
        params = []
        for ins in List:
            params.append((attached_string,ins))
        return params
            
    def GetChannelID(self,usernameList):#max 100 per I think
        localDict = {}
        headers = {
            'Authorization': 'Bearer ' + self.AccessToken,
            'Client-Id': self.ClientID}
        response = requests.get('https://api.twitch.tv/helix/users',headers=headers,params=usernameList, allow_redirects=False)
        data = json.loads(response.content)
        for profile in data["data"]:
            localDict[profile['login']] = profile['id']
        return localDict

    def GetLiveChannels(self,usernameList):#Could be ids as well
        temp = []
        headers = {
            'Authorization': 'Bearer ' + self.AccessToken,
            'Client-Id': self.ClientID}
        response = requests.get('https://api.twitch.tv/helix/streams',headers=headers,params=self.generateParams(usernameList,'user_login'), allow_redirects=False)
        data = json.loads(response.content)
        for live_channel in data["data"]:
            if live_channel["type"] == "live":
                temp.append(live_channel["user_login"])
        self.Live = temp.copy()
        

    def TwitchGlobalEmotes(self):#some "rare" sets might not be included
        headers = {
            'Authorization': 'Bearer ' + self.AccessToken,
            'Client-Id': self.ClientID}
        response = requests.get('https://api.twitch.tv/helix/chat/emotes/global',headers=headers)#includes status codes
        data = json.loads(response.content)
        for emote in data['data']:
            self.GlobalAndSubEmotes.append(emote['name'])
            links = []
            for url in emote['images'].values():
                links.append(url)
            self.EmotesAndLinks[emote["name"]] = links.copy()
            
    def Emotes(self,channel_id):#Channel specific Twitch emotes
        headers = {
            'Authorization': 'Bearer ' + self.AccessToken,
            'Client-Id': self.ClientID}
        response = requests.get('https://api.twitch.tv/helix/chat/emotes',headers=headers,params={'broadcaster_id':channel_id}, allow_redirects=False)
        print(response)
        data = json.loads(response.content)
        for emote in data["data"]:
            self.GlobalAndSubEmotes.append(emote["name"])
            links = []
            for url in emote['images'].values():
                links.append(url)
            self.EmotesAndLinks[emote["name"]] = links.copy()

    def SanitizeEmotes(self,message,channel):
        newcontent = []
        for word in message.split(' '):
            if word in self.GlobalAndSubEmotes or word in self.Channel3rdPartyEmotes[channel]:
                newcontent.append('[EMOTE]')
            else:
                newcontent.append(word)
        newcontent = ' '.join(newcontent)
        return newcontent

    def InformationExtraction(self,chat_messages):#sometimes it can fail when there are exclusively emotes in chat
        #Vectorize using tf-idf
        lsa = TruncatedSVD(n_components=2,algorithm='arpack')#from docs: SVD suffers from a problem called “sign indeterminacy”, which means the sign of the components_ and the output from transform depend on the algorithm and random state. To work around this, fit instances of this class to data once, then keep the instance around to do transformations.
        try:
            X = self.vectorizer.fit_transform(chat_messages)
            #Reducing dimensionality of feature space
            X_lsa = self.lsa.fit_transform(X)
            #Top messages from LSA scores
            top_indices = np.argsort(X_lsa.sum(axis=1))[:-6:-1]
            top_sentences = [chat_messages[i] for i in top_indices]
            return top_sentences
        except Exception as e:
            import traceback
            print(f"Information Extraction failed with {len(chat_messages)}")
            t = traceback.format_exc()
            print(t)
            self.send_health_check(t)
            return []#Better than NoneType

    def RelevantEmotes(self,messages):
        relevant_emotes = {}
        for message in messages:
            for word in message.split():
                for emote,urls in self.EmotesAndLinks.items():
                    if word == emote:
                        relevant_emotes[emote] = urls
                        break           
        return relevant_emotes

    def send_health_check(self,msg):
        hc = {
            'content':msg,
            'username':'twitch-at-a-glance health'}
        requests.post(self.WURL,json=hc)

    async def refresh_emotes_task(self,sch):
        while True:
            await asyncio.sleep(sch)
            self.start()#and restart
            

    async def get_live(self,sch):
        await asyncio.sleep(10)
        while True:
            try:
                self.GetLiveChannels([channel.name for channel in self.connected_channels])
            except:
                print("Error occoured while trying to get live channels!")
            await asyncio.sleep(sch)

    async def new_task(self,sch):
        await asyncio.sleep(20)
        self.Condensed['channels'] = []
        self.Condensed['emotes'] = {}
        #print(self.Tracker)#Just a debug
        while True:
            
            for channel,data in self.Tracker.items():
                messages = [message['message'] for message in data['logs']]
                if len(messages) > 5:#all 'unique' messages
                    only_relevant_messages = []#To avoid unnecessary emotes in API
                    topics = get_topics(messages)
                    formatted_dict_list = []
                    for topic in topics[:10]:#Top 10 topics
                        top,num = topic
                        formatted_dict_list.append({'topic':top,'times_used':num})
                        only_relevant_messages.append(top)

                    #prepare, preprocess before constructing
                    top_sentences = self.InformationExtraction(messages)
                    only_relevant_messages.extend(top_sentences)

                        
                        
                    most_spammed_messages = sorted(data['logs'], key=itemgetter('num'), reverse=True)[:5]
                    #We already have topics
                    
                    only_relevant_messages.extend([m["message"] for m in most_spammed_messages])
                    
                    #Let's construct our json. Yolo with references
                    self.Condensed['channels'].append({channel: {
                                                            'top_sentences': top_sentences,
                                                            'topics': formatted_dict_list,
                                                            'most_spammed_messages': most_spammed_messages,
                                                            'spammers': sorted(data['chatters'], key=itemgetter('num'), reverse=True)[:3],
                                                            'status': 'GOOD',
                                                            'live': 'LIVE' if channel in self.Live else 'OFFLINE'
                                                            }
                                                       }
                                                      )
                    self.Condensed['emotes'].update(self.RelevantEmotes(only_relevant_messages))
                else:
                    self.Condensed['channels'].append({channel: {'status': 'Less than 5 messages.', 'live': 'LIVE' if channel in self.Live else 'OFFLINE'}})
                    print(f'{channel} has less than 5 UNIQUE messages.')
            
            #We reset for next interval
            self.serve_api_and_cleanup()
            print(self.Live)
            await asyncio.sleep(sch)
            
    def serve_api_and_cleanup(self):
        global DATA
        global JSON
        DATA = json.dumps(self.Tracker, indent = 4)
        JSON = json.dumps(self.Condensed, indent = 4)
        print("API is updated!")
        self.Condensed['channels'] = []
        self.Condensed['emotes'] = {}
        for ch,data in self.Tracker.items():
            self.Tracker[ch] = {'logs':[],'chatters':[]}
        gc.collect()
        print("Cleanup done!")
    
        
    #@bot.event
    async def event_message(self, ctx):
        self.cnt += 1
        #Deciding if we want emotes as text or [EMOTE]
        if self.sanitize == True:
            message = self.SanitizeEmotes(ctx.content,ctx.channel.name)
        else:
            message = ctx.content
            
        vader_sentiment,pattern_sentiment = TextSentiment(message)

        #Channel logs here
        skip_flag = False
        for mes in self.Tracker[ctx.channel.name]['logs']:
            if mes['message'] == message:
                mes['num'] += 1
                if not(ctx.author.name in mes['who']):
                    mes['who'].append(ctx.author.name)
                    skip_flag = True
                    break
        if not skip_flag:
            self.Tracker[ctx.channel.name]['logs'].append({'message':message,
                                                           'num':1,
                                                           'who':[ctx.author.name]
                                                           })
        #Channel chatters here
        skip_flag = False
        for spammer in self.Tracker[ctx.channel.name]['chatters']:
            if spammer['username'] == ctx.author.name:
                spammer['num'] += 1
                skip_flag = True
                break
        if not skip_flag:
            self.Tracker[ctx.channel.name]['chatters'].append({'username':ctx.author.name,
                                                               'num':1})
            
                    
        #topics = get_topics(message)
        
        if vader_sentiment['neu'] == 1.0 and pattern_sentiment[0] == 0.0:
            pass
        else:#cool messages only!
            if self.showchat:
                print(f'{self.cnt}|{ctx.channel.name}|{ctx.author.name}:{message}\nVader score: {vader_sentiment} Pattern score: {pattern_sentiment}\n')
            
def TextSentiment(text):
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Tokenize the text
    #tokens = word_tokenize(text)

    # Perform sentiment analysis using Vader
    vader_scores = sia.polarity_scores(text)

    # Perform sentiment analysis using Pattern
    pattern_scores = sentiment(text)

    # Print out results
    #Vader scores: {'neg': 0 to 1, 'neu': 0 to 1, 'pos': 0 to 1, 'compound': -1 to 1}
    #Pattern scores: (polarity{-1.0 neg to 1.0 pos}, subjectivity {0.0 objective to 1.0  subjective})

    return vader_scores, pattern_scores

def get_topics(messages):
    # Preprocess each message and concatenate the tokens
    all_tokens = []
    for message in messages:
        #tokens = preprocess_text(message)
        tokens = word_tokenize(message)
        all_tokens += tokens

    # Perform part-of-speech tagging
    tagged_tokens = pos_tag(all_tokens)

    # Identify noun phrases as topics
    topics = []
    noun_phrases = []
    for (token, pos) in tagged_tokens:
        if pos.startswith('NN'):  # Select nouns and noun phrases
            noun_phrases.append(token)
        elif noun_phrases:  # Reached the end of a noun phrase
            topics.append(' '.join(noun_phrases))
            noun_phrases = []

    # Calculate the frequency distribution of topics
    freq_dist = FreqDist(topics)
    
    # Get the most common topics
    most_common_topics = freq_dist.most_common()

    return most_common_topics


if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    bot = Bot()
    bot.run()
