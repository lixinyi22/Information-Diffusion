import os
import re
import ast
import json
import jieba
import pickle
import random
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from gensim import corpora, models
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_KEY = "your api key"
SECRET_KEY = "your secret key"

def ensure_dir(path: str) -> None:
    '''Make sure the directory exists
    
    Args: 
        path(str): the path of the directory
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def read_cascades(cascades_path: str) -> dict:
    """Read the cascades file
    
    Args:
        cascades_path(str): the path of the cascades file
    
    Returns:
        cascades(dict): the cascades of each post in dataset
    """
    cascades = defaultdict(list)
    
    split_char_1 = ' '
    split_char_2 = ','
    
    with open(cascades_path, 'r') as f:
        for line in f:
            chunks = line.strip().split(split_char_1)
            post = int(chunks[0])
            for chunk in chunks:
                if len(chunk.split(split_char_2)) == 2:
                    user, timestamp = chunk.split(split_char_2)
                    cascades[post].append((user, timestamp))
                    
    del_idx = []
    for post, cas in cascades.items():
        if len(cas) < 3:
            del_idx.append(post)
        elif len(cas) > 500:
            cascades[post] = cas[:500]
    for idx in del_idx:
        del cascades[idx]
    return cascades

def filter_dataset(interaction_path: str, dataset_name: str, output_folder: str) -> pd.DataFrame:
    """Filter the Twitter and Weibo dataset
    
    Args:
        interaction_path(str): the path of the interaction file
        dataset_name(str): the name of the dataset
        output_folder(str): the path of the output folder
    
    Returns:
        df(pd.DataFrame): the filtered interaction file
    """
    df = pd.read_csv(interaction_path)
    
    # filter duplicate interactions
    df.drop_duplicates(subset=['user_id', 'post_id'], inplace=True)
    
    # filter cold start users and posts
    if dataset_name == 'Twitter':
        user_count = df['user_id'].value_counts()
        df = df[df['user_id'].isin(user_count[user_count >= 3].index)]
        post_count = df['post_id'].value_counts()
        df = df[df['post_id'].isin(post_count[post_count >= 3].index)]
    elif dataset_name == 'Weibo':
        user_count = df['user_id'].value_counts()
        post_count = df['post_id'].value_counts()
        while post_count.min() < 5 or user_count.min() < 5:
            df = df[df['post_id'].isin(post_count[post_count >= 5].index)]
            df = df[df['user_id'].isin(user_count[user_count >= 5].index)]
            post_count = df['post_id'].value_counts()
            user_count = df['user_id'].value_counts()
            
    df.to_csv(os.path.join(output_folder, 'filtered_interaction.csv'), index=False)
    return df

def interaction2cascade(interaction: pd.DataFrame, output_folder: str) -> dict:
    """Transform the interaction file to cascades file
    
    Args:
        interaction(pd.DataFrame): the filtered interaction file
        output_folder(str): the path of the output folder
    
    Returns:
        cascades(dict): the cascades of each post id in dataset
    """
    interaction.sort_values(by='timestamp', inplace=True)
    
    # extract the cascades of each post id
    cascades = defaultdict(list)
    total_length = 0
    for row in tqdm(interaction.to_dict(orient='records')):
        post = row['post_id']
        user = row['user_id']
        timestamp = row['timestamp']
        cascades[post].append((user, timestamp))
        total_length += 1
    
    total_cascades = len(cascades)
    print(f'Cascades num: {total_cascades}', f'Avg cascade length: {total_length / total_cascades}')
    
    # write the cascades to file
    with open(os.path.join(output_folder, 'cascades.txt'), 'w') as f:
        for post, cas in cascades.items():
            if len(cas) < 3:
                continue
            if len(cas) > 500:
                cas = cas[:500]
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas]) + '\n')

    return cascades

def split_cascades(cascades: dict, output_folder: str) -> None:
    """Split the cascades to train, valid and test
    
    Args:
        cascades(dict): the cascades of each post id in dataset
        output_folder(str): the path of the output folder
    """
    with open(os.path.join(output_folder, 'cascade.txt'), 'w') as f:
        for post, cas in cascades.items():
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas[:-2]]) + '\n')
    with open(os.path.join(output_folder, 'cascadevalid.txt'), 'w') as f:
        for post, cas in cascades.items():
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas[:-1]]) + '\n')
    with open(os.path.join(output_folder, 'cascadetest.txt'), 'w') as f:
        for post, cas in cascades.items():
            f.write(f'{post} ' + ' '.join([f'{user},{timestamp}' for user, timestamp in cas]) + '\n')
    
    # index the user
    users = set()
    for cas in cascades.values():
        users.update([user for user, _ in cas])
    user2id = {user: idx for idx, user in enumerate(users)}
    with open(os.path.join(output_folder, 'u2idx.pickle'), 'wb') as f:
        pickle.dump(user2id, f)
    id2user = [user for user, _ in sorted(user2id.items(), key=lambda x: x[1])]
    with open(os.path.join(output_folder, 'idx2u.pickle'), 'wb') as f:
        pickle.dump(id2user, f)
    print(f'User number: {len(user2id)}')  
        
def split_cascades4rec(cascades: dict, output_folder: str) -> None:
    """Split the cascades to train, valid and test for ReChorus
    
    Args:
        cascades(dict): the cascades of each post id in dataset
        output_folder(str): the path of the output folder
    """
    train_df, dev_df, test_df = list(), list(), list()
    for post, cascade in cascades.items():
        for cas in cascade[:-2]:
            train_df.append({'item': cas[0], 'user': post, 'time': cas[1]})
        dev_df.append({'item': cascade[-2][0], 'user': post, 'time': cascade[-2][1]})
        test_df.append({'item': cascade[-1][0], 'user': post, 'time': cascade[-1][1]})
    
    train_df = pd.DataFrame(train_df)
    dev_df = pd.DataFrame(dev_df)
    test_df = pd.DataFrame(test_df)
    
    # index the user and post
    all_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    post2id = {post: idx+1 for idx, post in enumerate(all_df['user'].unique())}
    user2id = {user: idx+1 for idx, user in enumerate(all_df['item'].unique())}
    for type, df in {'train': train_df, 'dev': dev_df, 'test': test_df}.items():
        df['user_id'] = df['user'].map(post2id)
        df['item_id'] = df['item'].map(user2id)
        df.drop(columns=['item', 'user'], inplace=True)
        df.sort_values(by=['time'], inplace=True)
        df.to_csv(os.path.join(output_folder, f'{type}.csv'), index=False, sep='\t')
    user_meta = pd.DataFrame({'item_id': list(user2id.values()), 'user_name': list(user2id.keys())})
    user_meta.to_csv(os.path.join(output_folder, f'item_meta.csv'), index=False, sep='\t')
    post_meta = pd.DataFrame({'user_id': list(post2id.values()), 'post_name': list(post2id.keys())})
    post_meta.to_csv(os.path.join(output_folder, f'user_meta.csv'), index=False, sep='\t')
    print(f'Total User number: {len(user2id)}', f'Total Post number: {len(post2id)}')

def generate_situation_meta(dataset_folder: str) -> None:
    """Generate the situation features
    
    Args:
        dataset_folder(str): the path of the dataset folder
    """
    for dataset in ['train.csv', 'dev.csv', 'test.csv']:
        dataset_path = os.path.join(dataset_folder, dataset)
        df = pd.read_csv(dataset_path, sep='\t')
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        #for col in df.columns:
        #    if col.startswith('c_'):
        #        df.drop(columns=[col], inplace=True)

        df['c_weekday_c'] = df['time'].dt.weekday

        def get_time_range(hour): # according to the Britannica dictionary
            # https://www.britannica.com/dictionary/eb/qa/parts-of-the-day-early-morning-late-morning-etc
            if hour>=5 and hour<=8:
                return 0
            if hour>8 and hour<11:
                return 1
            if hour>=11 and hour<=12:
                return 2
            if hour>12 and hour<=15:
                return 3
            if hour>15 and hour<=17:
                return 4
            if hour>=18 and hour<=19:
                return 5
            if hour>19 and hour<=21:
                return 6
            if hour>21:
                return 7
            return 8 # 0-4 am
        df['c_period_c'] = df['time'].dt.hour.apply(get_time_range)
        df['time'] = df['time'].apply(lambda x: x.timestamp())
        df.to_csv(dataset_path, index=False, sep='\t')

def edges2item_meta(edges_path: str, item_meta_path: str):
    """Transfor the edges to the item meta
    
    Args:
        edges_path(str): the path of the edges file
        item_meta_path(str): the path of the item meta file
    """
    user_meta = pd.read_csv(item_meta_path, sep='\t')
    user_meta['user_name'] = user_meta['user_name'].astype(str)
    
    user_relation = defaultdict(list)
    idx2user = user_meta.set_index('item_id')['user_name'].to_dict()
    user2idx = {user: idx for idx, user in idx2user.items()}
    with open(edges_path, 'r') as f:
        for line in f:
            user, friend = line.strip().split(',')
            if user in user2idx and friend in user2idx:
                user_relation[user2idx[user]].append(user2idx[friend])
                user_relation[user2idx[friend]].append(user2idx[user])
    user_relation = pd.DataFrame(user_relation.items(), columns=['item_id', 'r_friends'])
    user_meta = pd.merge(left=user_meta, right=user_relation, how='left', on='item_id')
    user_meta.fillna('[]', inplace=True)
    user_meta.to_csv(item_meta_path, index=False, sep='\t')

def generate_user_feature_4twitter(user_meta_path: str, post_path: str, interaction_path: str):
    """Generate the user feature for Twitter dataset
    
    Args:
        user_meta_path(str): the path of the user meta file
        post_path(str): the path of the post file
        interaction_path(str): the path of the interaction file
    """
    user_meta = pd.read_csv(user_meta_path, sep='\t')
    user_meta['r_friends'] = user_meta['r_friends'].apply(ast.literal_eval)
    post = pd.read_csv(post_path)
    interaction = pd.read_csv(interaction_path)
    
    # The number of posts of the user
    users_post_count = post['user_id'].value_counts().to_dict()
    user_meta['i_posts_f'] = user_meta['user_name'].apply(lambda x: users_post_count[x] if x in users_post_count else 0)
    # The number of interactions of the user
    user_interaction_count = interaction['user_id'].value_counts().to_dict()
    user_meta['i_interactions_f'] = user_meta['user_name'].apply(lambda x: user_interaction_count[x] if x in user_interaction_count else 0)
    # The number of social relations of the user
    user_meta['i_friends_f'] = user_meta['r_friends'].apply(lambda x: len(x))
    
    user_meta.to_csv(user_meta_path, sep='\t', index=False)

def generate_post_feature_4twitter(post_meta_path: str, post_path: str, interaction_path: str):
    """Generate the post feature for Twitter dataset
    
    Args:
        post_meta_path(str): the path of the post meta file
        interaction_path(str): the path of the log file
    """
    post_meta = pd.read_csv(post_meta_path, sep='\t')
    post = pd.read_csv(post_path)
    interaction = pd.read_csv(interaction_path)
    
    # The number of interactions of the post
    post_interaction_count = interaction['post_id'].value_counts().to_dict()
    post_meta['u_interactions_f'] = post_meta['post_name'].apply(lambda x: post_interaction_count[x] if x in post_interaction_count else 0)
    
    # The length of the post
    post.set_index('post_id', inplace=True)
    post_length = post['length'].to_dict()
    post_meta['u_text_f'] = post_meta['post_name'].apply(lambda x: post_length[x] if x in post_length else 0)
    
    # The sentiment of the post
    '''analyzer = SentimentIntensityAnalyzer()
    def get_sentiment_label(score_dict):
        # pos: 3, neu: 2, neg: 1, unknown: 0
        if score_dict['pos'] > score_dict['neg'] and score_dict['pos'] > score_dict['neu'] and score_dict['compound'] >= 0.5:
            return 3, score_dict['pos']
        elif score_dict['neu'] > score_dict['pos'] and score_dict['neu'] > score_dict['neg'] and score_dict['compound'] > -0.5 and score_dict['compound'] < 0.5:
            return 2, score_dict['neu']
        elif score_dict['neg'] > score_dict['pos'] and score_dict['neg'] > score_dict['neu'] and score_dict['compound'] <= -0.5:
            return 1, score_dict['neg']
        else:
            return 0, 0
    sentiment = [get_sentiment_label(analyzer.polarity_scores(content)) for content in post_meta['rtpost_content'].tolist()]
    post_meta['u_sentiment_c'] = [x[0] for x in sentiment]
    post_meta['u_sentiment_f'] = [x[1] for x in sentiment]'''
    
    post_meta.to_csv(post_meta_path, sep='\t', index=False)
    
def generate_user_feature_4weibo(user_meta_path: str, profile_path: str, interaction_path: str) -> None:
    """Generate the user feature for Weibo dataset
    Args:
        user_meta_path(str): the path of the user meta file
        profile_path(str): the path of the profile file
        interaction_path(str): the path of the interaction file
    """
    user_meta = pd.read_csv(user_meta_path, sep='\t')
    profile = pd.read_csv(profile_path)
    interaction = pd.read_csv(interaction_path)
    
    # The number of posts of the user
    profile.set_index('user_id', inplace=True)
    user_post_count = profile['post_count'].to_dict()
    user_meta['i_posts_f'] = user_meta['user_name'].apply(lambda x: user_post_count[x] if x in user_post_count else 0)
    # The number of interactions of the user
    user_interaction_count = interaction['user_id'].value_counts().to_dict()
    user_meta['i_interactions_f'] = user_meta['user_name'].apply(lambda x: user_interaction_count[x] if x in user_interaction_count else 0)
    # The number of social relations of the user
    def mixed_to_int(value):
        if isinstance(value, str):
            # 匹配数字部分
            match = re.match(r'([\d\.]+)', value)
            if not match:
                return None
            number = float(match.group(1))
            
            # 判断单位
            if '万' in value:
                return int(number * 10_000)
            elif '亿' in value:
                return int(number * 100_000_000)
            else:
                return int(number)
        return None
    profile['follower_count'] = profile['follower_count'].apply(mixed_to_int)
    profile['i_friends_f'] = profile['follower_count'] + profile['followee_count']
    profile.reset_index(inplace=True)
    user_meta = pd.merge(user_meta, profile[['user_id', 'i_friends_f']], left_on='user_name', right_on='user_id', how='left')
    user_meta.fillna(0, inplace=True)
    user_meta.drop(columns=['user_id'], inplace=True)
    user_meta.to_csv(user_meta_path, sep='\t', index=False)

def baidu_sentiment_analysis(text: str):
    def get_access_token():
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))
    
    url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token=" + get_access_token()
    
    pattern_link = r'http[s]?://[^\s<>"]+'
    text = re.sub(pattern_link, '', text)
    payload = json.dumps({
        "text": text
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    if response.status_code == 200:
        response = response.json()
        result = response['items'][0]
        if result['confidence'] > 0.5:
            if result['positive_prob'] > result['negative_prob']:
                return result['sentiment'], result['positive_prob']
            else:
                return result['sentiment'], result['negative_prob']
        else:
            return 1, 0

def generate_post_feature_4weibo(post_meta_path: str, interaction_path: str, post_path: str):
    """Generate the post feature for Weibo dataset
    
    Args:
        post_meta_path(str): the path of the post meta file
        interaction_path(str): the path of the interaction file
        post_path(str): the path of the post file
    """
    post_meta = pd.read_csv(post_meta_path, sep='\t')
    interaction = pd.read_csv(interaction_path)
    post = pd.read_csv(post_path)
    
    # The number of interactions of the post
    post_count = interaction['post_id'].value_counts().to_dict()
    post_meta['u_interactions_f'] = post_meta['post_name'].apply(lambda x: post_count[x] if x in post_count else 0)
    # The length of the post
    def clean_and_count(text):
        emoji_count = len(re.findall(r'\[.*?\]', text))
        text_no_brackets = re.sub(r'\[.*?\]', '', text)
        cleaned_text = text_no_brackets.strip()
        char_count = len(cleaned_text) + emoji_count
        return char_count
    post['u_text_f'] = post['post_content'].apply(clean_and_count)
    post_meta = pd.merge(post_meta, post[['post_id', 'u_text_f', 'post_content']], left_on='post_name', right_on='post_id', how='left')
    post_meta.drop(columns=['post_id'], inplace=True)
    # The sentiment of the post
    '''post_meta['sentiment'] = post_meta['post_content'].apply(baidu_sentiment_analysis)
    post_meta['u_sentiment_c'] = post_meta['sentiment'].apply(lambda x: int(x[0]))
    post_meta['u_sentiment_f'] = post_meta['sentiment'].apply(lambda x: x[1])
    post_meta.drop(columns=['post_id', 'post_content', 'sentiment'], inplace=True)'''
    post_meta.to_csv(post_meta_path, sep='\t', index=False)

def clean(line):
    """Clean the text"""
    rep=['【】','【','】','👍','🤝',
        '🐮','🙏','🇨🇳','👏','❤️','………','🐰','...、、','，，','..','💪','🤓',
         '⚕️','👩','🙃','😇','🍺','🐂','🙌🏻','😂','📖','😭','✧٩(ˊωˋ*)و✧','🦐','？？？？','//','😊','💰','😜','😯',
         '(ღ˘⌣˘ღ)','✧＼٩(눈౪눈)و/／✧','🌎','🍀','🐴',
         '🌻','🌱','🌱','🌻','🙈','(ง•̀_•́)ง！','🉑️','💩',
         '🐎','⊙∀⊙！','🙊','【？','+1','😄','🙁','👇🏻','📚','🙇',
         '🙋','！！！！','🎉','＼(^▽^)／','👌','🆒','🏻',
         '🙉','🎵','🎈','🎊','0371-12345','☕️','🌞','😳','👻','🐶','👄','\U0001f92e\U0001f92e','😔','＋1','🛀','🐸','🐷','➕1',
         '🌚','：：','💉','√','x','！！！','🙅','♂️','💊','👋','o(^o^)o','mei\u2006sha\u2006shi','💉','😪','😱',
         '🤗','关注','……','(((╹д╹;)))','⚠️','Ծ‸Ծ','⛽️','😓','🐵',
         '🙄️','🌕','…','😋','[]','[',']','→_→','💞','😨','&quot;','😁','ฅ۶•ﻌ•♡','😰','🎙️',
         '🤧','😫','(ง•̀_•́)ง','😁','✊','🚬','😤','👻','😣','：','😷','(*^▽^)/★*☆','🐁','🐔','😘','🍋','(✪▽✪)','(❁´ω`❁)','1⃣3⃣','(^_^)／','☀️',
	     '🎁','😅','🌹','🏠','→_→','🙂','✨','❄️','•','🌤','💓','🔨','👏','😏','⊙∀⊙！','👍','✌(̿▀̿\u2009̿Ĺ̯̿̿▀̿̿)✌',
         '😊','👆','💤','😘','😊','😴','😉','🌟','♡♪..𝙜𝙤𝙤𝙙𝙣𝙞𝙜𝙝𝙩•͈ᴗ•͈✩‧₊˚','👪','💰','😎','🍀','🛍','🖕🏼','😂','(✪▽✪)','🍋','🍅','👀','♂️','🙋🏻','✌️','🥳','￣￣)σ',
         '😒','😉','🦀','💖','✊','💪','🙄','🎣','🌾','✔️','😡','😌','🔥','❤','🏼','🤭','🌿','丨','✅','🏥','ﾉ','☀','5⃣⏺1⃣0⃣','🚣','🎣','🤯','🌺',
         '🌸',
         ]
    pattern_1=re.compile('【.*?】')
    pattern_2=re.compile('肺炎@([\u4e00-\u9fa5\w\-]+)')
    pattern_3=re.compile('@([\u4e00-\u9fa5\w\-]+)')
    pattern_4=re.compile(u'[\U00010000-\U0010ffff\uD800-\uDBFF\uDC00-\uDFFF]')
    pattern_5=re.compile('L.*?的微博视频')
    pattern_6=re.compile('（.*?）')
    pattern_link = r'http[s]?://[^\s<>"]+'
    line=line.replace('O网页链接','')
    line=line.replace('-----','')
    line=line.replace('①','')
    line=line.replace('②','')
    line=line.replace('③','')
    line=line.replace('④','')
    line=line.replace('>>','')
    line=re.sub(pattern_1, '', line,0)
    line=re.sub(pattern_2, '', line,0)
    line=re.sub(pattern_3, '', line,0)
    line=re.sub(pattern_4, '', line,0)
    line=re.sub(pattern_5, '', line,0) 
    line=re.sub(pattern_6, '', line,0) 
    line=re.sub(r'\[\S+\]', '', line,0)
    line=re.sub(pattern_link, '', line,0)
    
    for i in rep:
        line=line.replace(i,'')
    return line

def generate_topic(user_meta_path: str, post_meta_path: str, post_path: str, stopword_path: str, num_topic : int = 21) -> None:
    """Generate the topic feature for Weibo dataset
    
    Args:
        user_meta_path(str): the path of the user meta file
        post_meta_path(str): the path of the post meta file
        post_path(str): the path of the post file
        stopword_path(str): the path of the stopword file
        num_topic(int): the number of topics
    """
    user_meta = pd.read_csv(user_meta_path, sep='\t')
    post_meta = pd.read_csv(post_meta_path, sep='\t')
    post = pd.read_csv(post_path)
    
    post_meta = pd.merge(post_meta, post[['post_id', 'post_content']], how='left', left_on='post_name', right_on='post_id')
    posts = post_meta['post_content'].tolist()
    
    # clean and segment the text
    cleaned_posts = [clean(post) for post in posts]
    def seg_sentence(sentence, filepath):
        sentence = re.sub(u'[0-9\.]+', u'', sentence)
        sentence_seged =jieba.cut(sentence.strip(),cut_all=False,use_paddle=10)
        
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        outstr = []
        for word in sentence_seged:
            if word not in stopwords and word.__len__()>1 and word != '\t':
                outstr.append(word)
        return outstr
    seg_posts = [seg_sentence(post, stopword_path) for post in cleaned_posts]
    
    # Create Dictionary
    id2word = corpora.Dictionary(seg_posts)
    corpus = [id2word.doc2bow(text) for text in seg_posts]   # Term Document Frequency
    tfidf = models.TfidfModel(corpus)
    corpus = tfidf[corpus]
    lda_model = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topic, passes=30, random_state=42)
    # lda_model.print_topics()
    post_topic = []
    # Get main topic in each document
    for i, row in enumerate(lda_model[corpus]):
        topic_dist = np.zeros(num_topic)
        for topic_num, prob in row:
            topic_dist[topic_num] = prob
        
        post_topic.append(topic_dist)
    
    post_meta['topic_vector'] = post_topic
    def get_topic(vector):
        sorted_list = sorted(enumerate(vector), key=lambda x: x[1], reverse=True)
        return int(sorted_list[0][0])
    post_meta['u_topic_c'] = post_meta['topic_vector'].apply(get_topic)
    post_meta['u_topic_o'] = post_meta['topic_vector'].apply(list)
    post_meta.drop(columns=['post_id', 'post_content', 'topic_vector'], inplace=True)
    post_meta.to_csv(post_meta_path, sep='\t', index=False)
    
    def get_topic_vector(user):
        user_log = interaction[interaction['user_id'] == user]
        user_post = user_log['post_id'].unique().tolist()
        user_post_vector = post_meta[post_meta['post_name'].isin(user_post)]['u_topic_o'].tolist()
        user_topic = np.mean(user_post_vector, axis=0)
        return list(user_topic)
    user_meta['i_topic_o'] = user_meta['user_name'].apply(get_topic_vector)
    
    def get_topic(vector):
        sorted_list = sorted(enumerate(vector), key=lambda x: x[1], reverse=True)
        max_value, max_index = sorted_list[0][1], sorted_list[0][0]
        second_max_value, second_max_index = sorted_list[1][1], sorted_list[1][0]
        return int(max_index), max_value, int(second_max_index), second_max_value
    user_meta[['i_topic1_c', 'i_topic1_f', 'i_topic2_c', 'i_topic2_f']] = user_meta['i_topic_o'].apply(get_topic).apply(pd.Series)
    user_meta.to_csv(user_meta_path, sep='\t', index=False)

if __name__ == '__main__':
    dataset = 'Douban'
    
    if dataset == 'Twitter' or dataset == 'Weibo':
        interaction_path = f'./{dataset}/origin/interaction.csv'
        output_folder = f'./{dataset}/origin/'
        ensure_dir(output_folder)
        # filter users and posts
        interaction = filter_dataset(interaction_path, dataset, output_folder)
    
        # generate cascade-format dataset
        cascades = interaction2cascade(interaction, output_folder)
    else:
        cascades_path = f'./{dataset}/origin/cascades.txt'
        cascades = read_cascades(cascades_path)
    
    # split the cascades to train, valid and test dataset
    output_folder = f'./{dataset}/IF/'
    ensure_dir(output_folder)
    split_cascades(cascades, output_folder) # edges is same as the social graph, so we can use the edges directly
    
    # split the cascades to train, valid and test dataset
    output_folder = f'./{dataset}/Rec/'
    ensure_dir(output_folder)
    split_cascades4rec(cascades, output_folder)
    
    # generate features for users and posts
    post_meta_path = f'./{dataset}/Rec/user_meta.csv'
    user_meta_path = f'./{dataset}/Rec/item_meta.csv'
    dataset_folder = f'./{dataset}/Rec/'
    profile_path = f'./{dataset}/origin/user.csv'
    post_path = f'./{dataset}/origin/post.csv'
    interaction_path = f'./{dataset}/origin/filtered_interaction.csv'
        
    #generate_situation_meta(dataset_folder)
    
    if dataset == 'Twitter':
        edges_path = f'./{dataset}/IF/edges.txt'
        edges2item_meta(edges_path, user_meta_path)
        generate_user_feature_4twitter(user_meta_path, post_path, interaction_path)
        generate_post_feature_4twitter(post_meta_path, post_path, interaction_path)
    elif dataset == 'Weibo':
        stopwordslist_path = f'./{dataset}/origin/stopwordslist.txt'
        
        generate_user_feature_4weibo(user_meta_path, profile_path, interaction_path)
        generate_post_feature_4weibo(post_meta_path, interaction_path, post_path)
        generate_topic(user_meta_path, post_meta_path, post_path, stopwordslist_path)
    
