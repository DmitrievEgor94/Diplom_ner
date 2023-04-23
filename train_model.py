import sqlite3
import pandas as pd
from nerus import load_nerus, NERMarkup, Span, Record
import numpy as np
import random
import torch
from pymystem3 import Mystem

from pathlib import Path
import os
import wget
import sys
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer
import nltk
from nltk.corpus import stopwords

import pickle

FOLDER_DATA = 'data'


def set_seeds(random_state=1):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)


def prepare_word_ind_map(file_zip_path, path_to_save):
    docs = load_nerus(file_zip_path)

    words_dic = set()
    i = 0

    for doc in docs:
        doc_ner = doc.ner
        text = doc_ner.text.lower()

        words_dic.update(text.split(' '))
        sys.stdout.write("\r" + f'Processed {i} lines')
        sys.stdout.flush()

        i += 1

    word_dic_map = {word: i + 1 for i, word in enumerate(words_dic)}
    word_dic_map['PAD'] = 1
    word_dic_map['UNKNOWN'] = len(word_dic_map)

    pickle.dump(word_dic_map, open(path_to_save, 'wb'))


def gen_sentences_with_passport(word_map_file_path):
    serial_first = pd.read_csv('data/regions_ind', sep='\t', names=['ind', 'region_name', 'number'])['number'] \
        .astype('str').unique()

    serial_last = [str(el) for el in list(range(90, 99))]
    serial_last.extend(['0' + str(el) for el in range(0, 9)])
    serial_last.extend([str(el) for el in range(10, 24)])

    pass_numbers = []

    for _ in tqdm(range(100000)):
        pass_numbers.append((random.choice(serial_first) + random.choice(serial_last),
                             str(random.randint(100000, 999999))))

    sentences = ['Я достал паспорт с номером {} {}', 'Серия и номер моего паспорта - {} {}',
                 'Гражданин с паспортом серии {} и номером {} пройдите сюда', 'Вот клиент с паспортом {} {}',
                 'Клиент с паспортом {} {}', 'Вот: серия - {}, номер - {}', 'Ну номер - {}, а серия {}']

    ner_mark_up_list = []
    words = set()

    for _ in tqdm(range(100000)):
        pass_number = random.choice(pass_numbers)
        sentence = random.choice(sentences).format(pass_number[0], pass_number[1])
        spans = [Span(start=sentence.find(pass_number[0]), stop=sentence.find(pass_number[0]) + 4, type='PASS_SERIAL'),
                 Span(start=sentence.find(pass_number[1]), stop=sentence.find(pass_number[1]) + 6, type='PASS_NUMBER')]

        ner_mark_up_list.append(NERMarkup(sentence, spans))
        words.update(sentence.split(' '))

    word_ind_map = pickle.load(open(word_map_file_path, 'rb'))
    print(list(word_ind_map.items())[:123])


def bar_progress(current, total):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


if __name__ == '__main__':
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download("stopwords")

    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    print(russian_stopwords)

    folder_path = Path(FOLDER_DATA)
    zip_file_data = folder_path / 'nerus_lenta.conllu.gz'
    word_ind_map_file_path = folder_path / 'word_ind_map.pickle'

    if not os.path.exists(zip_file_data):
        wget.download('https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz',
                      bar=bar_progress, out=str(zip_file_data))

    if not os.path.exists(word_ind_map_file_path):
        prepare_word_ind_map(zip_file_data, word_ind_map_file_path)

    set_seeds()
    gen_sentences_with_passport(word_ind_map_file_path)
