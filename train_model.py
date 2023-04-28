import pandas as pd
from nerus import load_nerus, NERMarkup, Span
import numpy as np
import random
import torch
from sklearn.metrics import classification_report, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from pathlib import Path
import os
import wget
import sys

from torch.optim import Adam
from tqdm import tqdm
import fasttext.util
import re
import time

from ner_lstm import NerLSTM

import os

import pickle

FOLDER_DATA = 'data'

MAP_TAG_IND = {
    'OBJ': 0,
    'PER': 1,
    'PASS_NUMBER': 2,
    'PHONE_NUMBER': 3
}

MAX_SENT_LENTGH = 100


def set_seeds(random_state=1):
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    random.seed(random_state)


# def prepare

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


def get_ner_nerus_data(file_zip_path, path_to_save):
    docs = load_nerus(file_zip_path)

    ner_person_list = []

    for i, doc in enumerate(docs):
        ner_person_list.append(doc.ner)

        sys.stdout.write("\r" + f'Processed {i} lines')
        sys.stdout.flush()

    pickle.dump(ner_person_list, open(path_to_save, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


def gen_sentences_with_passport():
    serial_first = pd.read_csv('data/regions_ind', sep='\t', names=['ind', 'region_name', 'number'])['number'] \
        .astype('str').unique()

    serial_last = [str(el) for el in list(range(90, 99))]
    serial_last.extend(['0' + str(el) for el in range(0, 9)])
    serial_last.extend([str(el) for el in range(10, 24)])

    pass_numbers = []

    for _ in tqdm(range(100000)):
        pass_numbers.append((random.choice(serial_first) + random.choice(serial_last),
                             str(random.randint(100000, 999999))))

    sentences = ['Я достал паспорт с номером {} {}, оформи его плиз', 'Серия и номер моего паспорта - {} {}, проверь его',
                 'Гражданин с паспортом серии {} и номером {}, пройдите сюда', 'Вот клиент с паспортом {} {}',
                 'Клиент с паспортом {} {}, не могу с ним ничего сделать', 'Вот: серия - {}, номер - {}',
                 'Ну номер - {}, а серия {}']

    ner_mark_up_list = []
    words = set()

    for _ in tqdm(range(50000)):
        pass_number = random.choice(pass_numbers)
        sentence = random.choice(sentences).format(pass_number[0], pass_number[1])
        spans = [Span(start=sentence.find(pass_number[0]), stop=sentence.find(pass_number[0]) + 4, type='PASS_NUMBER'),
                 Span(start=sentence.find(pass_number[1]), stop=sentence.find(pass_number[1]) + 6, type='PASS_NUMBER')]

        ner_mark_up_list.append(NERMarkup(sentence, spans))
        words.update(sentence.split(' '))

    return ner_mark_up_list


def get_metrics_on_set(model: nn.Module, samples: torch.Tensor, targets: torch.Tensor, batch_size=50):
    predicts_classes_all = []
    predicts_probs_all = []
    targets_all = []

    for i in tqdm(range(0, samples.size()[0] - batch_size, batch_size)):
        res = model(samples[i:i + batch_size])
        res_classes = res.argmax(axis=-1)

        for j in range(batch_size):
            inds = targets[i + j] != -1

            predicts_probs_all.extend(res[j, inds].detach().numpy())
            predicts_classes_all.extend(res_classes[j, inds])
            targets_all.extend(targets[i + j, inds].detach().numpy())

    return pd.DataFrame(classification_report(targets_all, predicts_classes_all, output_dict=True)).T, \
           log_loss(y_true=targets_all, y_pred=predicts_probs_all)

def gen_sentences_with_phone_number():
    first_number = ['+7', '8', '']

    sentences = ['У клиента номер {}', 'Позвонил ему по номеру {}', 'Номерок у него - {}', 'номер у чела - {}',
                 'С телефоном {}', 'Можно позвонить ему по номеру {}', 'Клиент с номером телефона {}, проверь его счет',
                 'Телефон клиента - {}, дозвониться не можем']

    ner_mark_up_list = []

    for _ in tqdm(range(50000)):
        use_bracket = random.randint(0, 1)
        number = random.choice(first_number)
        gen_second_part = random.randint(10, 99)

        if use_bracket:
            number += '(' + '9' + str(gen_second_part) + ')'
        else:
            number += '9' + str(gen_second_part)

        number += str(random.randint(1000000, 9999999))
        sentence = random.choice(sentences).format(number)

        spans = [Span(start=sentence.find(number), stop=sentence.find(number) + len(number),
                      type='PHONE_NUMBER')]

        ner_mark_up_list.append(NERMarkup(sentence, spans))

    return ner_mark_up_list


def get_vectors_and_labels(ner_list):
    random.shuffle(ner_list)

    fasttext.util.download_model('ru', if_exists='ignore')
    ft = fasttext.load_model('cc.ru.300.bin')

    all_vectors = []
    all_labels = []

    part = 1
    interval = len(ner_list) // 10

    for j in tqdm(range(len(ner_list))):
        text = ner_list[j].text
        spans = ner_list[j].spans

        tokens = text.split(' ')
        labels = [0] * len(tokens)

        if len(tokens) > MAX_SENT_LENTGH:
            continue

        ind_start = 0

        word_vectors = []

        for i, token in enumerate(tokens):
            for span in spans:
                if span.start <= ind_start <= span.stop and span.type in MAP_TAG_IND:
                    labels[i] = MAP_TAG_IND[span.type]
                    break

            ind_start += len(token) + 1

            word_vectors.append(torch.tensor(ft.get_word_vector(token), dtype=torch.float).reshape((1, 300)))

        all_vectors.append(torch.cat(word_vectors))

        labels = torch.tensor(labels, dtype=torch.int64)
        all_labels.append(labels)

        if j == part * interval - 1:
            word_vectors_padded = pad_sequence(all_vectors).transpose(1, 0)
            tags_targets = pad_sequence(all_labels, padding_value=-1).transpose(1, 0)

            print(word_vectors_padded.size())
            print(tags_targets.size())

            pickle.dump(word_vectors_padded, open(f'data/all_vectors_{part}.pickle', 'wb'),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(tags_targets, open(f'data/tags_targets_{part}.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            part += 1

            all_vectors = []
            all_labels = []


def bar_progress(current, total):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def load_pickle_files(regex_pattern):
    folder_path = Path(FOLDER_DATA)

    tensor = torch.cat([pickle.load(open(folder_path / file_name, 'rb'))
                        for i, file_name in tqdm(enumerate(sorted(os.listdir(folder_path))))
                        if re.search(regex_pattern, file_name) is not None])

    return tensor


if __name__ == '__main__':
    folder_path = Path(FOLDER_DATA)
    zip_file_data = folder_path / 'nerus_lenta.conllu.gz'
    word_ind_map_file_path = folder_path / 'word_ind_map.pickle'
    ner_person_list_file_path = folder_path / 'ner_list_nerus.pickle'
    ner_person_list_file_path_sampled = folder_path / 'ner_list_nerus_sampled.pickle'

    if not os.path.exists(zip_file_data):
        wget.download('https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz',
                      bar=bar_progress, out=str(zip_file_data))

    if not os.path.exists(ner_person_list_file_path):
        get_ner_nerus_data(zip_file_data, ner_person_list_file_path)

    set_seeds()
    if not os.path.exists(ner_person_list_file_path_sampled):
        ner_person_list = pickle.load(open(ner_person_list_file_path, 'rb'))
        ner_person_list_sampled = random.choices([el for el in ner_person_list
                                                  if 'PER' in [span.type for span in el.spans]
                                                  and len(el.text.split(' ')) < MAX_SENT_LENTGH], k=50000)
        pickle.dump(ner_person_list_sampled, open(ner_person_list_file_path_sampled, 'wb'))
    else:
        ner_person_list_sampled = \
            pickle.load(open(ner_person_list_file_path_sampled, 'rb'))

    # if not os.path.exists(folder_path / 'all_vectors_10.pickle'):
    ner_phone_numbers = gen_sentences_with_phone_number()
    ner_passport_list = gen_sentences_with_passport()

    ner_person_list_sampled.extend(ner_phone_numbers)
    ner_person_list_sampled.extend(ner_passport_list)

    ner_list_full = ner_person_list_sampled

    print("Полное количество сэмплов в датасете:", len(ner_list_full))
    get_vectors_and_labels(ner_list_full)

    start = time.time()

    X = load_pickle_files('all_vectors_\d+')
    tags_targets = load_pickle_files('tags_targets_\d+')

    print(len(X))

    train_ind, test_ind = train_test_split(np.arange(0, len(tags_targets)), test_size=0.2)
    print(train_ind)

    net = NerLSTM(300, 64, len(MAP_TAG_IND))

    target_count = pd.Series(tags_targets.flatten()).value_counts()
    target_count.drop(index=-1, inplace=True)
    target_count.sort_index(inplace=True)

    weights = torch.tensor(target_count.values) / target_count.sum()
    weights = 1. / weights

    loss_func = CrossEntropyLoss(ignore_index=-1, weight=weights, reduction='mean')

    BATCH_SIZE = 100
    EPOCH_NUMBER = 13

    optimizer = Adam(net.parameters())

    train_samples, test_samples = X[train_ind], X[test_ind]
    train_tags, test_tags = tags_targets[train_ind], tags_targets[test_ind]

    for j in range(EPOCH_NUMBER):
        for i in tqdm(range(0, train_samples.shape[0] - BATCH_SIZE, BATCH_SIZE)):
            train_samples_batch, train_targets_batch = train_samples[i:i + BATCH_SIZE], train_tags[i:i + BATCH_SIZE]

            res = net(train_samples_batch)

            loss = loss_func(res.view(-1, len(MAP_TAG_IND)), train_targets_batch.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if j % 2 == 0:
            res_df, loss = get_metrics_on_set(net, test_samples, test_tags, batch_size=120)
            print('\n', res_df)

    pickle.dump(net, open('model.pickle', 'wb'))


