import pickle

from flask import Flask, request
import sqlite3
import pandas as pd
from waitress import serve
from utils import create_table_if_not_exists
import yaml
import json
import fasttext as ft
import torch
from telegram_bot import run_telegram_bot
from multiprocessing import Process


app = Flask(__name__)

con = sqlite3.connect("ner_diplom.db", check_same_thread=False)

params = yaml.load(open('params.yaml'), Loader=yaml.Loader)

API_TOKEN = params['telegram_params']['api_token']
HOST_FOR_MODEL_API = params['host_for_model_api']
PORT_FOR_MODEL_API = params['port_for_model_api']
URL_PATH_MODEL_API = params['url_path_for_model_api']

ft_model = ft.load_model('cc.ru.300.bin')
net = pickle.load(open('model.pickle', 'rb'))

MAP_IND_TAG = {
    0: 'OBJ',
    1: 'PER',
    2: 'PASS_NUMBER',
    3: 'PHONE_NUMBER'
}


@app.get("/show_all_data")
def show_table():
    return pd.read_sql('SELECT * FROM violations', con=con).to_html(index=False)


@app.get(f"/{URL_PATH_MODEL_API}")
def use_model():
    print(request.data)
    text = json.loads(request.data)['text']
    print(text)

    words = text.split(' ')

    input_vec = torch.cat([torch.tensor(ft_model.get_word_vector(word).reshape((1, 300)),
                                        dtype=torch.float) for word in words])

    result = [words[i] for i, el in enumerate(net(input_vec).argmax(dim=1).tolist()) if MAP_IND_TAG[el] != 'OBJ']

    return " ".join(result)


if __name__ == '__main__':
    create_table_if_not_exists(con)

    # pd.DataFrame([[pd.to_datetime('today'), 'text', 'Moisov', 'problems']],
    #              columns=['date',  'text', 'name', 'bad_words']).to_sql('violations', if_exists='append', index=False,
    #                                                                     con=con)
    proc = Process(target=run_telegram_bot)
    proc.start()

    print(23)
    serve(app, host='0.0.0.0', port=PORT_FOR_MODEL_API)

