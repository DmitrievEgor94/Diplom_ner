from flask import Flask, request
import sqlite3
import pandas as pd
from waitress import serve
from utils import create_table_if_not_exists
import yaml
import json

app = Flask(__name__)

con = sqlite3.connect("ner_diplom.db", check_same_thread=False)

params = yaml.load(open('params.yaml'), Loader=yaml.Loader)

API_TOKEN = params['telegram_params']['api_token']
HOST_FOR_MODEL_API = params['host_for_model_api']
PORT_FOR_MODEL_API = params['port_for_model_api']
URL_PATH_MODEL_API = params['url_path_model_api']


@app.get("/show_all_data")
def show_table():
    return pd.read_sql('SELECT * FROM violations', con=con).to_html(index=False)


@app.get(f"/{URL_PATH_MODEL_API}")
def use_model():
    text = json.loads(request.data)['text']
    return


if __name__ == '__main__':
    create_table_if_not_exists(con)

    # pd.DataFrame([[pd.to_datetime('today'), 'text', 'Moisov', 'problems']],
    #              columns=['date',  'text', 'name', 'bad_words']).to_sql('violations', if_exists='append', index=False,
    #                                                                     con=con)

    serve(app, host='0.0.0.0', port=PORT_FOR_MODEL_API)

