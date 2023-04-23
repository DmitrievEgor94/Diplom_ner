from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes
import requests

import yaml

params = yaml.load(open('params.yaml'), Loader=yaml.Loader)
API_TOKEN = params['telegram_params']['api_token']
HOST_FOR_MODEL_API = params['host_for_model_api']
PORT_FOR_MODEL_API = params['port_for_model_api']
URL_PATH_MODEL_API = params['url_path_for_model_api']


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    user_name = user.first_name + ' ' + user.last_name
    print(user_name)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)

if __name__ == '__main__':
    application = ApplicationBuilder().token(API_TOKEN).build()

    echo_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echo)

    application.add_handler(echo_handler)
    application.run_polling()
