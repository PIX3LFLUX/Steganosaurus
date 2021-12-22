from telegram.ext import (Updater,
                        MessageHandler,
                        CommandHandler,
                        Filters,
                        CallbackContext,
                        CallbackQueryHandler)
from telegram import Update
from telegram.inline.inlinekeyboardbutton import InlineKeyboardButton
from telegram.inline.inlinekeyboardmarkup import InlineKeyboardMarkup
from telegram.inline.inlinequeryresultphoto import InlineQueryResultPhoto
import Constants as keys
import Responses as R

print("Bot started")

import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)



def start_cmd(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Okaayy, let's go")
    #update.message.reply_text('Type something to get started...')


def help_cmd(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Send a steganogram with /steg\nAnimation and Video is not supported yet")
    #update.message.reply_text('Have you already tried googling?')

# callback data
IMG, GIF, VID = range(3)
# inline keyboard for sending images, gifs, videos
def send_steg(update: Update, context: CallbackContext):
    keyboard = [
        [
            InlineKeyboardButton("Image", callback_data=str(IMG)),
            InlineKeyboardButton("Animation", callback_data=str(GIF)),
            InlineKeyboardButton("Video", callback_data=str(VID)),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    context.bot.send_message(chat_id=update.effective_chat.id, text='Please choose:', reply_markup=reply_markup)
    #update.message.reply_text('Please choose:', reply_markup=reply_markup)

# process image request
def img(update: Update, context: CallbackContext) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    query.answer()
    query.edit_message_text(f"Selected option: {query.data}")
    context.bot.send_photo(update.effective_chat.id, 'https://picsum.photos/')

# process gif request
def gif(update: Update, context: CallbackContext) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    query.answer()
    query.edit_message_text(f"Selected option: {query.data}")
    #context.bot.send_animation(update.effective_chat.id, 'url')
    context.bot.send_animation(update.effective_chat.id, animation=open("ImageSources\\gifs\\matrix-dodge.gif", 'rb')).animation

# process video (mp4) request
def vid(update: Update, context: CallbackContext) -> int:
    """Show new choice of buttons"""
    query = update.callback_query
    query.answer()
    query.edit_message_text(f"Selected option: {query.data}")
    f = open("ImageSources\\mp4\\apored.mp4", 'rb')
    context.bot.send_video(chat_id=update.effective_chat.id, supports_streaming=True, video=f)
    #context.bot.send_message(chat_id=update.effective_chat.id, text="Not supported yet")



# print error to console
def error(update: Update, context: CallbackContext):
    print(f'Update {update} caused error {context.error}')


def main():
    updater = Updater(keys.API_KEY, use_context=True)
    # dispatcher shortcut
    dp = updater.dispatcher


    # attach start and help commands
    dp.add_handler(CommandHandler("start", start_cmd))
    dp.add_handler(CommandHandler("help", help_cmd))
    
    dp.add_handler(CommandHandler('steg', send_steg))
    #dp.add_handler(CallbackQueryHandler(steg_button))
    dp.add_handler(CallbackQueryHandler(img, pattern='^' + str(IMG) + '$'))
    dp.add_handler(CallbackQueryHandler(gif, pattern='^' + str(GIF) + '$'))
    dp.add_handler(CallbackQueryHandler(vid, pattern='^' + str(VID) + '$'))

    # attach error handler
    dp.add_error_handler(error)

    # poll telegram bot endpoint with api-key
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()