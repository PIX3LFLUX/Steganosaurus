from telegram import Bot


def sample_responses(input_text):
    usr_msg = str(input_text).lower()

    # return username of bot
    if usr_msg in ("whoami"):
        return Bot.username

    return "I don't speak chinese"


def echo_image(input_img):
    return input_img