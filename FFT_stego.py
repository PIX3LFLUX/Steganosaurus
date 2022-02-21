""" string version """
from re import I
import numpy as np
from PIL import Image
import os
import configparser
from time import perf_counter
from enum import Enum
from typing import Tuple, List

import copy

# Colorspaces available for embedding/decoding
class Colorspace(Enum):
    RGB     = 0
    YCbCr   = 1

# Image file formats when saving steganogram
class ImageType(Enum):
    png     = 0
    tif     = 1
    webp    = 2

# currently unused
class MessageType(Enum):
    utf8    = 0
    txt     = 1
    pdf     = 2
    gif     = 3

HEADER_LEN = 97

# generates the path for the stego image from the name of the cover image and the path the current python file resides on
def stego_path_generator(cover_img_path: str, img_type: ImageType):
    full_name = str(os.path.basename(cover_img_path))
    name = full_name.rsplit(".", maxsplit=1)[0]
    steg_name = name + "_steg." + img_type.name
    cwdname = os.getcwd()
    if not os.path.exists(cwdname + "\\Steganograms"):
        os.mkdir(os.path.join(cwdname, "Steganograms"))
    filedir = os.path.join(cwdname, "Steganograms")
    return os.path.join(filedir, steg_name)

# same as the above, but append _crop to the image name
def crop_path_generator(img_path: str, img_type: ImageType):
    full_name = img_path.split("\\")[-1]
    name = full_name.split(".")[0]
    steg_name = name + "_crop." + img_type.name
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ImageSources\\Steganograms\\')
    return filename + steg_name

# same as the above, but append _crop to the image name
def resize_path_generator(img_path: str, img_type: ImageType):
    full_name = img_path.split("\\")[-1]
    name = full_name.split(".")[0]
    steg_name = name + "_resize." + img_type.name
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ImageSources\\Steganograms\\')
    return filename + steg_name

# same as the above, but append _crop to the image name
def rotate_path_generator(img_path: str, img_type: ImageType):
    full_name = img_path.split("\\")[-1]
    name = full_name.split(".")[0]
    steg_name = name + "_rotate." + img_type.name
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ImageSources\\Steganograms\\')
    return filename + steg_name

# create a namespace for generating the binary Message. also referred to as "encoding"
def generateMessage(gain: int or list, colorspace: int) -> Tuple[list,list]:

    # skip message generation if already created
    # if text == message:
    #     return full_bin_message_gain, full_bin_message

    # converts a string to its binary representation
    def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
        bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
        return bits.zfill(8 * ((len(bits) + 7) // 8))

    # splits a list into 2/3 pieces, according to the colorspace chosen previously
    def splitList(lst: list) -> list:
        split_lst = []
        if Colorspace(colorspace) == Colorspace.RGB:
            # divide into 3 pieces
            cut_points = findCutPoints(len(lst), 3)
            split_lst.append(lst[:cut_points[0]])
            split_lst.append(lst[cut_points[0]:cut_points[1]])
            split_lst.append(lst[cut_points[1]:])
        else:
            # divide into 2 pieces
            cut_points = findCutPoints(len(lst), 2)
            split_lst.append(lst[:cut_points[0]])
            split_lst.append(lst[cut_points[0]:])
        return split_lst

    # a function for finding the optimal point for splitting a list into sublists of equal length
    def findCutPoints(n,k) -> list:
        q,r = divmod(n-k+1,k)
        bigSteps = list(range(q+1, r*(q+2), q+2))
        littleSteps = list(range(r*(q+2) + q, n, q + 1))
        return bigSteps + littleSteps

    # returns a list with the lengths of the sublists in binary representation.
    def getLengthBinary(split_message: list) -> list:
        message_len = []

        # append length of 0 if first channel is unused (-> YCbCr)
        if len(split_message) == 2:
            message_len.append(0)

        for sm in split_message:
            message_len.append(len(sm))
        # print("message_len", message_len)

        len_binary_gained = []
        for length in message_len:
            # get length of bitstream in binary representation (overflow error if too large)
            len_binary = bin(length)[2:]

            # check if length is under 32 bits
            while len(len_binary) < 32:
                len_binary = '0' + len_binary
            # create int array and multiply with gain
            len_binary_gained.append([int(j) for j in len_binary])
        
        return len_binary_gained
    
    def colorspaceBinary(colorspace: Colorspace) -> Colorspace:
        if (colorspace != Colorspace.RGB) and (colorspace != Colorspace.YCbCr):
            raise ValueError("This colorspace is unknown!")
        return colorspace.value

    # assembles parts of the header previously transformed to binary form (integer list/array)
    def header(len_binary: list, colorspace_bin: int) -> list:
        head = []

        # flatten list and append to a single header list
        head.append(colorspace_bin)
        for len in len_binary:
            for l in len:
                head.append(l)
        # return [gain*h for h in head]
        return head

    # generate header
    binary_text = text_to_bits(message)
    split_bin_text = splitList(binary_text)
    len_binary = getLengthBinary(split_bin_text)
    colorspace_bin = colorspaceBinary(colorspace)
    head = header(len_binary, colorspace_bin)

    # generate message and append header to last channel
    body = []
    for bin_text in split_bin_text:
        body.append([int(s) for s in bin_text])
    full_bin_message = body
    full_bin_message[-1] = [h for h in head] + body[-1]
    
    # apply gain
    if type(gain) == int:
        full_bin_message_gain = [[i*gain for i in fbm] for fbm in full_bin_message]
    elif type(gain) == list:
        full_bin_message_gain = [[i*gain[ix] for i in fbm] for ix, fbm in enumerate(full_bin_message)]
    
    return full_bin_message_gain, full_bin_message

# create a namespace for parsing the binary Message. also referred to as "decoding"
def parseMessage(message: list, threshold: list) -> list:

    # convert inverse transformed message to parseable binary message
    def message2bin(message, threshold: int):
        digital = copy.deepcopy(message)
        for ix, m in enumerate(message):
            for iix in range(len(m)):
                if message[ix][iix] > threshold[ix]:
                    digital[ix][iix] = 1
                else:
                    digital[ix][iix] = 0
        return digital

    # parses the header from the binary message and returns its elements
    def parseHeader(message_bin: list):
        # isolate header channel
        header_body = message_bin[-1][:HEADER_LEN]
        colorspace = header_body[0]

        message_len_bin = []
        message_len_bin.append(header_body[1:HEADER_LEN-32*2])
        message_len_bin.append(header_body[HEADER_LEN-32*2:HEADER_LEN-32])
        message_len_bin.append(header_body[HEADER_LEN-32:HEADER_LEN])

        # convert each element to string to concatenate the array
        parsed_message_len = []
        for channel in message_len_bin:
            message_len = ""
            for bit in channel:
                message_len = message_len + str(bit)
            # convert string to int (base 2) and calculate its decimal representation
            tmp = int(message_len, 2)
            # skip appending if length is 0
            if tmp == 0:
                continue
            parsed_message_len.append(tmp)

        return Colorspace(colorspace), parsed_message_len

    def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
        n = int(bits,2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

    def parseContent(message_binary: list, parsed_message_len: list) -> str:

        string_bits_concat = []
        for ix, mes_bin in enumerate(message_binary):
            string_bits_concat.append([str(i) for i in mes_bin[(ix==(len(parsed_message_len)-1))*HEADER_LEN:(ix==(len(parsed_message_len)-1))*HEADER_LEN+parsed_message_len[ix]]])
             
        string_concat = ""
        string_decoded = ""
        # concatenate each element to one string
        for channel in string_bits_concat:
            for string in channel:
                string_concat += string
        string_decoded += text_from_bits(string_concat)

        return string_decoded

    message_binary = message2bin(message, threshold)
    parsed_colorspace, parsed_message_len = parseHeader(message_binary)

    message_binary_header = []
    for ix, mb in enumerate(message_binary):
        # increase the length at the last channel by the length of the header
        message_binary_header.append(mb[:parsed_message_len[ix]+HEADER_LEN*(ix==(len(message_binary)-1))].tolist())
        
    string_decoded = parseContent(message_binary, parsed_message_len)
    return string_decoded, message_binary_header


# normalize a channel by its max and min values
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# resize image if too large (>1600/900) and return Pillow Image object (not stored yet)
def resize(cover_img_path: str) -> Image:
    image = Image.open(cover_img_path)
    # get real size
    cols, rows = image.size

    # handle exception first
    if (cols < max_col) and (rows < max_row):
        return image

    # calculate aspect ratio
    ratio = cols/rows

    # check if either dimension is greater than 1600:900
    if ratio >= max_col/max_row:
        # resize columns to max (1600) and adjust rows accordingly
        if cols > max_col:
            new_cols = max_col
            new_rows = new_cols//ratio
    else:
        # resize rows to max (900) and adjust columns accordingly
        if rows > max_row:
            new_rows = max_row
            new_cols = int(np.around(new_rows*ratio, decimals=0))

    im_resize = image.resize((int(np.around(new_cols, decimals=0)), int(np.around(new_rows, decimals=0))))
    return im_resize

    
# create the mask for the absolute fft by either using the default cut or calculating the optimal with the length of the message
def create_FFTmask(columns, rows, message_digital) -> tuple:
    global optcut
    global max_cut

    # calculate minimum part to be cut and add another 3% because of rounding errors and for "safety"
    cut = np.sqrt(2*len(message_digital)/(rows*columns))*1.03
    if cut > max_cut:
        raise Exception("The message is too large. Try to increase max cut or decrease message size.")
    elif not optcut:
        cut = max_cut

    #cut off high frequencies from R channel
    mask = np.full((rows, columns), False)
    row_start = np.around(rows/2 * (1.0-cut), decimals=0).astype(np.uint16)
    row_stop =  np.around(rows/2 * (1.0+cut), decimals=0).astype(np.uint16)
    col_start = np.around(columns/2 * (1.0-cut), decimals=0).astype(np.uint16)
    col_stop =  np.around(columns/2 * (1.0+cut), decimals=0).astype(np.uint16)
    mask[row_start:row_stop, col_start:col_stop] = True  # rectangular

    return mask, cut


# embed the message into the absolute fft and inverse the fourier transform to recover the channel
def embedBin2FFT(cover_channel, mask, message_digital):
    fft = np.fft.fft2(cover_channel)
    fft_abs = np.abs(fft)
    
    message_len = len(message_digital)

    masked_fft = fft_abs[mask]
    for ii in range(message_len):
        masked_fft[ii] = message_digital[ii]
    fft_abs[mask] = masked_fft
    
    #IFFT on single channel. Take filtered absolute and inverse with original phase, imaginary part should be negligable
    cover_masked = np.fft.ifft2(np.multiply(fft_abs, np.exp(np.multiply(1j, np.angle(fft))))).real
    return cover_masked


# calculate and return mask, default cut set with set_maxcut()
def calculate_FFTmask(columns, rows, cut: float=None):
    if not cut:
        global max_cut
        cut = max_cut
    stego_fft_mask = np.full((rows, columns), False)
    row_start = np.around(rows/2 * (1.0-cut), decimals=0).astype(np.uint16)
    row_stop =  np.around(rows/2 * (1.0+cut), decimals=0).astype(np.uint16)
    col_start = np.around(columns/2 * (1.0-cut), decimals=0).astype(np.uint16)
    col_stop =  np.around(columns/2 * (1.0+cut), decimals=0).astype(np.uint16)
    stego_fft_mask[row_start:row_stop, col_start:col_stop] = True  # rectangular

    return stego_fft_mask


# returns a list with analog values (int), by shifting the channel into the frequency domain and looking at every pixel where the mask is 0
def get_message(stego_channel, mask) -> list:
    # transform R channel into frequency domain
    stego_fft =np.fft.fft2(stego_channel)
    stego_fft_abs = np.abs(stego_fft)

    # calculate message length from mask -> predicted message length > true message length!
    message_length = int(np.count_nonzero(mask == True)//2)
    # create message buffer
    message_analog=np.zeros(message_length, dtype='uint32')
    
    stego_fft_masked = stego_fft_abs[mask]
    for ii in range(message_length):
        message_analog[ii] = stego_fft_masked[ii]

    return message_analog

# ------------------------------------------------------------------------------------------------------------
# some global variables
cover_img_path = ""
stego_img_path = ""
message = ""
optcut = False
colorspace = ""
max_cut = 0.4
max_row = 900
max_col = 1600
resize_enable = False
recursive_count = 0
image_type = 0
static_gain = 0

# image path setter
def set_img_path(cover_path, stego_path):
    global cover_img_path
    global stego_img_path
    cover_img_path = cover_path
    stego_img_path = stego_path

# pass message string
def set_message(string):
    global message
    message = string

# encodes string into abs fft of the image previously declared with set_img_path().
# encoding happens with a specific gain and cut value
# the default cut value is 0.4, but an option for optcut can be passed and an optimal cut value will be calculated, which has to be passed to the receiver later on.
def steg_encode(gain: int or list) -> Tuple[float, list]:
    global message
    global cover_img_path
    global stego_img_path
    global colorspace
    global resize_enable

    if resize_enable:
        image = resize(cover_img_path)
    else:
        image = Image.open(cover_img_path)
    
    # image.convert(colorspace_dict[colorspace])
    image = image.convert(Colorspace(colorspace).name)

    # image = convert_colorspace(image, 0, colorspace)
    channel = image.split() #split image into its 3 channels

     # convert utf-8 to binary with 2 bytes prepended for telling length of message
    bin_encoded, bin_encoded_raw =  generateMessage(gain, Colorspace(colorspace))

    # create rectangular fft mask
    cover_fft_mask, cut = create_FFTmask(*(image.size), bin_encoded[-1])

    # embed the message into their respective channels by using the mask from the 'header channel'
    cover_masked = []
    for ix, bin_encoded_channel in enumerate(bin_encoded):
        cover_masked.append(embedBin2FFT(channel[ix+3-len(bin_encoded)], cover_fft_mask, bin_encoded_channel))

    # normalize output
    cover_masked_clip = []
    for channel_masked in cover_masked:
        cover_masked_clip.append(np.clip(channel_masked, 0,255).astype("uint8"))
    # cover_r_masked_norm = convert(cover_r_masked, 0,255, np.uint8)

    # merge layers
    if len(bin_encoded) == 2:
        stego =  np.stack((channel[0], cover_masked_clip[0], cover_masked_clip[1]), axis=2).astype('uint8')
    elif len(bin_encoded) == 3:
        stego =  np.stack((cover_masked_clip[0], cover_masked_clip[1], cover_masked_clip[2]), axis=2).astype('uint8')

    # create steganogram
    stego_img = Image.fromarray(stego, Colorspace(colorspace).name).convert(Colorspace.RGB.name)

    stego_img.save(stego_img_path)     #save image as png
    # print("Image saved: " + stego_img_path)

    return cut, bin_encoded_raw

# decodes the message from the abs fft of the previously passed image (stego). if the steganogram was created with a specific cut value,
# this can also be passed. default is cut=0.4
def steg_decode(cut: float=None) -> Tuple[str,list]:
    global stego_img_path
    global colorspace
    stego_img = Image.open(stego_img_path).convert(Colorspace(colorspace).name)

    steg_channel = stego_img.split() #split image into its channels

    stego_fft_mask = calculate_FFTmask(*(stego_img.size), cut)

    # copy analog values into buffer
    raw_message = []
    for steg_ch in steg_channel:
        raw_message.append(get_message(steg_ch, stego_fft_mask))

    # convert message values to binary
    # try decoding all 3 channels first, then decode in YCbCr colorspace
    try:
        # calculate threshold the fast and easy way
        threshold = [np.max(m)/2 for m in raw_message]
        string_decoded, stego_binary = parseMessage(raw_message, threshold)
    except IndexError:
        threshold = [np.max(m)/2 for m in raw_message[1:]]
        string_decoded, stego_binary = parseMessage(raw_message[1:], threshold)

    return string_decoded, stego_binary


# goes through the whole encoding and decoding process once.
# returns text and cut value and also binary representation of the message
def search(gain: int or list) -> Tuple[str, float, list, list]:
    cut, bin_tx_raw = steg_encode(gain)
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------TRANSMISSION------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    text, bin_rx_raw = steg_decode(cut)
    
    return text, cut, bin_tx_raw, bin_rx_raw


# doubles gain until one encoding and decoding process succeeds. returns gain and previous gain
def gain_booster(gain: list):# -> Tuple[list or int, list or int, float]:
    # initialize variables
    global message
    prev_gain = [0]*len(gain)
    bin_tx_raw = [0]*len(gain)
    bin_rx_raw = [1]*len(gain)

    # try to find a gain for which each channel can be decoded properly
    while bin_tx_raw != bin_rx_raw:
        try:
            Text, cut, bin_tx_raw, bin_rx_raw = search(gain)
            if bin_tx_raw != bin_rx_raw:
                prev_gain = gain
                gain = [g*2 for g in gain]
        except UnicodeDecodeError:
            print("No gain found, almost there: ", gain)
            prev_gain = gain
            gain = [g*2 for g in gain]
        except IndexError: #ValueError
            print("No gain found, might take a while: ", gain)
            prev_gain = gain
            gain = [g*2 for g in gain]
        for g in gain:
            if g > 1_000_000:
                raise Exception("Gain is too high, aborting...")
    print("Gain found!: ", gain)
    return prev_gain, gain, cut

# find the best gain with recursion. returns only successful gain
def gain_optimizer(low: list, high: list, num_recur: int) -> list:
    recursive_cnt = 0
    success_gain = [0]*len(high)
    def binary_search(low: list, high: list, num_recur: int) -> list:
        nonlocal recursive_cnt
        nonlocal success_gain
        global message
        recursive_cnt += 1

        # initialize gain
        gain = [0]*len(high)
        # do not enter condition if at least one element of high is not greater equals low
        if not False in [high[ix] >= low[ix] for ix in range(len(high))]:
            for ix in range(len(high)):
                gain[ix] = low[ix] + (high[ix] - low[ix])//2

            try:
                Text, cut, bin_tx_raw, bin_rx_raw = search(gain)
            except UnicodeDecodeError or ValueError as err:
                # generate samples which will evaluate as false when being compared below
                bin_rx_raw = [0]*len(high)
                bin_tx_raw = [1]*len(high)

            # store first gain (also success gain from boost function)
            if recursive_cnt == 1:
                success_gain = high

            # pass the found gain to the outside world
            if recursive_cnt == num_recur:
                if bin_tx_raw == bin_rx_raw:
                    return gain
                else:
                    return success_gain

            # actual binary search
            new_low  = [0]*len(low)
            new_high = [0]*len(high)
            for ix in range(len(high)):
                if bin_tx_raw[ix] == bin_rx_raw[ix]:
                    # save last successful gain
                    success_gain[ix] = gain[ix]
                    # Search the left half
                    new_low[ix] = low[ix]
                    new_high[ix] = gain[ix]-1
                else:
                    new_low[ix] = gain[ix]+1
                    new_high[ix] = high[ix]
            return binary_search(new_low, new_high, num_recur)
        return -1
    if num_recur > 0: 
        return binary_search(low, high, num_recur)
    return high


# a simple encoder using the default cut value and not improving the gain
def steg_encode_simple(cover_img_path: str, string: str) -> tuple:
    load_settings()
    set_img_path(cover_img_path, stego_path_generator(cover_img_path, ImageType(image_type)))
    set_message(string)
    if static_gain == 0:
        if Colorspace(colorspace) == Colorspace.RGB:
            prev_gain, gain, cut = gain_booster([10000]*3)
        elif Colorspace(colorspace) == Colorspace.YCbCr:
            prev_gain, gain, cut = gain_booster([1000]*2)

        if recursive_count > 0:
            gain = gain_optimizer(prev_gain, gain, recursive_count)
        # overwrite last attempt with successful attempt (takes time, since successful attempt was overwritten)
        cut = steg_encode(gain)[0]
    elif static_gain > 0:
        cut = steg_encode(static_gain)[0]
        gain = static_gain
    return cut, gain


# a simple decoder using the optional cut value (secret key) and colorspace
def steg_decode_simple(stego_img_path: str, cut: float=None, colorspace: Colorspace=Colorspace.RGB) -> str:

    set_img_path(cover_img_path, stego_img_path)
    # set_colorspace(colorspace)
    set_settings(colorspace_=colorspace)
    try:
        text = steg_decode(cut)
    except UnicodeDecodeError:
        print("Message could not be parsed")
        return
    except ValueError:
        print("Message could not be parsed")
        return
    return text


# load settings from settings.ini and returns tuple which can be passed to steg_encode_simple by unpack tuple
def load_settings() -> tuple:
    global optcut
    global colorspace
    global max_cut
    global max_row
    global max_col
    global resize_enable
    global recursive_count
    global image_type
    global static_gain

    # load settings
    config = configparser.ConfigParser()
    config.read('settings.ini')
    optcut          = config.getboolean('USER', 'optcut')
    colorspace      = config.getint('USER', 'colorspace')
    max_cut         = config.getfloat('USER', 'max_cut')
    max_row         = config.getint('USER', 'max_row')
    max_col         = config.getint('USER', 'max_col')
    resize_enable   = config.getboolean('USER', 'resize_enable')

    image_type      = config.getint('USER', 'image_type')
    recursive_count = config.getint('USER', 'recursive_count')
    static_gain     = config.getint('USER', 'static_gain')

    return (optcut, colorspace, max_cut, max_row, max_col, resize_enable, recursive_count, image_type, static_gain)


def reset_settings():
    config = configparser.ConfigParser()
    config.read('settings.ini')

    # reset globals
    global optcut
    global colorspace
    global max_cut
    global max_row
    global max_col
    global resize_enable
    optcut = config['DEFAULT']['optcut']
    colorspace = config['DEFAULT']['colorspace']
    max_cut = config['DEFAULT']['max_cut']
    max_row = config['DEFAULT']['max_row']
    max_col = config['DEFAULT']['max_col']
    resize_enable = config['DEFAULT']['resize_enable']

    # reset settings.ini
    config['USER']['optcut']  = config['DEFAULT']['optcut']
    config['USER']['colorspace']  = config['DEFAULT']['colorspace']
    config['USER']['max_cut'] = config['DEFAULT']['max_cut']
    config['USER']['max_row'] = config['DEFAULT']['max_row']
    config['USER']['max_col'] = config['DEFAULT']['max_col']
    config['USER']['resize_enable'] = config['DEFAULT']['resize_enable']
    config['USER']['recursive_count'] = config['DEFAULT']['recursive_count']
    config['USER']['image_type']  = config['DEFAULT']['image_type']
    config['USER']['static_gain'] = config['DEFAULT']['static_gain']

    try:
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)
        print("USER settings reset success!")
    except Exception as e:
        print("Settings could not be reset: ", e)

    return


def set_settings(optcut_: bool=False, colorspace_: Colorspace=Colorspace.RGB, max_cut_: float=0.4, max_row_: int=900, max_col_: int=1600, resize_enable_: bool=False, recursive_count_: int=0, image_type_: ImageType=ImageType.png, static_gain_: int=0):
    config = configparser.ConfigParser()
    config.read('settings.ini')

    config["USER"]["optcut"] = str(optcut_)
    global optcut
    optcut = optcut_

    config["USER"]["colorspace"] = str(colorspace_.value)
    global colorspace
    colorspace = colorspace_.value

    config["USER"]["max_cut"] = str(max_cut_)
    global max_cut
    max_cut = max_cut_

    config["USER"]["max_row"] = str(max_row_)
    global max_row
    max_row = max_row_

    config["USER"]["max_col"] = str(max_col_)
    global max_col
    max_col = max_col_

    config["USER"]["resize_enable"] = str(resize_enable_)
    global resize_enable
    resize_enable = resize_enable_

    config["USER"]["recursive_count"] = str(recursive_count_)
    global recursive_count
    recursive_count = recursive_count_
    
    config["USER"]["image_type"] = str(image_type_.value)
    global image_type
    image_type = image_type_.value

    config["USER"]["static_gain"] = str(static_gain_)
    global static_gain
    static_gain = static_gain_

    try:
        with open('settings.ini', 'w') as configfile:
            config.write(configfile)
        print("USER variable(s) set!")
    except Exception as e:
        print("Setting could not be set: ", e)

    return