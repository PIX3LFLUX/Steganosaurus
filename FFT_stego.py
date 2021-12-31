""" string version """
import numpy as np
from PIL import Image
import os


# generates the path for the stego image from the name of the cover image and the path the current python file resides on
def stego_path_generator(cover_img_path: str, img_type: str):
    full_name = cover_img_path.split("\\")[-1]
    name = full_name.split(".")[0]
    steg_name = name + "_steg." + img_type
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ImageSources\\Steganograms\\')
    return filename + steg_name

# same as the above, but append _crop to the image name
def crop_path_generator(img_path: str, img_type: str):
    full_name = img_path.split("\\")[-1]
    name = full_name.split(".")[0]
    steg_name = name + "_crop." + img_type
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ImageSources\\Steganograms\\')
    return filename + steg_name

# same as the above, but append _crop to the image name
def resize_path_generator(img_path: str, img_type: str):
    full_name = img_path.split("\\")[-1]
    name = full_name.split(".")[0]
    steg_name = name + "_resize." + img_type
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'ImageSources\\Steganograms\\')
    return filename + steg_name


# turns a utf-8 string into its binary counterpart
def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

# coverts binary into a readable utf-8 string
def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits,2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

    
# convert a utf-8 string to its digital counterpart.
# prepend 2 bytes of message length to the returned digital message
def text_to_bits_int(text: str, gain: float) -> list:
    string_bits = text_to_bits(text)
    # convert each char to int
    message_bits = [int(i)*gain for i in string_bits]
    # get length of bitstream as bytes (overflow error if too large)
    formated = bin(len(message_bits))[2:]
    while len(formated) < 16:
        formated = '0' + formated
    message_len = [int(j)*gain for j in formated]
    # append length onto the first 2 bytes of the message
    message_bits = message_len + message_bits
    return message_bits


# convert an array of 1s and 0s to a utf-8 string
def text_from_bits_int(bits: list):
    # convert each element to string
    message_len_bits = bits[:16]
    message_len = ""
    for bit in message_len_bits:
        message_len = message_len + str(bit)
    message_len = int(message_len, 2)
    string_bits = [str(i) for i in bits[16:message_len+16]]
    string_concat = ""
    # concatenate each element to one string
    for string in string_bits:
        string_concat += string
    string_decode = text_from_bits(string_concat)
    return string_decode, message_len


# normalize a channel by its max and min values
def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# convert inverse transformed message to parsable binary message
def message2bin(message_analog, threshold) -> list:
    message_len = len(message_analog)
    digital = np.zeros(message_len).astype('uint8')
    for ix in range(message_len):
        if message_analog[ix] > threshold:
            digital[ix] = 1
        else:
            digital[ix] = 0
    return digital


# create the mask for the absolute fft by either using the default cut or calculating the optimal with the length of the message
def create_FFTmask(columns, rows, message_digital) -> tuple:
    global optcut
    # calculate minimum part to be cut and add another 3% because of rounding errors and for "safety"
    cut = np.sqrt(2*len(message_digital)/(rows*columns))*1.03
    if cut > 0.7:
        raise Exception("The message is too large. Major distortions are to be expected.")
    if not optcut:
        if cut > 0.4:
            raise Exception("The message is too large. Major distortions are to be expected.")
        else:
            cut = 0.4

    #cut off high frequencies from R channel
    mask = np.full((rows, columns), True)
    row_start = round(rows/2*(1-cut))
    row_stop = round(rows/2*(1+cut))
    col_start = round(columns/2*(1-cut))
    col_stop = round(columns/2*(1+cut))
    mask[row_start:row_stop, col_start:col_stop] = False  # rectangular

    return mask, cut


# embed the message into the absolute fft and inverse the fourier transform to recover the channel
def embedBin2FFT(cover_channel, mask, message_digital):
    fft = np.fft.fft2(cover_channel)
    fft_abs = np.abs(fft)
    rows, cols = fft_abs.shape
    
    # cache message length
    message_len = len(message_digital)
    counter=0
    for i in range(rows):
        # if cover_rows == 50:
        #     print("max values", np.max(cover_r_fft_abs[i]))
        for j in range(cols):
            # write where coefficients are zero -> previously filtered out.
            if mask[i,j]==0:
                if counter==message_len:
                    break
                # write hidden message inside absolute part by overwriting coefficients where the mask is 0
                fft_abs[i,j]=message_digital[counter]
                # print(cover_r_fft_abs[i,j])
                counter+=1

    # # mirror reverse loop
    # counter = 0
    # for i in range(cover_rows-1, -1, -1):
    #     for j in range(cover_cols-1, -1, -1):
    #         if cover_r_fft_mask[i,j]==0:
    #             if counter < len(bin_encoded):
    #                 cover_r_fft_abs[i,j]=bin_encoded[counter]
    #                 counter+=1

    #IFFT on single channel. Take filtered absolute and inverse with original phase, imaginary part should be negligable
    cover_masked = np.fft.ifft2(fft_abs*np.exp(1j*np.angle(fft))).real
    # print(cover_r_masked)
    return cover_masked


# calculate and return mask, default cut = 0.4
def calculate_FFTmask(columns, rows, cut = None):
    if not cut:
        cut = 0.4
    stego_fft_mask = np.full((rows, columns), True)
    row_start = round(rows/2*(1-cut))
    row_stop = round(rows/2*(1+cut))
    col_start = round(columns/2*(1-cut))
    col_stop = round(columns/2*(1+cut))
    stego_fft_mask[row_start:row_stop, col_start:col_stop] = False  # rectangular

    return stego_fft_mask


# returns a list with analog values (int), by shifting the channel into the frequency domain and looking at every pixel where the mask is 0
def get_message(stego_channel, mask) -> list:
    # transform R channel into frequency domain
    stego_r_fft =np.fft.fft2(stego_channel)
    stego_r_fft_abs = np.abs(stego_r_fft)
    rows, cols = stego_r_fft_abs.shape

    # calculate message length from mask
    message_length = int(np.count_nonzero(mask == False)//2)

    message_analog=np.zeros(message_length, dtype='uint32')
    counter=0
    for i in range(rows):
        for j in range(cols):
            if mask[i,j]==0:
                if counter==message_length:
                    break
                message_analog[counter] = stego_r_fft_abs[i,j]
                counter+=1 

    return message_analog

# ------------------------------------------------------------------------------------------------------------
# some global variables
cover_img_path = ""
stego_img_path = ""
optcut = None
message = ""

# image path setter
def set_img_path(cover_path, stego_path):
    global cover_img_path
    global stego_img_path
    cover_img_path = cover_path
    stego_img_path = stego_path

# enable optimal cut
def set_optcut(enable: bool):
    global optcut
    if enable:
        optcut = True
    else:
        optcut = None

# pass message string
def set_message(string):
    global message
    message = string

# encodes string into abs fft of the image previously declared with set_img_path().
# encoding happens with a specific gain and cut value
# the default cut value is 0.4, but an option for optcut can be passed and an optimal cut value will be calculated, which has to be passed to the receiver later on.
def steg_encode(gain: int) -> float:
    global message
    global cover_img_path
    global stego_img_path
     # convert utf-8 to binary with 2 bytes prepended for telling length of message
    bin_encoded =  text_to_bits_int(message, gain)

    image = Image.open(cover_img_path)
    # image.load()

    Rot, Grün, Blau= image.split() #split image into its RGB channels

    # create rectangular fft mask
    cover_r_fft_mask, cut = create_FFTmask(*(image.size), bin_encoded)

    # cover_r_fft_masked = np.abs(np.fft.fft2(Rot))*cover_r_fft_mask

    # calculate channel with embedded binary data in frequency domain and reverse fft
    cover_r_masked = embedBin2FFT(Rot, cover_r_fft_mask, bin_encoded)

    # normalize output
    cover_r_masked_norm = np.clip(cover_r_masked, 0,255)
    # cover_r_masked_norm = convert(cover_r_masked, 0,255, np.uint8)

    # merge layers
    stego =  np.stack((cover_r_masked_norm, Grün, Blau), axis=2).astype('uint8')

    # create steganogram
    stego_img = Image.fromarray(stego)

    stego_img.save(stego_img_path)     #save image as png

    if cut==0.4:
        return None
    return cut

# decodes the message from the abs fft of the previously passed image (stego). if the steganogram was created with a specific cut value,
# this can also be passed. default is cut=0.4
def steg_decode(cut: float=None) -> str:
    global stego_img_path
    stego_img = Image.open(stego_img_path)

    stego_r, stego_g, stego_b = stego_img.split() #split image into its RGB channels

    stego_fft_mask = calculate_FFTmask(*(stego_img.size), cut)

    message_analog = get_message(stego_r, stego_fft_mask)

    # calculate threshold
    threshold = np.max(message_analog)/2

    # convert message values to binary
    binary = message2bin(message_analog, threshold)

    text, _ = text_from_bits_int(binary)

    return text


# goes through the whole encoding and decoding process once. returns text and cut value
def search(gain: float) -> tuple:
    cut = steg_encode(gain)
    # ---------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------TRANSMISSION------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------
    # if optcut is enabled, parameter cut is redundant
    text = steg_decode(cut)
    
    return text, cut


# doubles gain until one encoding and decoding process succeeds. returns gain and previous gain
def gain_booster(gain: int=10000):
    global message
    prev_gain = 0

    # reset text
    Text = ""
    while Text != message:
        try:
            Text, cut = search(gain)
            print("gain\t", gain, "\ttext\t", Text[:10])    
            if Text != message:
                prev_gain = gain
                Text = ""
                gain *= 2
        # except UnicodeDecodeError as err:
        except ValueError as err:
            print("gain\t", gain, "\ttext\t", Text[:10])    
            prev_gain = gain
            Text = ""
            gain *= 2

    return prev_gain, gain, cut


recursive_cnt = 0
success_gain = 0
# find the best gain with recursion. returns only successful gain
def binary_search(low, high, num_recur: int=5) -> float:
    global recursive_cnt
    global success_gain
    global success_text
    global message
    recursive_cnt += 1
    
    if high >= low:
        gain = low + (high - low)//2

        try:
            Text, _ = search(gain)
        except UnicodeDecodeError:
            Text = ""
        except ValueError:
            Text = ""
        print("iteration:", recursive_cnt, "\tgain\t", gain, "\tparsed text:\n", Text[:10])

        if recursive_cnt == num_recur:
            recursive_cnt = 0
            if Text == message:
                return gain
            else:
                return success_gain

        if Text == message:
            # save last successful gain
            success_gain = gain
            # Search the left half
            return binary_search(low, gain-1, num_recur)
            # Search the right half
        else:
            return binary_search(gain + 1, high, num_recur)

    else:
        return -1


# a simple encoder using the default cut value and not improving the gain
def steg_encode_simple(cover_img_path: str, stego_img_path: str, string: str, optcut: bool, recursive_cnt: int=0) -> None:
    set_img_path(cover_img_path, stego_img_path)
    set_message(string)
    set_optcut(optcut)
    prev_gain, gain = gain_booster()[:2]
    if recursive_cnt>0:
        gain = binary_search(prev_gain, gain, recursive_cnt)
    # overwrite last attempt with successful attempt (takes time, since successful attempt was overwritten)
    cut = steg_encode(gain)
    return cut

def steg_decode_simple(stego_img_path: str, cut: float=None) -> None:
    set_img_path(cover_img_path, stego_img_path)
    try:
        text = steg_decode(cut)
    except UnicodeDecodeError:
        print("Message could not be parsed")
        return
    except ValueError:
        print("Message could not be parsed")
        return
    return text