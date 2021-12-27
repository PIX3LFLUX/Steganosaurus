""" string version """
import numpy as np
from PIL import Image

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits,2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

    
""" int version """
def text_to_bits_int(text, gain):
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


def text_from_bits_int(bits):
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
def message2bin(message, threshold):
    digital = np.zeros(len(message)).astype('uint8')
    for ix, m in enumerate(message):
        if m > threshold:
            digital[ix] = 1
        else:
            digital[ix] = 0
    return digital


def create_FFTmask(xlen, ylen, message):
    # calculate minimum part to be cut and add another 3% because of rounding errors and for "safety"
    cut = np.sqrt(2*len(message)/(ylen*xlen))*1.03

    #cut off high frequencies from R channel
    mask = np.full((ylen, xlen), True)
    row_start = round(ylen/2*(1-cut))
    row_stop = round(ylen/2*(1+cut))
    col_start = round(xlen/2*(1-cut))
    col_stop = round(xlen/2*(1+cut))
    mask[row_start:row_stop, col_start:col_stop] = False  # rectangular

    return mask, cut


def embedBin2FFT(cover_channel, mask, message):
    fft = np.fft.fft2(cover_channel)
    fft_abs = np.abs(fft)
    ylength, xlength = fft_abs.shape
    
    # write hidden message into filtered absolute part
    message_len = len(message)
    counter=0
    for i in range(ylength):
        # if cover_rows == 50:
        #     print("max values", np.max(cover_r_fft_abs[i]))
        for j in range(xlength):
            # write where coefficients are zero -> previously filtered out.
            if mask[i,j]==0:
                if counter==message_len:
                    break
                # write hidden message inside absolute part by overwriting coefficients where the mask is 0
                fft_abs[i,j]=message[counter]
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

    #IFFT on R channel. Take filtered absolute and inverse with original phase, imaginary part should be negligable
    cover_r_masked = np.fft.ifft2(fft_abs*np.exp(1j*np.angle(fft))).real
    # print(cover_r_masked)
    return cover_r_masked


def calculate_FFTmask(xlength, ylength, cut):
    stego_fft_mask = np.full((ylength, ylength), True)
    row_start = round(xlength/2*(1-cut))
    row_stop = round(xlength/2*(1+cut))
    col_start = round(ylength/2*(1-cut))
    col_stop = round(ylength/2*(1+cut))
    stego_fft_mask[row_start:row_stop, col_start:col_stop] = False  # rectangular

    return stego_fft_mask


def get_message(stego_channel, mask):
    # transform R channel into frequency domain
    stego_r_fft =np.fft.fft2(stego_channel)
    stego_r_fft_abs = np.abs(stego_r_fft)
    ylen, xlen = stego_r_fft_abs.shape

    # calculate message length from mask
    message_length = int(np.count_nonzero(mask == False)//2)

    message=np.zeros(message_length, dtype='uint32')
    counter=0
    for i in range(ylen):
        for j in range(xlen):
            if mask[i,j]==0:
                if counter==message_length:
                    break
                message[counter] = stego_r_fft_abs[i,j]
                counter+=1 

    return message


def steg_encode(cover_img_path, stego_img_path, string, gain):
     # convert utf-8 to binary with 2 bytes prepended for telling length of message
    bin_encoded =  text_to_bits_int(string, gain)

    image = Image.open(cover_img_path)
    # image.load()

    Rot, Grün, Blau= image.split() #split image into its RGB channels

    # create rectangular fft mask
    cover_r_fft_mask, cut = create_FFTmask(*(image.size), bin_encoded)

    # cover_r_fft_masked = np.abs(np.fft.fft2(Rot))*cover_r_fft_mask

    # calculate channel with embedded binary data in frequency domain and reverse fft
    cover_r_masked = embedBin2FFT(Rot, cover_r_fft_mask, bin_encoded)

    # normalize output
    cover_r_masked_norm = convert(cover_r_masked, 0,255, np.uint8)

    # merge layers
    stego =  np.stack((cover_r_masked_norm, Grün, Blau), axis=2).astype('uint8')

    # create steganogram
    stego_img = Image.fromarray(stego)

    stego_img.save(stego_img_path)     #save image as png

    return cut


def steg_decode(stego_img_path, cut):
    stego_img = Image.open(stego_img_path)

    stego_r, stego_g, stego_b = stego_img.split() #split image into its RGB channels

    stego_fft_mask = calculate_FFTmask(*(stego_img.size), cut)

    message = get_message(stego_r, stego_fft_mask)

    # calculate threshold
    threshold = np.max(message)/2

    # convert message values to binary
    binary = message2bin(message, threshold)

    text, _ = text_from_bits_int(binary)

    return text