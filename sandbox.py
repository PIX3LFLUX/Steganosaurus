import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

PATH = "test\\jfif\\obi-wan_kenobi.jfif"

""" string version """
def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode(encoding, errors) or '\0'

    
""" int version """
def text_to_bits_int(text):
    string_bits = text_to_bits(text)
    # convert each char to int
    return [int(i) for i in string_bits]

def text_from_bits_int(bits):
    # convert each element to string
    string_bits = [str(i) for i in bits]
    string_concat = ""
    # concatenate each element to one string
    for string in string_bits:
        string_concat += string
    string_decode = text_from_bits(string_concat)
    return string_decode


def show_fft_rock():
    image = cv2.imread("test\\jfif\\the_rock.jfif")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (R,G,B) = cv2.split(image)


    # rot fft
    rt_fft = np.fft.fft2(R)
    rt_fft = np.fft.fftshift(rt_fft)
    rt_mag = 20*np.log(np.abs(rt_fft))

    # grün fft
    gt_fft = np.fft.fft2(G)
    gt_fft = np.fft.fftshift(gt_fft)
    gt_mag = 20*np.log(np.abs(gt_fft))

    # blau fft
    bt_fft = np.fft.fft2(B)
    bt_fft = np.fft.fftshift(bt_fft)
    bt_mag = 20*np.log(np.abs(bt_fft))

    plt.subplot(221)
    plt.imshow(image)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])

    plt.subplot(222)
    plt.imshow(rt_mag, cmap = 'gray')
    plt.title('FFT Red'), plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(gt_mag, cmap = 'gray')
    plt.title('FFT Grün'), plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    plt.imshow(bt_mag, cmap = 'gray')
    plt.title('FFT Blau'), plt.xticks([]), plt.yticks([])

def show_stuffed_bits(data, value, title , color='green'):
    if type(value) == tuple:
        masked_array = np.ma.masked_where(data == (value[0] or value[1]), data)
    else:
        masked_array = np.ma.masked_where(data == value, data)

    #cmap = matplotlib.cm.spring  # Can be any colormap that you want after the cm

    cmap = matplotlib.cm.get_cmap("spring").copy() 
    cmap.set_bad(color='green')

    plt.figure()
    plt.imshow(masked_array, cmap=cmap)
    plt.title(title)

cover = cv2.imread(PATH)
cover_YCrCb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb)
cover_Y, cover_Cr, cover_Cb = cv2.split(cover_YCrCb)

cover_BGR = cv2.cvtColor(cover_YCrCb, cv2.COLOR_YCrCb2BGR)
# cv2.imshow("cover_Cb", cover_Cb)
# cv2.imshow("cover", cover_BGR)
# cv2.waitKey(0)


cover_Cb_fft = np.fft.fft2(cover_Cb)

"""only for visualization"""
cover_Cb_fft_shift = np.fft.fftshift(cover_Cb_fft)

print(cover_Cb_fft_shift[0:20])
cover_Cb_fft_abs = 20*np.log10(np.abs(cover_Cb_fft_shift))
cover_Cb_fft_pha = np.angle(cover_Cb_fft)

plt.subplot(121)
plt.imshow(cover_Cb_fft_abs)
plt.tight_layout()
plt.subplot(122)
plt.imshow(cover_Cb_fft_pha)
plt.tight_layout()

cover_Cb_ifft_shift = np.fft.ifftshift(cover_Cb_fft_shift)
cover_Cb_ifft_abs = np.fft.ifft2(np.abs(cover_Cb_ifft_shift)*np.angle(cover_Cb_fft))

# show_stuffed_bits(cover_Cb_fft_abs, (1,0), "original", 'green')
"""----------------------"""

cover_Cb_sorted=np.sort(np.abs(cover_Cb_fft.reshape(-1)))

keep = 0.1
cover_Cb_fft_tresh = cover_Cb_sorted[int(np.floor((1-keep)*len(cover_Cb_sorted)))]  #keep = 0.9
print(cover_Cb_fft_tresh.shape)
# cover_Cb_fft_mask = np.abs(cover_Cb_fft)<cover_Cb_fft_tresh             #filling all with 0 whats over the threshold, else fill with 1
cover_Cb_fft_mask = np.abs(cover_Cb_fft_shift)>cover_Cb_fft_tresh             #filling all with 0 whats over the threshold, else fill with 1
#cover_Cb_fft_low = cover_Cb_fft*cover_Cb_fft_mask                    #//returns low frequencies
cover_Cb_fft_low = np.abs(cover_Cb_fft_shift)*cover_Cb_fft_mask                    #//returns low frequencies
# print(cover_Cb_fft_mask)
print(cover.shape)
print(cover_Cb_fft_low.shape)

""" only for visuallzation """
# cover_Cb_fft_low_shift = np.fft.fftshift(cover_Cb_fft_low)
result = 20*np.log10(cover_Cb_fft_low)

# show_stuffed_bits(result, (0,1), "filtered image before input", 'green')

""" -------------------- """

string = "apo R E D ist mal wieder am start meine freunde mit einem weiteren cideo jatjajaaajaahahahhahaahh moin leute lksajdflkösajdflösakdflksölskadjflköaslksdjaflksdaflkösalökdflsadfkjlsdafjklsdfajkljkafsdagdkkjghnjkgjaegkdsljsadgöjkladgsjksdagksdgkjlkfgsekaghgauioasiouargsui"
bin_encoded =  text_to_bits_int(string)
#print("bin_encoded:", bin_encoded)
counter = 0
for row in range(cover.shape[0]):
    for col in range(cover.shape[1]):
        if cover_Cb_fft_mask[row,col]==0:
            if counter==len(bin_encoded):
                break
            cover_Cb_fft_low[row,col]=bin_encoded[counter]
            counter += 1

# show_stuffed_bits(20*np.log10(cover_Cb_fft_low), (0,1), "filtered image after input", 'green')

plt.show()

def fft_plot(img):
    image = np.fft.fft2(img)
    Rt_shift=np.fft.fftshift(image)
    plt.figure()
    plt.imshow(20*np.log(np.abs(Rt_shift)), cmap="gray")

def compression(channel):
    Rt=np.fft.fft2(channel)
    Rtsort=np.sort(np.abs(Rt.reshape(-1)))

    for keep in (1, 0.5, 0.05):

        #cut off high frequencies from R channel
        R_tresh = Rtsort[int(np.floor((1-keep)*len(Rtsort)))]
        R_mask = np.abs(Rt)>R_tresh             #filling all with 0 whats over the threshold, else fill with 1
        Rtlow = Rt*R_mask                    #mask R channel with R_ind matrix
        Rlow =np.fft.ifft2(Rtlow).real

        #remerge all 3 channels into one matrix
        rgbArray = np.zeros((image.size[1],image.size[0]), 'uint8') #create empty 3D matrix in the shape of the original image
        rgbArray = Rlow  #fill first dimension with filtered R Channel
        img = Image.fromarray(rgbArray)     #create image from remerged matrix
        fft_plot(img)

        # plt.figure()
        # plt.imshow(img)
        # plt.axis('off')

    plt.show()