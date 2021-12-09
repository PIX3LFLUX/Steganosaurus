import cv2
import numpy as np

PATH = "test\\jfif\\the_rock.jfif"
img = cv2.imread(PATH)
# print(img)
cv2.imshow('image', img)
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image

# converts an image to rgb bitplane. Last element per channel is MSB, first LSB
def img2bitplane(img):
    # Get image dimensions
    height, width, channels = img.shape

    # Splitting the color channels
    b, g, r = cv2.split(img)
    # image with 8 bit integer
    bitdepth = 8
    # converting integer decimal values into 8 bit binary values
    b_bits = np.unpackbits(np.array([b], dtype=np.uint8), axis = 1)
    g_bits = np.unpackbits(np.array([g], dtype=np.uint8), axis = 1)
    r_bits = np.unpackbits(np.array([r], dtype=np.uint8), axis = 1)
    print(b_bits.shape)

    # lists to store sliced bit planes
    b_out = []
    g_out = []
    r_out = []
    # loop over all 8 bits
    for p in range(bitdepth):
        # create binary mask with 1 at position 'p' / (2**p) means 2 to the power of p
        plane = np.full((height, width), 2 ** p, np.uint8)
        # Bitwise and operation to only obtain bits at position 'p'
        b_slice = cv2.bitwise_and(plane,b)
        g_slice = cv2.bitwise_and(plane,g)
        r_slice = cv2.bitwise_and(plane,r)
        
        # save the sliced planes in a list
        b_out.append(b_slice)
        g_out.append(g_slice)
        r_out.append(r_slice)
        
    return b_out, g_out, r_out

# last element is MSB, first LSB
g_bit, b_bit, r_bit = img2bitplane(img)

for ix,bit in enumerate(g_bit):
    for row in range(len(bit)):
        for col in range(len(bit[row])):
            if bit[row][col] > 0:
                bit[row][col] = 255
    cv2.imshow(str(ix), bit)
cv2.waitKey(0)