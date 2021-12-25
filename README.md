# Steganography
 
This is a project from the University of applied sciences in Karlsruhe, licensed under the GNU license.

The goal of this project is to send text messages hidden inside images using the Discrete Fourier Transform (or rather FFT).
The benefits over 'converntional' LSB modification are that the hidden information is much harder for the human eye to see, since it was hidden in the frequency domain, which corresponds to subtle changes of color in high frequency parts of the stego image. It is also spread much more, which makes it even harder to detect with the naked eye.

But this also comes at a cost:
The payload cannot be as big as that of simple LSB embedding in the spatial domain.
Furthermore, no two images are the same, which changes the behaviour of the frequency domain and can make it harder to embed and later decode information.
To top it off, storing text in a binary manner relies heavily on the 'distinguishability', or distance between two binary values. If even one bit is falsly recognized, the whole letter becomes corrupt, and maybe the message can not even be read anymore. So it is crucial to embed the hidden message with as much 'weight' as possible, since this also correlates with consistency.

Example images will follow...

Example differentiation images will follow...
