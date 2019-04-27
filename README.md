## Image Compression/Color Quantization using K-Means

Implemented K-Means algorithm from scratch in C++ for compressing a RGB image  into another image having the *K* most dominant colours. Used opencv for Image reading and processing.

Make sure you have opencv C++ installed, then run,

	g++ KMeansCompressv2.cc -o output `pkg-config --libs opencv`
	./output rain_princess.png rain_princess_compressed.png 16

Command line arguments are `<input_file_path> <output_file_path> <n color vectors> `.

PNG format recommended, JPEG actually increases file size in some cases(which I think is due to the nature of the format and the fact that we are saving with 100% quality).

### Example images
![Original Image](https://github.com/Aftaab99/ImageColorQuantization/blob/master/rain_princess.png)
![Compressed Image, with 16 colours](https://github.com/Aftaab99/ImageColorQuantization/blob/master/rain_princess_compressed.png)