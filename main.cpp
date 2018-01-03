// Instructions:
// For question 1, only modify function: histogram_equalization
// For question 2, only modify functions: low_pass_filter, high_pass_filter,
// deconvolution
// For question 3, only modify function: laplacian_pyramid_blending

#include "./header.h"

using namespace std;
using namespace cv;

void help_message(char *argv[]) {
  cout << "Usage: [Question_Number] [Input_Options] [Output_Options]" << endl;
  cout << "[Question Number]" << endl;
  cout << "1 Histogram equalization" << endl;
  cout << "2 Frequency domain filtering" << endl;
  cout << "3 Laplacian pyramid blending" << endl;
  cout << "[Input_Options]" << endl;
  cout << "Path to the input images" << endl;
  cout << "[Output_Options]" << endl;
  cout << "Output directory" << endl;
  cout << "Example usages:" << endl;
  cout << argv[0] << " 1 "
       << "[path to input image] "
       << "[output directory]" << endl;
  cout << argv[0] << " 2 "
       << "[path to input image1] "
       << "[path to input image2] "
       << "[output directory]" << endl;
  cout << argv[0] << " 3 "
       << "[path to input image1] "
       << "[path to input image2] "
       << "[output directory]" << endl;
}

void displayImageinWindow(string windowName, Mat img) {
	namedWindow( windowName, WINDOW_AUTOSIZE);
	imshow( windowName, img);
}

// ===================================================
// ======== Question 1: Histogram equalization =======
// ===================================================

Mat histogram_equalization(const Mat &img_in) {

	bool showHistVisuals = false;
	bool showEqualizeHist = false;

	// Write histogram equalization codes here
	Mat img_out;

	// Split the image into BRG channels
	// Assumption is that image has 3 channels
	float pixels = img_in.rows * img_in.cols;
	int channel_count = 3;
	Mat split_channels[channel_count];
	split(img_in, split_channels);

	// To visualising the histogram
	int histSize = 256;
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w / histSize );
	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );
	Mat nHistImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	// Calculate CDF and PDF here
	Mat hist[channel_count];
	vector<float> pdf[channel_count];

	for(int i=0; i<channel_count; i++) {
		// Calculate hist
		const int channels[] = {0};
		const int sizes[] = {256};
		float range[] = {0,256};
		const float *ranges[] = {range};
		calcHist(
			&split_channels[i],	// input mat reference
			1,			// no in images
			channels,// no of channels
			Mat(),	// mask, if any
			hist[i],	// output histogram array
			1,			// dimensions
			sizes,	// hist size, no of bins/steps we want
			ranges,	// hist range
			true,		// bin sizes uniform
			false		// hist cleared at beginning
		);

		// plot histogram and normalised histogram
		for( int j = 1; j < histSize; j++ ) {
			line(	histImage,
					Point( bin_w*(j-1), hist_h - cvRound(hist[i].at<float>(j-1)) ),
					Point( bin_w*(j), hist_h - cvRound(hist[i].at<float>(j)) ),
					Scalar( (i==0) ? 255 : 0, (i==1) ? 255 : 0, (i==2) ? 255 : 0),
					2,
					8,
					0
			);
		}
		Mat nHist;
		normalize(hist[i], nHist, 0, nHistImage.rows, NORM_MINMAX, -1, Mat() );
		for( int j = 1; j < histSize; j++ ) {
			line(	nHistImage,
					Point( bin_w*(j-1), hist_h - cvRound(nHist.at<float>(j-1)) ),
					Point( bin_w*(j), hist_h - cvRound(nHist.at<float>(j)) ),
					Scalar( (i==0) ? 255 : 0, (i==1) ? 255 : 0, (i==2) ? 255 : 0),
					2,
					8,
					0
			);
		}

		// Calculate CDF here
		vector<float> cdf;
		float sum = 0;
		for(int j=0; j < histSize; j++ ) {
			sum = sum + hist[i].at<float>(j);
			cdf.push_back( sum );
		}

		// Calculate PDF here
		for ( int j=0; j < histSize; j++ ) {
			float temp = cdf.at(j) * 255;
			float temp1 = temp / pixels ;
			pdf[i].push_back( temp1 );
		}

		// OpenCV function to equalize hist, used for comparison of image
		equalizeHist(split_channels[i], split_channels[i]);
	}

	// Display histogram
	if(showHistVisuals) {
		displayImageinWindow( "RBG Histogram", histImage);
		displayImageinWindow( "Normalised RBG Histogram", nHistImage);
	}

	// Merge and show the equalizeHist result
	if(showEqualizeHist) {
		merge(split_channels, channel_count, img_out);
		displayImageinWindow( "Filtered image using equalizeHist", img_out);
	}

	// Changing the image as per LUT from PDF
	img_out = img_in.clone();
	for(int i=0; i<img_out.rows; i++ ) {
		for(int j=0; j<img_out.cols; j++ ) {
			for(int c=0; c<channel_count; c++) {
				int int_pixel_value =  (int) img_out.at<Vec3b>(i,j)[c];
				img_out.at<Vec3b>(i,j)[c] = (uchar) pdf[c].at( int_pixel_value );
			}
		}
	}
	
	// Return the image
	return img_out;
}

bool Question1(char *argv[]) {
	bool showIOImages = false;
	// Read in input images
	Mat input_image = imread(argv[2], IMREAD_COLOR);
	if(showIOImages)
		displayImageinWindow( "Question 1 Input", input_image);

	// Histogram equalization
	Mat output_image = histogram_equalization(input_image);
	if(showIOImages) {
		displayImageinWindow( "Question 1 Output", output_image);
		waitKey(0);
	}

	// Write out the result
	string output_name = string(argv[3]) + string("output1.png");
	imwrite(output_name.c_str(), output_image);

	return true;
}

// ===================================================
// ===== Question 2: Frequency domain filtering ======
// ===================================================
/******************************************************************************/

// Moves origin to the center of the image, by quadrant manipulation
void centerDFT( Mat& img ) {
  	Mat tmp, q0, q1, q2, q3;

	// first crop the image, if it has an odd number of rows or columns
	img = img(Rect(0, 0, img.cols & -2, img.rows & -2));

	int cx = img.cols/2;
	int cy = img.rows/2;

	q0 = img(Rect(0, 0, cx, cy));
	q1 = img(Rect(cx, 0, cx, cy));
	q2 = img(Rect(0, cy, cx, cy));
	q3 = img(Rect(cx, cy, cx, cy));

	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void visualDFT(const Mat &complexI) {
	Mat planes[2];
	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	// string rawstr = "output_imgs/raw.jpg";
	// imwrite(rawstr.c_str(), planes[0]);
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];
}

void createImgDFT(Mat &image, Mat &complexI, bool cvtScale) {

	if(cvtScale) {
		cvtColor(image, image, CV_BGR2GRAY);
	}

	// Expand input image to optimal size, on the border add zero values
	Mat padded;
	int m = getOptimalDFTSize( image.rows );
	int n = getOptimalDFTSize( image.cols );
	copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols, BORDER_CONSTANT, Scalar::all(0));

	// Add to the expanded another plane with zeros
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	// Mat complexI;
	merge(planes, 2, complexI);

	// this way the result may fit in the source matrix
	dft(complexI, complexI, DFT_SCALE|DFT_COMPLEX_OUTPUT);

	// to visualise the fft
	visualDFT(complexI);
}

void create_lowpass_filter(const Mat &filter) {
	int filter_length = 20;
	Mat filter_plane = Mat::zeros(filter.rows, filter.cols, CV_32F);

	int start_i = filter.rows / 2 - filter_length/2;
	int start_j = filter.cols / 2 - filter_length/2;

	for( int i_counter = 0; i_counter < filter_length; i_counter++ ) {
		for( int j_counter = 0; j_counter < filter_length; j_counter++ ) {
			filter_plane.at<float>( i_counter+start_i, j_counter+start_j ) = (float) 1;
		}
	}

	Mat array_of_planes[] = {filter_plane, filter_plane};
	merge(array_of_planes, 2, filter);
}

void create_highpass_filter(const Mat &filter) {
	int filter_length = 20;
	Mat filter_plane = Mat(filter.rows, filter.cols, CV_32F, cvScalar(1.0));

	int start_i = filter.rows / 2 - filter_length/2;
	int start_j = filter.cols / 2 - filter_length/2;

	for( int i_counter = 0; i_counter < filter_length; i_counter++ ) {
		for( int j_counter = 0; j_counter < filter_length; j_counter++ ) {
			filter_plane.at<float>( i_counter+start_i, j_counter+start_j ) = (float) 0;
		}
	}

	Mat array_of_planes[] = {filter_plane, filter_plane};
	merge(array_of_planes, 2, filter);
}

Mat low_pass_filter( const Mat &complex ) {

	// create a clone for internal morphing
	Mat complexI = complex.clone();

	// create the filter, of same size
	Mat filter = complexI.clone();
	create_lowpass_filter(filter);

	// apply the filter
	centerDFT(complexI);
	mulSpectrums(complexI, filter, complexI, 0);
	centerDFT(complexI);

	// inverse dft to get the image here
	Mat inverse;
	dft(complexI, inverse, DFT_INVERSE|DFT_REAL_OUTPUT);

	// convert image for display again
	inverse.convertTo(inverse, CV_8U);

	return inverse;
}

Mat high_pass_filter( const Mat &complex ) {

	// create a clone for internal morphing
	Mat complexI = complex.clone();

	// create the filter
	Mat filter = complexI.clone();
	create_highpass_filter(filter);

	// apply the filter
	centerDFT(complexI);
	mulSpectrums(complexI, filter, complexI, 0);
	centerDFT(complexI);

	// inverse dft to get the image here
	Mat inverse;
	dft(complexI, inverse, DFT_INVERSE|DFT_REAL_OUTPUT);

	// convert image to 8unsigned int for display again
	inverse.convertTo(inverse, CV_8U);

	return inverse;
}

Mat deconvolution(const Mat &img_in) {

	Mat img = img_in.clone();
	// Create DFT of image first
	Mat complexI;
	createImgDFT(img, complexI, false);
	centerDFT(complexI);

	int channel_count = 2;
	Mat split_channels[channel_count];
	split(complexI, split_channels);

	// displayImageinWindow("fft0", split_channels[0]);
	// displayImageinWindow("fft1", split_channels[1]);

	Mat gKernel = getGaussianKernel(20, 5, img_in.type() );
	Mat gKernelTranspose;
	transpose(gKernel, gKernelTranspose);
	gKernel = gKernel * gKernelTranspose;

	Mat complexK;
	createImgDFT(gKernel, complexK, false);
	centerDFT(complexK);

	resize(complexK, complexK, img.size() );
	split(complexK, split_channels);
	// displayImageinWindow("k0", split_channels[0]*255);
	// displayImageinWindow("k1", split_channels[1]*255);

	divide(complexI, complexK, complexI);
	centerDFT(complexI);

	split(complexI, split_channels);
	// displayImageinWindow("ft divide 0", split_channels[0]*255);
	// displayImageinWindow("ft divide 1", split_channels[1]*255);

	waitKey();
	// inverse dft to get the image here
	Mat inverse;
	dft(complexI, inverse, DFT_INVERSE|DFT_REAL_OUTPUT);

	// displayImageinWindow("idft", inverse);
	// displayImageinWindow("idft 255", inverse*255);

	return inverse;
}

bool Question2(char *argv[]) {
	// Read in input images
	Mat input_image1 = imread(argv[2], IMREAD_COLOR);
	Mat input_image2 = imread(argv[3], IMREAD_ANYCOLOR|IMREAD_ANYDEPTH);

	// Create DFT of image first, since it will be used twice
	Mat complexI;
	createImgDFT(input_image1, complexI, true);

	// Low and high pass filters
	Mat output_image1 = low_pass_filter(complexI);
	Mat output_image2 = high_pass_filter(complexI);

	// Deconvolution
	Mat output_image3 = deconvolution(input_image2);

	// Write out the result
	string output_name1 = string(argv[4]) + string("output2LPF.png");
	string output_name2 = string(argv[4]) + string("output2HPF.png");
	string output_name3 = string(argv[4]) + string("output2deconv.png");
	imwrite(output_name1.c_str(), output_image1);
	imwrite(output_name2.c_str(), output_image2);
	// displayImageinWindow(output_name3.c_str(), output_image3);
	// imwrite(output_name3.c_str(), output_image3);

	waitKey();
	return true;
}

// ===================================================
// ===== Question 3: Laplacian pyramid blending ======
// ===================================================
void lapPyrCreate(const Mat& img, vector<Mat_<Vec3f> >& lapPyr, Mat& smallestLevel, int levels) {
	Mat input_img = img;
	for (int i=0; i<levels; i++) {
		Mat down,up;
		pyrDown(input_img, down);
		pyrUp(down, up, input_img.size());
		Mat lap = input_img - up;
		lapPyr.push_back(lap);
		input_img = down;
	}
	input_img.copyTo(smallestLevel);
}

Mat lapPyrBlend(const Mat& img_in1, const Mat& img_in2, int levels) {
	// Write laplacian pyramid blending codes here
	Mat_<Vec3f> left;
	img_in1.convertTo(left, CV_32F, 1.0/255.0);
	Mat_<Vec3f> right;
	img_in2.convertTo(right, CV_32F, 1.0/255.0);

	Mat_<float> initialImg(left.rows, left.cols, 0.0);
	vector< Mat_<Vec3f> > blurMask;
	Mat currentImg;
	initialImg(Range::all(),Range(0, initialImg.cols/2)) = 1.0;
	cvtColor(initialImg, currentImg, CV_GRAY2BGR);
	blurMask.push_back(currentImg);

	// Generate Laplacian Pyramids
	Mat lsl, rsl;
	vector< Mat_<Vec3f> > pyrLeft, pyrRight, pyrResult;
	lapPyrCreate(left, pyrLeft, lsl, levels);
	lapPyrCreate(right, pyrRight, rsl, levels);

	// Create a blur mask for each layer in pyramid
	currentImg = initialImg;
	for( int i=1; i<levels+1; i++ ) {
    	Mat temp; 
    	if( pyrLeft.size() > i )
    		pyrDown( currentImg, temp, pyrLeft[i].size() );
    	else
    		pyrDown( currentImg, temp, lsl.size() );
    	Mat down; 
    	cvtColor(temp, down, CV_GRAY2BGR);
    	blurMask.push_back(down);
    	currentImg = temp;
	}

	// Now add left and right halves of images in each level
	Mat img_out = lsl.mul(blurMask.back()) + rsl.mul(Scalar(1.0,1.0,1.0) - blurMask.back());
	for (int i=0; i<levels; i++) {
		Mat A = pyrLeft[i].mul(blurMask[i]);
		Mat mask = Scalar(1.0,1.0,1.0) - blurMask[i];
		Mat B = pyrRight[i].mul(mask);
		Mat_<Vec3f> blendedLevel = A + B;
		pyrResult.push_back(blendedLevel);
	}

	// Reconstruct
	for (int i=levels-1; i>=0; i--) {
		Mat up;
		pyrUp(img_out, up, pyrResult[i].size());
		img_out = up + pyrResult[i];
	}

	return img_out;
}

bool Question3(char* argv[]) {
	// Read in input images
	Mat input_image1 = imread(argv[2], IMREAD_COLOR);
	Mat input_image2 = imread(argv[3], IMREAD_COLOR);

	// Resize the images
	Size s(520, 580);
	resize(input_image2, input_image2, s, 0, 0);
	input_image2(Range(30, input_image2.rows-10), Range(25, input_image2.cols-15)).copyTo(input_image2);
	Size sq(500, 500);
	resize(input_image1, input_image1, sq, 0, 0);
	resize(input_image2, input_image2, sq, 0, 0);

	// Running the Laplacian Pyramid Blending
	Mat output_image = lapPyrBlend(input_image1, input_image2, 6);
	output_image *= 255;
	string output_name = string(argv[4]) + string("output3.png");
	imwrite(output_name.c_str(), output_image);

	return true;
}

int main(int argc, char *argv[]) {

	int question_number = -1;

	// Validate the input arguments
	if (argc < 4) {
		help_message(argv);
		exit(1);
	}
	else {
		question_number = atoi(argv[1]);

		if (question_number == 1 && !(argc == 4)) {
			help_message(argv);
			exit(1);
		}
		if (question_number == 2 && !(argc == 5)) {
			help_message(argv);
			exit(1);
		}
		if (question_number == 3 && !(argc == 5)) {
			help_message(argv);
			exit(1);
		}
		if (question_number > 3 || question_number < 1 || argc > 5) {
			cout << "Input parameters out of bound ..." << endl;
			exit(1);
		}
	}

	switch (question_number) {
		case 1:
			Question1(argv);
			break;
		case 2:
			Question2(argv);
			break;
		case 3:
			Question3(argv);
			break;
	}

	return 0;
}
