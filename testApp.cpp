#include "testApp.h"

//--------------------------------------------------------------
void testApp::setup(){
	//ofSetFrameRate(24);
	gpuDeviceCount = cv::gpu::getCudaEnabledDeviceCount();
	
	ofBackground(0,0,0);
	ofSetWindowTitle("Infinity-by-Nine");
	ofEnableAlphaBlending();
	ofSetVerticalSync(true);
	blurDistance = 3;
	gaussBlurDistance = 5;
	vel = 1;
	sc = .25;
	#ifdef _USE_LIVE_VIDEO
        vidGrabber.setVerbose(true);
        vidGrabber.initGrabber(320,240);
	#else
	//centerMovie.loadMovie("movies/TheThirdMan.mov");
	//centerMovie.loadMovie("movies/Rango.mov");
	//centerMovie.loadMovie("movies/El_Topo.mov");
	//centerMovie.loadMovie("movies/MeettheFeebles.mov");
	//centerMovie.setFrame(35000);
	//centerMovie.loadMovie("movies/DonaldInMathmagicLand.mov");
	//centerMovie.loadMovie("movies/StarWarsEpisodeIV.mov");
	//centerMovie.setFrame(12300);
	centerMovie.loadMovie("movies/BladeRunner.mov");
	centerMovie.setFrame(80000);
	//centerMovie.loadMovie("movies/BullitCarChase.mov");
	//centerMovie.setFrame(7080);
	//centerMovie.loadMovie("movies/George_01_02.mov");
	//centerMovie.setFrame(8750);
	//centerMovie.loadMovie("movies/DoctorWho.mov");
	//centerMovie.loadMovie("movies/DoctorWho320.mov");
	//centerMovie.loadMovie("AmericanGraffiti.mov");
	//centerMovie.loadMovie("movies/Nemo.mov");
	//centerMovie.setFrame(1600);//1600
	#endif
	w = centerMovie.width;
	h = centerMovie.height;
	
	blackImage.loadImage("movies/black_rgb0001.png");
	maskImage.loadImage("movies/mask0001.png");
	maskLeftImage.loadImage("movies/maskLeft0001.png");
	maskTopImage.loadImage("movies/maskTop0001.png");
	
	colorImg.allocate(w,h);
	colorTopImg.allocate(w,h);
	colorRightImg.allocate(w,h);
	colorRightTempImg.allocate(w,h);
	colorRightPastImg.allocate(w,h);
	colorLeftImg.allocate(w,h);
	grayImage.allocate(w,h);
	grayBg.allocate(w,h);
	grayDiff.allocate(w,h);
	topImage.allocate(w,h,GL_RGB);
	leftImage.allocate(w,h,GL_RGB);
	rightImage.allocate(w,h,GL_RGB);
	map_x.allocate(w,h);
	map_y.allocate(w,h);

	topPixels 			= new unsigned char [w*h*3];
	moviePixels 		= new unsigned char [w*h*3];
	rightPixels 		= new unsigned char [w*h*3];
	rightFinalPixels	= new unsigned char [w*h*3];
	leftFinalPixels		= new unsigned char [w*h*3];
	leftPixels 			= new unsigned char [w*h*3];
	grayDiffPixels		= new unsigned char [w*h*3];
	
	maskRightShader.load("shaders/composite.vert", "shaders/composite.frag");
	maskRightShader.begin();
	maskRightShader.setUniformTexture("Tex0", blackImage.getTextureReference(), 0);
	maskRightShader.setUniformTexture("Tex1", maskImage.getTextureReference(), 1);
	maskRightShader.end();

	maskLeftShader.load("shaders/composite.vert", "shaders/composite.frag");
	maskLeftShader.begin();
	maskLeftShader.setUniformTexture("Tex0", blackImage.getTextureReference(), 0);
	maskLeftShader.setUniformTexture("Tex1", maskLeftImage.getTextureReference(), 1);
	maskLeftShader.end();

	maskTopShader.load("shaders/composite.vert", "shaders/composite.frag");
	maskTopShader.begin();
	maskTopShader.setUniformTexture("Tex0", blackImage.getTextureReference(), 0);
	maskTopShader.setUniformTexture("Tex1", maskTopImage.getTextureReference(), 1);
	maskTopShader.end();

	blurRightHorizShader.load("shaders/simpleBlurHorizontal.vert","shaders/simpleBlurHorizontal.frag");
	blurRightHorizShader.begin();
	blurRightHorizShader.setUniformTexture("Tex0", rightImage,0);
	blurRightHorizShader.end();

	blurRightVertShader.load("shaders/simpleBlurVertical.vert","shaders/simpleBlurVertical.frag");
	blurRightVertShader.begin();
	blurRightVertShader.setUniformTexture("Tex0", rightImage,0);
	blurRightVertShader.end();

	
	blurLeftHorizShader.load("shaders/simpleBlurHorizontal.vert","shaders/simpleBlurHorizontal.frag");
	blurLeftHorizShader.begin();
	blurLeftHorizShader.setUniformTexture("Tex0", leftImage,0);
	blurLeftHorizShader.end();

	blurLeftVertShader.load("shaders/simpleBlurVertical.vert","shaders/simpleBlurVertical.frag");
	blurLeftVertShader.begin();
	blurLeftVertShader.setUniformTexture("Tex0", leftImage,0);
	blurLeftVertShader.end();

	blurTopShader.load("shaders/simpleBlurHorizontal.vert","shaders/simpleBlurHorizontal.frag");
	blurTopShader.begin();
	blurTopShader.setUniformTexture("Tex0", topImage,0);
	blurTopShader.end();

	frameByframe = false;
	mode3 = true;
	clearTex = false;
	bLearnBakground = true;
	threshold = 55;
	diffPercent = 0.135;//.135
	sceneChange = true;
	sceneChangeEnabled = true;
	debugDraw = false;
	avg = 1;
	pastIntRight = 0;
	pastIntLeft = 0;
	leftMask = .15;
	rightMask = .85;

	
	
	centerMovie.play();
	
	
	
}

//--------------------------------------------------------------
void testApp::update(){
	
	const int MAX_COUNT = 500;
	bool bNewFrame = false;

	#ifdef _USE_LIVE_VIDEO
       vidGrabber.grabFrame();
	   bNewFrame = vidGrabber.isFrameNew();
    #else
        centerMovie.idleMovie();
        bNewFrame = centerMovie.isFrameNew();
	#endif

	if (bNewFrame){

		#ifdef _USE_LIVE_VIDEO
            colorImg.setFromPixels(vidGrabber.getPixels(), 320,240);
	    #else
            colorImg.setFromPixels(centerMovie.getPixels(), w,h);
        #endif
		
        grayImage = colorImg;
		if (bLearnBakground == true){
			grayBg = grayImage;		
			bLearnBakground = false;
			grayBg.blur(1);
			grayBg.flagImageChanged();
		}
		grayImage.blur(1);//default = 1

		//Histograms for scene change detection.
		iplImageGray = grayImage.getCvImage();
		iplImagePastGray = grayBg.getCvImage();
	
		plane = &iplImageGray;
		planePast = &iplImagePastGray;
	
		int hist_size[] = { 30 };
		float range[] = { 0, 180 };
		float* ranges[] = { range };

		curr_hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY, ranges, 1 );
		past_hist = cvCreateHist( 1, hist_size, CV_HIST_ARRAY, ranges, 1 );

		cvCalcHist( plane, curr_hist, 0, 0 );
		cvCalcHist( planePast, past_hist, 0, 0 );
	
		cvNormalizeHist( curr_hist, 20*255 ); //Normalize
		cvNormalizeHist( past_hist, 20*255 ); 

		if (cvCompareHist(curr_hist, past_hist,3) > diffPercent){
			sceneChange = true;
			bLearnBakground = true;
		}
	
		cvClearHist(curr_hist);
		cvReleaseHist( &curr_hist );
		cvClearHist(past_hist);
		cvReleaseHist( &past_hist );
		//END Histograms
	
		
		
		//grayDiff.absDiff(grayBg, grayImage);
		//grayDiff.threshold(threshold);
		//cv::Size kern(12,12);
		//cv::gpu::GpuMat M ;
		//M = grayDiff.getCvImage();
		//cv::gpu::GaussianBlur(M,M,kern,0,0,4,-1,cv::gpu::Stream::Null());
		//cv::gpu::threshold(M,M,threshold,100,1);
		
	}
	//FAST Corner Detection and the LK OpticalFlow
	cv::Size winSize(12,12);
	vector<cv::KeyPoint> keyPoints;
	vector<cv::KeyPoint> nextPoints;
	cv::FAST(grayBg.getCvImage(), keyPoints,25,true); //default = 25
	//cv::FAST(grayImage.getCvImage(), nextPoints,25,true);
		// convert vector of keypoints to vector of points
	cv::KeyPoint::convert(keyPoints, points_keyPoints);
	cv::KeyPoint::convert(nextPoints, points_nextPoints);
	cv::TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,1,0.1);//Default = 20,.03
	vector<uchar> status;
    vector<float> err;
	
	//ofRectangle roi(0, 0, 500, 50);
	
	//grayBg.setROI(roi);
	//grayImage.setROI(roi);
	cv::calcOpticalFlowPyrLK(grayBg.getCvImage(),grayImage.getCvImage(),points_keyPoints,points_nextPoints, status, err, winSize, 2, termcrit, 0.4, 1);
	//grayImage.resetROI();
	//grayBg.resetROI();
	//cv::Mat nextPointsM(points_nextPoints);
	//cv::Mat keyPointsM(points_keyPoints);
	//cv::Mat sumPointsM;
	
	//cv::Mat R = cv::estimateRigidTransform(keyPointsM,nextPointsM,true);
	//cv::Mat H = cv::findHomography(keyPointsM,nextPointsM,CV_RANSAC,3.0);
	
    
	grayBg = grayImage;

	//calculate velocities of screen edges.
	int minThreshFlow = 0;
	int maxThreshFlow = 24;
	float sum = 0.0;
	float sumLeft = 0.0;
	float sumRight = 0.0;
	float sumDown = 0.0;
	float sumUp = 0.0;			
	float avgFloatLeft = 0.0;
	float avgFloatRight = 0.0;
	float avgFloatUp = 0.0;
	float avgFloatDown = 0.0;
	float avgFloat = 0.0;
	int countLeft = 0;
	int countRight = 0;
	int countDown = 0;
	int countUp = 0;
	
	for( int i=0; i < points_keyPoints.size(); i++ ) {
		if((abs((points_keyPoints[i].x - points_nextPoints[i].x)) >= minThreshFlow || 
			abs((points_keyPoints[i].y - points_nextPoints[i].y)) >= minThreshFlow) &&
			(abs((points_keyPoints[i].x - points_nextPoints[i].x)) <= maxThreshFlow && 
			abs((points_keyPoints[i].y - points_nextPoints[i].y)) <= maxThreshFlow) //&&
			//(points_nextPoints[i].x < w *.25 || points_nextPoints[i].x > w*.75) &&
			//(points_nextPoints[i].y < h * .90) 
			){ 
			int tmp = (points_keyPoints[i].x - points_nextPoints[i].x);
			int tmpVert = (points_keyPoints[i].y - points_nextPoints[i].y);//neg goes down
			if(tmp < 0 && (points_nextPoints[i].x > w * rightMask && points_keyPoints[i].x > w * rightMask)){
				sumRight += tmp;
				countRight++;
			}
			if(tmp > 0 && (points_nextPoints[i].x < w * leftMask && points_keyPoints[i].x < w * leftMask)){
				sumLeft += tmp;
				countLeft++;
			}
			if(tmpVert < 0 && (points_nextPoints[i].x > w * rightMask || points_nextPoints[i].x < w * leftMask)){
				sumDown += tmpVert;
				countDown++;
			}
			if(tmpVert > 0 && (points_nextPoints[i].x < w * leftMask || points_nextPoints[i].x > w * rightMask)){
				sumUp += tmpVert;
				countUp++;
			}
			sum += (points_keyPoints[i].x - points_nextPoints[i].x);
		}
	}
	//sort(points_keyPoints.begin(), points_keyPoints.end());
	
	float total = points_keyPoints.size();
	avgFloatRight = abs(sumRight/countRight);
	avgFloatLeft = abs(sumLeft/countLeft);
	avgFloatUp = abs(sumUp/countUp);
	avgFloatDown = abs(sumDown/countDown);
	avgFloat = abs(sum / total);
	//convert float to int safely.
	int avgIntLeft = (avgFloatLeft >= 0) ? (int)(avgFloatLeft + 0.1) : (int)(avgFloatLeft - 0.1);
	int avgIntRight = (avgFloatRight >= 0) ? (int)(avgFloatRight + 0.1) : (int)(avgFloatRight - 0.1);
	int avgIntUp = (avgFloatUp >= 0) ? (int)(avgFloatUp + 0.1) : (int)(avgFloatUp - 0.1);
	int avgIntDown = (avgFloatDown >= 0) ? (int)(avgFloatDown + 0.1) : (int)(avgFloatDown - 0.1);
	avg = (avgFloat > 0) ? (int)(avgFloat + 0.1) : (int)(avgFloat - 0.1);
	if (avgIntLeft < 0){
		avgIntLeft = 0;
	}
	if (avgIntRight < 0){
		avgIntRight = 0;
	}
	if (avgIntUp < 0){
		avgIntUp = 0;
	}
	if (avgIntDown < 0){
		avgIntDown = 0;
	}
	if (avg < 0){
		avg = 0;
	}
	if (avg > 100){
		avg = 100;
	}
	
	if (mode3){
		vel = avg;
		interpAvgLeft = pastIntLeft += (avgIntLeft - pastIntLeft  )* 0.5;//.5 default
		interpAvgRight = pastIntRight += (avgIntRight - pastIntRight) * 0.5;
		interpAvgUp = pastIntUp += (avgIntUp - pastIntUp) * 0.65;
		interpAvgDown = pastIntDown += (avgIntDown - pastIntDown) * 0.65;
		velLeft = interpAvgLeft;//interpolated x1 += (x2-x1)*A
		velRight = interpAvgRight;//
		velUp = interpAvgUp;
		velDown = interpAvgDown;
		numRows = vel+1;
		numRowsLeft = velLeft+1;
		numRowsRight = velRight+1;
		numColUp = velUp;
		numColDown = velDown;
	}else{
		numRows = 1;
		numRowsLeft = 1;
		numRowsRight = 1;
		numColUp = 1;
		numColDown = 1;
	}
	pastIntRight = interpAvgRight;
	pastIntLeft	= interpAvgLeft;
	pastIntUp	= interpAvgUp;
	pastIntDown	= interpAvgDown;

	moviePixels = centerMovie.getPixels();
	
	

    //top----------------------------------------------------------
	int totalPixels = w*h*3;
	for (int i = 0; i < numRows * w * 3; i++){//get top numRows of pixels and place in bottom 10 rows of topImage.
			topPixels[i+totalPixels-(numRows*w*3)] = moviePixels[i];
		}
	if (!mode3){
	for (int i = 0; i < w ; i++){
		for (int j = 0; j < h-numRows; j++){//Fillin the rest from original row 0.
			topPixels[(j*w+i)*3 + 0 ] = moviePixels[(i)*3 + 0];//red
			topPixels[(j*w+i)*3 + 1 ] = moviePixels[(i)*3 + 1];//green
			topPixels[(j*w+i)*3 + 2 ] = moviePixels[(i)*3 + 2];//blue
		}
	}
	}else{
		for (int i = 0; i < w ; i++){
			for (int j = 0; j < h-numRows; j++){//CopyandMoveUp
			topPixels[(j*w+i)*3 + 0 ] = moviePixels[(i)*3 + 0];//red
			topPixels[(j*w+i)*3 + 1 ] = moviePixels[(i)*3 + 1];//green
			topPixels[(j*w+i)*3 + 2 ] = moviePixels[(i)*3 + 2];//blue
			}
		}
	}
	colorTopImg.setFromPixels(topPixels, w,h);
	colorTopImg.resize(16,16);
	colorTopImg.blurGaussian(25);
	colorTopImg.resize(w,h);
	topImage.loadData(colorTopImg.getPixels(), w,h, GL_RGB);
	
	//left----------------------------------------------------------
	for (int i = 0; i < numRowsLeft; i++){
		for (int j = 0; j < h; j++){//get first column of pixels.
			int destIndex = (j*w+i)*3 + (w*3-3*numRowsLeft);
			 if(destIndex > 0 && destIndex < w*h*3) {  
                int sourceIndex = (j*w+i)*3;
				leftPixels[destIndex + 0] = moviePixels[sourceIndex + 0];//red (w*3
				leftPixels[destIndex + 1] = moviePixels[sourceIndex + 1];//green (w*3
				leftPixels[destIndex + 2] = moviePixels[sourceIndex + 2];//blue (w*3
			 }
		}
	}
	if (!mode3){
	for (int i = 0; i  < w - numRowsLeft; i++){
		for (int j = 0; j < h; j++){//Fill in the rest with original column 0.
			int destIndex = (j*w+i)*3;
			 if(destIndex > 0 && destIndex < w*h*3) {  
                int sourceIndex = (j*w+0)*3;
				leftPixels[destIndex + 0] = moviePixels[sourceIndex + 0];//red
				leftPixels[destIndex + 1] = moviePixels[sourceIndex + 1];//green
				leftPixels[destIndex + 2] = moviePixels[sourceIndex + 2] ;//blue
			 }
		}
	}

	}else{
			if (velLeft >= 0){
			for (int i = 0; i  < w - numRowsLeft; i++){
				for (int j = 0; j < h; j++){//CopyandMove
					int destIndex = (j*w+i)*3;
					 if(destIndex > 0 && destIndex < w*h*3) {
						int sourceIndex = (j*w+(i+velLeft))*3;
						leftPixels[destIndex + 0] = leftPixels[sourceIndex + 0];//red
						leftPixels[destIndex + 1] = leftPixels[sourceIndex + 1];//green
						leftPixels[destIndex + 2] = leftPixels[sourceIndex + 2];//blue
						}
					}
				}
			}if(velLeft == 0 && velRight >= 0) {
			for (int i = w; i  > numRowsRight; i--){
				for (int j = 0; j < h; j++){//CopyandMove
					int destIndex = (j*w+i)*3;
					 if(destIndex > 0 && destIndex < w*h*3) {
						int sourceIndex = (j*w+(i-velRight))*3;
						leftPixels[destIndex + 0] = leftPixels[sourceIndex + 0];//red
						leftPixels[destIndex + 1] = leftPixels[sourceIndex + 1];//green
						leftPixels[destIndex + 2] = leftPixels[sourceIndex + 2];//blue
						}
					}
				}
			}
	
	}

	if (sceneChangeEnabled){
		//int tmpGaussBlurDistance = gaussBlurDistance;
		if (sceneChange){
			
			for (int i = 0; i  < w - 1; i++){
				for (int j = 0; j < h; j++){//Fill in the rest with original column 0.
					int destIndex = (j*w+i)*3;
					if(destIndex > 0 && destIndex < w*h*3) {
						int sourceIndex = (j*w+0)*3;
						leftPixels[destIndex + 0] = moviePixels[sourceIndex + 0];//red
						leftPixels[destIndex + 1] = moviePixels[sourceIndex + 1];//green
						leftPixels[destIndex + 2] = moviePixels[sourceIndex + 2];//blue
						}
					}
				}
			gaussBlurDistance = 50;
			
		
		}else{
			gaussBlurDistance = 25;
		}
	}
	colorLeftImg.setFromPixels(leftPixels, w,h);
	colorLeftImg.resize(128,128);
	colorLeftImg.blurGaussian(25);
	colorLeftImg.resize(w,h);
	//if (!mode3){
		//colorLeftImg.blurMedian(gaussBlurDistance);
	//}else{
		//colorLeftImg.blurGaussian(gaussBlurDistance);
	//}

	leftImage.loadData(colorLeftImg.getPixels(), w,h, GL_RGB);
	
	
	//right-----------------------------------------------------------------
	/*for (int i = w-numRowsRight; i < w; i++){
		for (int k = 0; k < h; k++){//get last column of pixels.
			int destIndex = (k*w+i)*3 - (w*3-3*numRowsRight);
			 if(destIndex > 0 && destIndex < w*h*3) {
				int sourceIndex = (k*w+i)*3;
				rightPixels[destIndex + 0] = moviePixels[sourceIndex + 0];
				rightPixels[destIndex + 1] = moviePixels[sourceIndex + 1];
				rightPixels[destIndex + 2] = moviePixels[sourceIndex + 2];
			 }
		}
	}
	if(!mode3){
	for (int i = numRows; i < w; i++){
		for (int k = 0; k < h; k++){//Fill in the rest with original last column.
			int destIndex = (k*w+i)*3;
			if(destIndex > 0 && destIndex < w*h*3) {
				int sourceIndex = (k*w+0)*3;
				rightPixels[destIndex + 0] = rightPixels[sourceIndex + 0];
				rightPixels[destIndex + 1] = rightPixels[sourceIndex + 1];
				rightPixels[destIndex + 2] = rightPixels[sourceIndex + 2];
			}
		}
	}
	}else{
		if (velRight >= 0){
		for (int i = w; i > numRowsRight ; i--){
			for (int k = 0; k < h; k++){//move them along
				int destIndex = (k*w+i)*3;
				if(destIndex > 0 && destIndex < w*h*3) {
					int sourceIndex = (k*w+(i-velRight))*3;
					rightPixels[destIndex + 0] = rightPixels[sourceIndex + 0];
					rightPixels[destIndex + 1] = rightPixels[sourceIndex + 1];
					rightPixels[destIndex + 2] = rightPixels[sourceIndex + 2];
					}
				}
			}
		}
		if (velRight == 0 && velLeft >= 0){
			for (int i = 0; i < w-numRowsLeft ; i++){
				for (int k = 0; k < h; k++){//move them along
				int destIndex = (k*w+i)*3;
				if(destIndex > 0 && destIndex < w*h*3) {
					int sourceIndex = (k*w+(i+velLeft))*3;
					rightPixels[destIndex + 0] = rightPixels[sourceIndex + 0];
					rightPixels[destIndex + 1] = rightPixels[sourceIndex + 1];
					rightPixels[destIndex + 2] = rightPixels[sourceIndex + 2];
					}
				}
			}

		}
	}*/
	
	if (sceneChangeEnabled){
		if (sceneChange){
			
			for (int i = 0; i < w; i++){
				for (int k = 0; k < h; k++){//Set all pixels to smear for sceneChange.
					int destIndex = (k*w+i)*3;
					if(destIndex > 0 && destIndex < w*h*3) {
						int sourceIndex = (k*w)*3 + (w*3-3*numRowsRight);
						rightPixels[destIndex + 0] = moviePixels[sourceIndex + 0];
						rightPixels[destIndex + 1] = moviePixels[sourceIndex + 1];
						rightPixels[destIndex + 2] = moviePixels[sourceIndex + 2];
						}
					}
				}
			//gaussBlurDistance = 50;
			colorRightImg.setFromPixels(rightPixels, w,h);
			//colorRightImg.blurGaussian(gaussBlurDistance);
			//colorRightImg.resize(256,256);
			colorRightImg.blurGaussian(25);
			//colorRightImg.resize(w,h);
			sceneChange = false;
			
		
		}else{
			rightPixels = colorRightPastImg.getPixels();
			for (int i = w-numRowsRight; i < w; i++){
				for (int k = 0; k < h; k++){//get last column of pixels.
					int destIndex = (k*w+i)*3 - (w*3-3*numRowsRight);
					if(destIndex > 0 && destIndex < w*h*3) {
					int sourceIndex = (k*w+i)*3;
						rightPixels[destIndex + 0] = moviePixels[sourceIndex + 0];
						rightPixels[destIndex + 1] = moviePixels[sourceIndex + 1];
						rightPixels[destIndex + 2] = moviePixels[sourceIndex + 2];
						}
					}
				}
			
			colorRightImg.setFromPixels(rightPixels, w,h);
		}
	}
	
	
	//colorRightImg.setFromPixels(rightPixels, w,h);
	//cv::Mat   src, dst, map_x, map_y;
	//src = colorRightImg.getCvImage();
	//dst = colorRightTempImg.getCvImage();
	//cv::warpAffine(src, dst, R, dst.size(),0,1);
	//cv::warpPerspective(src, dst, H, dst.size(),0,1);
	
	
	if (velRight >= 0){
		horizontalDistanceToMove = float(numRowsRight);
	}
	if (velRight == 0 && velLeft >= 0){
		horizontalDistanceToMove = float(numRowsLeft);
	}
	if (velDown == 0 && velUp >= 0){
		verticalDistanceToMove = float(numColUp);
	}
	if (velUp == 0 && velDown >= 0){
		verticalDistanceToMove = float(numColDown);
	}
  
	for(int j=0; j < h; j++){  
       for(int i=0; i < w; i++){  
           int positionInArray = j*w + i;  
           map_x.getPixelsAsFloats()[positionInArray] = fmodf((float)i + horizontalDistanceToMove,(float)w); 
           map_y.getPixelsAsFloats()[positionInArray] = fmodf((float)j + verticalDistanceToMove, (float)h); 
       }  
   }  
	colorRightImg.remap( map_x.getCvImage(), map_y.getCvImage());
	colorRightImg.flagImageChanged();  
      
	//cv::remap( src, dst, map_x, map_y, 0, 1, 0 );
	//colorRightImg = colorRightTempImg;

	
	
	//if (!mode3){
		//colorRightImg.blurMedian(gaussBlurDistance);
	//}else{
		//colorRightImg.blurGaussian(gaussBlurDistance);
	//}
	//colorRightImg.resize(512,512);
	//colorRightImg.blurGaussian(25);
	//colorRightImg.resize(w,h);

	rightImage.loadData(colorRightImg.getPixels(), w,h, GL_RGB);
	colorRightPastImg = colorRightImg;
	
	/*if (centerMovie.getCurrentFrame() > 8860){
		centerMovie.setFrame(8730);
		//sceneChange = true;
	}*/
	
	
}

//--------------------------------------------------------------
void testApp::draw(){
	ofSetHexColor(0xFFFFFF);
	float centeredX = ofGetWidth()*.5-w*.5;
	float centeredY = ofGetHeight()*.5-h*.5;
	
    //movieFullTexture.draw(0,0,1280,720);
	centerMovie.draw(0,0,1280,720);
	int nextPointCount = points_nextPoints.size();
	int keyPointCount = points_keyPoints.size();
	if (debugDraw){
		grayImage.draw(0,0);
	
	ofSetHexColor(0xffffff);
	
	for( int i=0; i < points_nextPoints.size(); i++ ) {
		if((points_nextPoints[i].x < w * leftMask|| points_nextPoints[i].x > w * rightMask) &&
			(points_nextPoints[i].y < h * .99)){
		ofNoFill();
		ofSetHexColor(0xFF0000);
		ofCircle(points_nextPoints[i].x,points_nextPoints[i].y,2);
		}
	}
	for( int i=0; i < points_keyPoints.size(); i++ ) {
		if((points_keyPoints[i].x < w *  leftMask || points_keyPoints[i].x > w * rightMask) &&
			(points_keyPoints[i].y < h * .99)){
		ofNoFill();
		ofSetHexColor(0x00FF00);
		ofCircle(points_keyPoints[i].x,points_keyPoints[i].y,2);
		}
	}
	int minThreshFlow = 0;
	int maxThreshFlow = 24;
	for( int i=0; i < points_keyPoints.size(); i++ ) {
		if(
			(abs((points_keyPoints[i].x - points_nextPoints[i].x)) >= minThreshFlow || 
			abs((points_keyPoints[i].y - points_nextPoints[i].y)) >= minThreshFlow) &&
			(abs((points_keyPoints[i].x - points_nextPoints[i].x)) <= maxThreshFlow && 
			abs((points_keyPoints[i].y - points_nextPoints[i].y)) <= maxThreshFlow) &&
			(points_nextPoints[i].x < w * leftMask || points_nextPoints[i].x > w* rightMask) &&
			(points_nextPoints[i].y < h * .99)
			){ 
				ofSetHexColor(0xFFFF00);
				ofLine(points_nextPoints[i].x,points_nextPoints[i].y,points_keyPoints[i].x,points_keyPoints[i].y);
		}
	}
	}
	//colorImg.draw(0,0);
	//glPushMatrix();
		//glScalef(2,2,1);
		//glTranslatef(0,0,0);
		//flow.draw();
	//glPopMatrix();

	blurTopShader.begin();
	blurTopShader.setUniform1f("blurAmnt", blurDistance);
	topImage.bind();
	topImage.draw(3841,0,1280,720);
	topImage.unbind();
	blurTopShader.end();//*/
	
	ofSetHexColor(0xFFFFFF);
	if (!mode3){
	blurRightVertShader.begin();
	blurRightVertShader.setUniform1f("blurAmnt", blurDistance);
	rightImage.bind();
	rightImage.draw(1279,0,1280,720);
	rightImage.unbind();
	blurRightVertShader.end();
	} else {
		blurRightHorizShader.begin();
		blurRightHorizShader.setUniform1f("blurAmnt", blurDistance);
		rightImage.bind();
		rightImage.draw(1279,0,1280,720);
		rightImage.unbind();
		blurRightHorizShader.end();
		}
	
	if (!mode3){
	blurLeftVertShader.begin();
	blurLeftVertShader.setUniform1f("blurAmnt", blurDistance);
	leftImage.bind();
	leftImage.draw(2559,0,1280,720);
	leftImage.unbind();
	blurLeftVertShader.end();
	} else {
		blurLeftHorizShader.begin();
		blurLeftHorizShader.setUniform1f("blurAmnt", blurDistance);
		leftImage.bind();
		//if (velLeft < 1 && velRight > 0){
			//ofTranslate(velRight*w*.5,0,0);
			//leftImage.draw(3840,0,-1280,720);//flipped
		//}else{
			leftImage.draw(2559,0,1280,720);//normal
		//}
		leftImage.unbind();
		blurLeftHorizShader.end();
		}
	
	//from forum sample:..."then draw a quad for the top layer using our composite shader to set the alpha"
	maskRightShader.begin();
	//our shader uses two textures, the top layer and the alpha
	//we can load two textures into a shader using the multi texture coordinate extensions
	glActiveTexture(GL_TEXTURE0_ARB);
	blackImage.getTextureReference().bind();

	glActiveTexture(GL_TEXTURE1_ARB);
	maskImage.getTextureReference().bind();

	//draw a quad over the rightImage.
	glBegin(GL_QUADS);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, 0,0);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, 0,0);		
	glVertex2f( 1279,0 );
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, blackImage.width, 0);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, maskImage.width, 0);		
	glVertex2f( 2559,0 );
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB,  blackImage.width,blackImage.height);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, maskImage.width,maskImage.height);
	glVertex2f( 2559,720);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, 0,blackImage.height);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, 0,maskImage.height);		
	glVertex2f(  1279,720 );

	glEnd();
	
	//deactive and clean up
	glActiveTexture(GL_TEXTURE1_ARB);
	blackImage.getTextureReference().unbind();
	
	glActiveTexture(GL_TEXTURE0_ARB);
	maskImage.getTextureReference().unbind();
	maskRightShader.end();

	//-------------------------------------------------------

	maskLeftShader.begin();
	
	glActiveTexture(GL_TEXTURE0_ARB);
	blackImage.getTextureReference().bind();

	glActiveTexture(GL_TEXTURE1_ARB);
	maskLeftImage.getTextureReference().bind();

	
	glBegin(GL_QUADS);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, 0,0);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, 0,0);		
	glVertex2f( 2559,720);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, blackImage.width, 0);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, maskLeftImage.width, 0);		
	glVertex2f( 3840, 720);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB,  blackImage.width,blackImage.height);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, maskLeftImage.width,maskLeftImage.height);
	glVertex2f( 3840,0);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, 0,blackImage.height);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, 0,maskLeftImage.height);		
	glVertex2f( 2559,0);

	glEnd();
	
	//deactive and clean up
	glActiveTexture(GL_TEXTURE1_ARB);
	blackImage.getTextureReference().unbind();
	
	glActiveTexture(GL_TEXTURE0_ARB);
	maskLeftImage.getTextureReference().unbind();
	
	maskLeftShader.end();

	maskTopShader.begin();

	glActiveTexture(GL_TEXTURE0_ARB);
	blackImage.getTextureReference().bind();

	glActiveTexture(GL_TEXTURE1_ARB);
	maskTopImage.getTextureReference().bind();

	
	glBegin(GL_QUADS);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, 0,0);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, 0,0);		
	glVertex2f( 3841, 0);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, blackImage.width, 0);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, maskTopImage.width, 0);		
	glVertex2f( 5120, 0);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB,  blackImage.width,blackImage.height);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, maskTopImage.width,maskTopImage.height);
	glVertex2f(5120, 720);
	
	glMultiTexCoord2f(GL_TEXTURE0_ARB, 0,blackImage.height);
	glMultiTexCoord2f(GL_TEXTURE1_ARB, 0,maskTopImage.height);		
	glVertex2f(  3841, 720);

	glEnd();
	
	//deactive and clean up
	glActiveTexture(GL_TEXTURE1_ARB);
	blackImage.getTextureReference().unbind();
	
	glActiveTexture(GL_TEXTURE0_ARB);
	maskTopImage.getTextureReference().unbind();
	
	maskTopShader.end();//*/

	

	 

    ofSetHexColor(0xFFFFFF);
	ofDrawBitmapString("press space to pause",20,520);
    //if(frameByframe) ofSetHexColor(0xCCCCCC);
    //ofDrawBitmapString("mouse speed position",20,540);
    if(!frameByframe) ofSetHexColor(0xCCCCCC); else ofSetHexColor(0xFFFFFF);
	ofDrawBitmapString("gpuDeviceCount: " + ofToString(gpuDeviceCount),20,540);
    //ofDrawBitmapString("keys <- -> frame by frame " ,20,540);
    ofSetHexColor(0xFFFFFF);
	ofDrawBitmapString("Blur: " + ofToString(blurDistance), 20,560);
    ofDrawBitmapString("frame: " + ofToString(centerMovie.getCurrentFrame()) + "/"+ofToString(centerMovie.getTotalNumFrames()),20,580);
    ofDrawBitmapString("duration: " + ofToString(centerMovie.getPosition()*centerMovie.getDuration(),2) + "/"+ofToString(centerMovie.getDuration(),2),20,600);
    ofDrawBitmapString("speed: " + ofToString(centerMovie.getSpeed(),2),20,620);
	ofDrawBitmapString("VelocityRight: " + ofToString(velRight),150,640);
	ofDrawBitmapString("VelocityLeft: " + ofToString(velLeft),20,640);
	ofDrawBitmapString("VelocityAverage: " + ofToString(avg),20,660);
	ofDrawBitmapString("keyPointCount: " + ofToString(keyPointCount),20,680);
	ofDrawBitmapString("nextPointCount: " + ofToString(nextPointCount),20,690);
    if(centerMovie.getIsMovieDone()){
        ofSetHexColor(0xFF0000);
        ofDrawBitmapString("end of movie",20,640);
    }
	
	
}

//--------------------------------------------------------------
void testApp::keyPressed  (int key){
    switch(key){
        case ' ':
            frameByframe=!frameByframe;
            centerMovie.setPaused(frameByframe);
        break;
        case OF_KEY_LEFT:
            centerMovie.previousFrame();
        break;
        case OF_KEY_RIGHT:
            centerMovie.nextFrame();
        break;
        case '0':
            centerMovie.firstFrame();
			//centerMovie.setFrame(8750);
        break;
		case '1':
            blurDistance=3;
			gaussBlurDistance=2;
			numRows=1;
			mode3 = false;
        break;
		case '2':
            blurDistance=120;
			mode3 = false;
			numRows=1;
			gaussBlurDistance=10;
        break;
		case '3':
			mode3 = true;
			numRows = vel;
			//vel = 10;
			blurDistance=1;
			gaussBlurDistance=30;
        break;
		case '4':
			mode3 = true;
			numRows = 0;
			blurDistance=4;
			gaussBlurDistance=50;
        break;
		//case '5':
			//mode3 = true;
			//numRows = vel;
			//blurDistance=4;
			//gaussBlurDistance=20;
        //break;
		case '+':
		if (numRows > 175){
		numRows = 175;
		}
        numRows++;
        break;
		case '-':
		if (numRows < 1){
		numRows=1;
		}
        numRows--;
        break;
		case 'b':
			blurDistance++;
		break;
		case 'v':
			blurDistance--;
		break;
		case 'a':
			vel+=.1;
		break;
		case 's':
			vel-=.1;
		break;
		case 'c':
			sceneChange=true;
		break;
		case 'e':
			sceneChangeEnabled=true;
		break;
		case 'w':
			sceneChangeEnabled=false;
		break;
		case 'd':
			debugDraw = !debugDraw;
			debugDraw = (debugDraw);
		break;
		case 'f':
            //centerMovie.firstFrame();
			centerMovie.setFrame(centerMovie.getCurrentFrame()+500);
        break;
		case 'g':
            //centerMovie.firstFrame();
			centerMovie.setFrame(centerMovie.getCurrentFrame()-500);
        break;
    }
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y ){
	
	
}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){
	
	
}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){
	if(!frameByframe){
        centerMovie.setPaused(true);
	}
}


//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){
	if(!frameByframe){
        centerMovie.setPaused(false);
	}
}

//--------------------------------------------------------------
void testApp::resized(int w, int h){

}
