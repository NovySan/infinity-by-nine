#ifndef _TEST_APP
#define _TEST_APP

#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"
#include <cv.h>
#include <highgui.h>
#include "opencv2\gpu\gpu.hpp"


//#define _USE_LIVE_VIDEO		// uncomment this to use a live camera
								// otherwise, we'll use a movie file


class testApp : public ofBaseApp{

	public:

		void setup();
		void update();
		void draw();

		void keyPressed  (int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void resized(int w, int h);
		

		#ifdef _USE_LIVE_VIDEO
		  ofVideoGrabber 		vidGrabber;
		#else
		  ofVideoPlayer 		vidPlayer;
		#endif

        int						gpuDeviceCount;
		  
		ofxCvColorImage			colorImg;
		ofxCvColorImage			colorTopImg;
		ofxCvColorImage			colorRightImg;
		ofxCvColorImage			colorRightPastImg;
		ofxCvColorImage			colorRightTempImg;
		ofxCvColorImage			colorLeftImg;
		ofxCvGrayscaleImage 	grayImage;
		ofxCvGrayscaleImage 	grayBg;
		ofxCvGrayscaleImage 	grayDiff;
		ofxCvFloatImage			map_x;  
		ofxCvFloatImage			map_y; 

		//Histogram Tools
		IplImage*				iplImageGray;
		IplImage*				iplImagePastGray;
		IplImage**				plane;
		IplImage**				planePast;

		CvHistogram*			curr_hist;
		CvHistogram*			past_hist;

        //ofxCvContourFinder 	contourFinder;
		//ofxCvOpticalFlowLK		flow;

		std::vector<cv::KeyPoint>	keyPoints;
		std::vector<cv::Point2f>	points_keyPoints, points_nextPoints;

		cv::Mat					src, dst; 

		unsigned char *			grayDiffPixels;

		int 				threshold;
		float				diffPercent;
		
		bool				bLearnBakground;
		bool				sceneChange;
		bool				sceneChangeEnabled;

		ofVideoPlayer 		centerMovie;
		
		ofImage				blackImage;
		ofImage				maskImage;
		ofImage				maskLeftImage;
		ofImage				maskTopImage;

		ofTexture			movieFullTexture;
		ofTexture			topImage;
		ofTexture			rightImage;
		ofTexture			leftImage;
		
		ofShader			maskRightShader;
		ofShader			maskLeftShader;
		ofShader			maskTopShader;
		ofShader			blurRightHorizShader;
		ofShader			blurRightVertShader;
		ofShader			blurLeftHorizShader;
		ofShader			blurLeftVertShader;
		ofShader			blurTopShader;

		bool                frameByframe;
		bool				mode3;
		bool				clearTex;
		bool				panLeft;
		bool				panRight;
		bool				debugDraw;
		
		int 			w, h;
		int				numRows;
		int				numRowsLeft;
		int				numRowsRight;
		int				numColUp;
		int				numColDown;
		float			blurDistance;
		int				gaussBlurDistance;
		int				vel;
		int				velLeft;
		int				velRight;
		int				velUp;
		int				velDown;
		float			leftMask;
		float			rightMask;
		int				pan;
		int				avg;
		int				pastIntRight;
		int				pastIntLeft;
		int				pastIntUp;
		int				pastIntDown;
		int				interpAvgRight;
		int				interpAvgLeft;
		int				interpAvgUp;
		int				interpAvgDown;
		int				sc;
		float			verticalDistanceToMove;
		float			horizontalDistanceToMove;

		unsigned char 	* topPixels;
		unsigned char   * rightPixels;
		unsigned char   * leftPixels;
		unsigned char 	* moviePixels;
		unsigned char 	* movieFullPixels;
		unsigned char	* rightFinalPixels;
		unsigned char	* leftFinalPixels;
		
};

#endif
