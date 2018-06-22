#include "ofMain.h"
#include "testApp.h"
#include "ofAppGlutWindow.h"

//========================================================================
int main( ){
	//cv::gpu::getCudaEnabledDeviceCount();
	int x = 5120;
	int y = 725;
    ofAppGlutWindow window;
	ofSetupOpenGL(&window, x,y, OF_WINDOW);// <-------- setup the GL context
	ofSetWindowPosition(0,-5);
	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN or OF_GAME_MODE
	// pass in width and height too:
	ofRunApp( new testApp());

}
