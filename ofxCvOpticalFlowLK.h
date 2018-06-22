/*
	Dhi Aurrahman
	dio.rahman@gmail.com

	Nurul Arif Setiawan
	n.arif.setiawan@gmail.com
*/

/**
 *	Notes
 *	This implementation use OpenCV function
 *	void cvCalcOpticalFlowPyrLK( const CvArr* prev, const CvArr* curr, ... );
 *  void cvGoodFeaturesToTrack( const CvArr* image, CvArr* eig_image, ... );
 *  ...
 */

#ifndef __OPTICALFLOW_H__
#define __OPTICALFLOW_H__

#include <cv.h>
#include "ofMain.h"

/**
 *	Constants for sparse point allocation.
 *  Make larger if more points are expected
 */
const static int MAX_COUNT_SPARSE = 400;

/**
 *	Optical flow type. 
 *	Dense will sampling rectangular grid of specified window size.
 *  Sparse will select good corner features.
 */
enum FlowType { 
	flowDense = 1,		
	flowSparse
};

/**
 *	Drawing type of optical flow for debug image.
 *  Vector as in vector, with direction and magnitude.
 *	Angle only show angle with same magnitude.
 */
enum FlowDrawType { 
	drawVector = 1,
	drawAngle
};

/**
 * Optical flow data.
 */
struct FlowData
{
	IplImage *currPyramid, *prevPyramid;
	IplImage *prevImg, *currImg; 
	IplImage *opticalFlowField;
	CvPoint2D32f* points[2];
	char* status;
};

/**
 * Optical flow parameters.
 */
struct FlowParams
{
	int count;
	int winSize;
	int gridStep;
	int pyrLevel;
	int flags;
	int isDense;
	float maxThresFlow;
	float minThresFlow;
	CvTermCriteria terminationCriteria;
};


/**
 *  Optical Flow Class. 
 *
 *	From OpenCV :
 *	The function implements sparse iterative version of Lucas-Kanade optical flow in pyramids. 
 *	It calculates coordinates of the feature points on the current video frame 
 *	given their coordinates on the previous frame. 
 *	The function finds the coordinates with sub-pixel accuracy.
 *  
 *	This class implementation : 
 *    - Constructor - initialize flow data 
 *    - Destructor - deallocates all of memory allocation.
 *    - Update optical flow
 *    - Get Optical flow data and Optical flow image
 *
 */
class ofxCvOpticalFlowLK
{
public:

	ofxCvOpticalFlowLK();

	/**
	 * Constructor. 
	 */
	ofxCvOpticalFlowLK(CvSize size, FlowParams *params = 0);

	/**
	 * Destructor.
	 * deallocating the variable 'flowdata' .
	 */
	~ofxCvOpticalFlowLK();

	/**
	* Update optical flow data .
	*
	* This data contains the new optical flow between two images ( one image is in flowdata - previous image ) .
	* After calculation , copy img to previous img
	* @param img   current image
	* @return void
	*/
	void updateFlowData(IplImage *img);

	int updateFlowVelocity(IplImage *img);

	/**
	* Get FlowData pointer
	*
	* @return * FlowData
	*/
	FlowData * getFlowData();

	/**
	* Get FlowData pointer
	*
	* @return  * FlowParams
	*/
	FlowParams * getFlowParams();

	/**
	* Get Optical flow image
	*
	* @return optical flow image that draw the optical flow vector in color image
	*/
	IplImage* getOpticalFlowImage( int drawtype = drawVector );

	/**
	* Initialize optical flow data and the variable 'flowdata' will be setted .
	*
	* @param size  image size
	* @param type  optical flow type - flow_dense, flow_sparse
	* @return void
	*/
	void initFlowData( CvSize size, FlowParams *params );

	/**
	* Draw flow data
	*/
	void draw( int drawtype = drawVector );

private:

	/**
	* Optical flow data.
	*/
	FlowData m_FlowData;

	/**
	* Optical flow parameters.
	*/
	FlowParams m_FlowParams;

	/**
	* Release flow data
	*/
	void releaseFlowData();

	/**
	* Helper for drawing arrow's head
	*/
	void drawArrowHead(CvPoint point1, CvPoint point2);
	void drawArrowHead(CvPoint point1, CvPoint point2, IplImage *display_image, CvScalar color);

	int isInit;
};
#endif