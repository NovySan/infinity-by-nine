
#include <stdio.h>
#include "ofxCvOpticalFlowLK.h"

ofxCvOpticalFlowLK::ofxCvOpticalFlowLK()
{
	isInit = 0;
}

ofxCvOpticalFlowLK::ofxCvOpticalFlowLK(CvSize size, FlowParams *params)
{	
	initFlowData(size, params);
	isInit = 1;
}

ofxCvOpticalFlowLK::~ofxCvOpticalFlowLK(void)
{
	if (isInit)
		releaseFlowData();
}

// TODO: load params from xml file
void ofxCvOpticalFlowLK::initFlowData(CvSize size, FlowParams *_params)
{
	// allocate IplImages
	m_FlowData.opticalFlowField = cvCreateImage( size, IPL_DEPTH_8U, 3 );
	m_FlowData.currPyramid = cvCreateImage(size, IPL_DEPTH_8U, 1 );
	m_FlowData.prevPyramid = cvCreateImage(size, IPL_DEPTH_8U, 1 );
	m_FlowData.prevImg = cvCreateImage(size, IPL_DEPTH_8U, 1 );
	m_FlowData.currImg = cvCreateImage(size, IPL_DEPTH_8U, 1 );
	m_FlowData.currImg->origin = 0;
	m_FlowData.prevImg->origin = 0;
	m_FlowData.opticalFlowField->origin = 0;
	m_FlowData.currPyramid->origin = 0;
	m_FlowData.prevPyramid->origin = 0;

	// if params is empty
	// Default parameters set 
	// TODO: loading from xml file should be better
	if( _params == 0 )
	{
		m_FlowParams.isDense = flowDense;
		m_FlowParams.count = 0;
		m_FlowParams.winSize = 64;
		m_FlowParams.gridStep = 64;
		m_FlowParams.flags = 0;
		m_FlowParams.pyrLevel = 3;//Default = 3
		m_FlowParams.minThresFlow = 4;
		m_FlowParams.maxThresFlow = 32;
		m_FlowParams.terminationCriteria = cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .03 );
	}
	else
		this->m_FlowParams = *_params;

	// allocating points sampling
	// if we want dense
	if (m_FlowParams.isDense == flowDense)
	{
		int pointSize = (size.width / m_FlowParams.gridStep ) * (size.height / (m_FlowParams.gridStep));
	//	printf("pointSize = %d\n",pointSize);

		// Allocate points and init necessary parameters
		m_FlowData.points[0] = (CvPoint2D32f*)cvAlloc(pointSize*sizeof(m_FlowData.points[0][0]));
		m_FlowData.points[1] = (CvPoint2D32f*)cvAlloc(pointSize*sizeof(m_FlowData.points[0][0]));
		m_FlowData.status = (char*)cvAlloc(pointSize);

		// Uniform grid sampling
		for(int j = m_FlowParams.gridStep; j < size.height; j+=m_FlowParams.gridStep) {
			for(int i = m_FlowParams.gridStep; i < size.width; i+=m_FlowParams.gridStep) {
				m_FlowData.points[0][m_FlowParams.count] = cvPoint2D32f(i, j);
				m_FlowParams.count++;
			}
		}
	//	printf("count = %d\n",m_FlowParams.count);
	}
	else
	// if we want sparse by selection of good features
	{
		// Allocate points and init necessary parameters
		m_FlowData.points[0] = (CvPoint2D32f*)cvAlloc(MAX_COUNT_SPARSE*sizeof(m_FlowData.points[0][0]));
		m_FlowData.points[1] = (CvPoint2D32f*)cvAlloc(MAX_COUNT_SPARSE*sizeof(m_FlowData.points[0][0]));
		m_FlowData.status = (char*)cvAlloc(MAX_COUNT_SPARSE);
	}

	isInit = 1;
}

void ofxCvOpticalFlowLK::releaseFlowData()
{
	cvReleaseImage( &m_FlowData.opticalFlowField );
	cvReleaseImage( &m_FlowData.currPyramid );
	cvReleaseImage( &m_FlowData.prevPyramid );
	cvReleaseImage( &m_FlowData.currImg );
	cvReleaseImage( &m_FlowData.prevImg );
	cvFree((void**) &m_FlowData.points[0]);
	cvFree((void**) &m_FlowData.points[1]);
	cvFree((void**) &m_FlowData.status);
}

void ofxCvOpticalFlowLK::updateFlowData(IplImage *img){
	// update previous data
	cvCopy(m_FlowData.currPyramid, m_FlowData.prevPyramid);
	cvCopy(m_FlowData.currImg, m_FlowData.prevImg);
	cvCopy(img, m_FlowData.currImg);

	// if flow type is dense
	if (m_FlowParams.isDense)
	{
		for(int i = 0; i < m_FlowParams.count; i++)
			m_FlowData.points[1][i] = cvPoint2D32f(0,0);

		// calculate optical flow
		cvCalcOpticalFlowPyrLK(m_FlowData.prevImg, m_FlowData.currImg, m_FlowData.prevPyramid, m_FlowData.currPyramid,
			m_FlowData.points[0], m_FlowData.points[1], m_FlowParams.count, cvSize(m_FlowParams.winSize, m_FlowParams.winSize), 
			m_FlowParams.pyrLevel, m_FlowData.status, 0, m_FlowParams.terminationCriteria, m_FlowParams.flags);

		m_FlowParams.flags |= CV_LKFLOW_PYR_A_READY;
	}
	else
	// if flow type is sparse
	{
		// select good features to track
		IplImage *eig_image = NULL, *temp_image = NULL;
		eig_image = cvCreateImage(cvGetSize(m_FlowData.prevImg), IPL_DEPTH_32F, 1 );
		temp_image = cvCreateImage(cvGetSize(m_FlowData.prevImg), IPL_DEPTH_32F, 1 );
		m_FlowParams.count = MAX_COUNT_SPARSE;

		cvGoodFeaturesToTrack(m_FlowData.prevImg, eig_image, temp_image, 
			m_FlowData.points[0], &m_FlowParams.count, .000001, .000001);

		// calculate optical flow
		cvCalcOpticalFlowPyrLK(m_FlowData.prevImg, m_FlowData.currImg, m_FlowData.prevPyramid, m_FlowData.currPyramid,
			m_FlowData.points[0], m_FlowData.points[1], m_FlowParams.count, cvSize(m_FlowParams.winSize, m_FlowParams.winSize), 
			m_FlowParams.pyrLevel, m_FlowData.status, 0, m_FlowParams.terminationCriteria, m_FlowParams.flags);

		m_FlowParams.flags |= CV_LKFLOW_PYR_A_READY;
	}
	

}

int ofxCvOpticalFlowLK::updateFlowVelocity(IplImage *img){
	// update previous data
	cvCopy(m_FlowData.currPyramid, m_FlowData.prevPyramid);
	cvCopy(m_FlowData.currImg, m_FlowData.prevImg);
	cvCopy(img, m_FlowData.currImg);

	// if flow type is dense
	if (m_FlowParams.isDense)
	{
		for(int i = 0; i < m_FlowParams.count; i++)
			m_FlowData.points[1][i] = cvPoint2D32f(0,0);

		// calculate optical flow
		cvCalcOpticalFlowPyrLK(m_FlowData.prevImg, m_FlowData.currImg, m_FlowData.prevPyramid, m_FlowData.currPyramid,
			m_FlowData.points[0], m_FlowData.points[1], m_FlowParams.count, cvSize(m_FlowParams.winSize, m_FlowParams.winSize), 
			m_FlowParams.pyrLevel, m_FlowData.status, 0, m_FlowParams.terminationCriteria, m_FlowParams.flags);

		m_FlowParams.flags |= CV_LKFLOW_PYR_A_READY;
	}
	else
	// if flow type is sparse
	{
		// select good features to track
		IplImage *eig_image = NULL, *temp_image = NULL;
		eig_image = cvCreateImage(cvGetSize(m_FlowData.prevImg), IPL_DEPTH_32F, 1 );
		temp_image = cvCreateImage(cvGetSize(m_FlowData.prevImg), IPL_DEPTH_32F, 1 );
		m_FlowParams.count = MAX_COUNT_SPARSE;

		cvGoodFeaturesToTrack(m_FlowData.prevImg, eig_image, temp_image, 
			m_FlowData.points[0], &m_FlowParams.count, .000001, .000001);

		// calculate optical flow
		cvCalcOpticalFlowPyrLK(m_FlowData.prevImg, m_FlowData.currImg, m_FlowData.prevPyramid, m_FlowData.currPyramid,
			m_FlowData.points[0], m_FlowData.points[1], m_FlowParams.count, cvSize(m_FlowParams.winSize, m_FlowParams.winSize), 
			m_FlowParams.pyrLevel, m_FlowData.status, 0, m_FlowParams.terminationCriteria, m_FlowParams.flags);

		m_FlowParams.flags |= CV_LKFLOW_PYR_A_READY;
	}
	
	int sum = 0;
	int avg = 0;
	for (int i = 0; i < m_FlowParams.count; i++){
		// Threshold velocities: not drawing very small or very large velocities 
		if((abs((m_FlowData.points[0][i].x - m_FlowData.points[1][i].x)) >= m_FlowParams.minThresFlow || 
			abs((m_FlowData.points[0][i].y - m_FlowData.points[1][i].y)) >= m_FlowParams.minThresFlow) &&
			(abs((m_FlowData.points[0][i].x - m_FlowData.points[1][i].x)) <= m_FlowParams.maxThresFlow && 
			abs((m_FlowData.points[0][i].y - m_FlowData.points[1][i].y)) <= m_FlowParams.maxThresFlow)) 
			{
			sum += (m_FlowData.points[0][i].x - m_FlowData.points[1][i].x);
			}
	}
		avg = sum / m_FlowParams.count;
		int vel = avg;
		return vel;

}

FlowData* ofxCvOpticalFlowLK::getFlowData()
{
	return &m_FlowData;
}

FlowParams* ofxCvOpticalFlowLK::getFlowParams()
{
	return &m_FlowParams;
}

IplImage* ofxCvOpticalFlowLK::getOpticalFlowImage(int drawtype)
{
	// If we want to draw with background image or not
	cvCvtPlaneToPix(m_FlowData.currImg, m_FlowData.currImg, m_FlowData.currImg, NULL, m_FlowData.opticalFlowField);

	CvPoint2D32f center = cvPoint2D32f(370, 370);
	CvPoint2D32f translate_direction = cvPoint2D32f(370, center.y - 250);

	for (int i = 0; i < m_FlowParams.count; i++) 
	{
		// Threshold velocities: not drawing very small or very large velocities 
		if((abs((m_FlowData.points[0][i].x - m_FlowData.points[1][i].x)) >= m_FlowParams.minThresFlow || 
			abs((m_FlowData.points[0][i].y - m_FlowData.points[1][i].y)) >= m_FlowParams.minThresFlow) &&
			(abs((m_FlowData.points[0][i].x - m_FlowData.points[1][i].x)) <= m_FlowParams.maxThresFlow && 
			abs((m_FlowData.points[0][i].y - m_FlowData.points[1][i].y)) <= m_FlowParams.maxThresFlow)) 
		{
			// flow vector end point
			CvPoint2D32f endpoint;
			endpoint.x = m_FlowData.points[1][i].x;
			endpoint.y = m_FlowData.points[1][i].y;

			float angle = atan2(m_FlowData.points[1][i].y - m_FlowData.points[0][i].y,
				m_FlowData.points[1][i].x - m_FlowData.points[0][i].x);

			if (drawtype == drawAngle) {
				endpoint.x = m_FlowData.points[0][i].x + 25 * cos(angle);
				endpoint.y = m_FlowData.points[0][i].y + 25 * sin(angle);
			}

			CvScalar color = CV_RGB(255, 0, 0);

		//	cvCircle(m_FlowData.opticalFlowField, cvPointFrom32f(m_FlowData.points[0][i]), 2, CV_RGB(0, 0, 255), -1, 8, 0);
			
			// The flow
			cvLine(m_FlowData.opticalFlowField, 
				cvPointFrom32f(m_FlowData.points[0][i]), cvPointFrom32f(endpoint), color, 5, 8, 0);
			// with head
			drawArrowHead(cvPointFrom32f(m_FlowData.points[0][i]), cvPointFrom32f(endpoint), 
				m_FlowData.opticalFlowField, color);			
		}
	}

	return m_FlowData.opticalFlowField;
}

void ofxCvOpticalFlowLK::drawArrowHead(CvPoint point1, CvPoint point2, IplImage *display_image, CvScalar color)
{
	CvPoint p1 = cvPoint((int)(point1.x + 0.7*(point2.x - point1.x)),
		(int)(point1.y + 0.7*(point2.y - point1.y)));

	CvPoint p2 = cvPoint((int)(point1.x + 0.7*(point2.x - point1.x)), point2.y);

	CvPoint p3 = cvPoint(point2.x, (int)(point1.y + 0.7*(point2.y - point1.y)));

	CvPoint p4 = cvPoint((int)(p1.x + 0.5*(p2.x-p1.x)), (int)(p1.y + 0.5*(p2.y-p1.y)));
	CvPoint p5 = cvPoint((int)(p1.x + 0.5*(p3.x-p1.x)), (int)(p1.y + 0.5*(p3.y-p1.y)));

	cvLine(display_image, point2, p4, color, 1, 8, 0);
	cvLine(display_image, point2, p5, color, 1, 8, 0);
}

// arrow head in GL environment
void ofxCvOpticalFlowLK::drawArrowHead(CvPoint point1, CvPoint point2){

	CvPoint p1 = cvPoint((int)(point1.x + 0.7*(point2.x - point1.x)),
		(int)(point1.y + 0.7*(point2.y - point1.y)));

	CvPoint p2 = cvPoint((int)(point1.x + 0.7*(point2.x - point1.x)), point2.y);

	CvPoint p3 = cvPoint(point2.x, (int)(point1.y + 0.7*(point2.y - point1.y)));

	CvPoint p4 = cvPoint((int)(p1.x + 0.5*(p2.x-p1.x)), (int)(p1.y + 0.5*(p2.y-p1.y)));
	CvPoint p5 = cvPoint((int)(p1.x + 0.5*(p3.x-p1.x)), (int)(p1.y + 0.5*(p3.y-p1.y)));

	ofLine(point2.x,point2.y,p4.x,p4.y);
	ofLine(point2.x,point2.y,p5.x,p5.y);
}

void ofxCvOpticalFlowLK::draw( int drawtype){

	ofEnableAlphaBlending();
	ofSetHexColor(0xFF0000);
	ofNoFill();
	int sum = 0;
	int avg = 0;
	for (int i = 0; i < m_FlowParams.count; i++) 
	{
		// Threshold velocities: not drawing very small or very large velocities 
		if((abs((m_FlowData.points[0][i].x - m_FlowData.points[1][i].x)) >= m_FlowParams.minThresFlow || 
			abs((m_FlowData.points[0][i].y - m_FlowData.points[1][i].y)) >= m_FlowParams.minThresFlow) &&
			(abs((m_FlowData.points[0][i].x - m_FlowData.points[1][i].x)) <= m_FlowParams.maxThresFlow && 
			abs((m_FlowData.points[0][i].y - m_FlowData.points[1][i].y)) <= m_FlowParams.maxThresFlow)) 
		{
			// flow vector end point
			CvPoint2D32f endpoint;
			endpoint.x = m_FlowData.points[1][i].x;
			endpoint.y = m_FlowData.points[1][i].y;

			float angle = atan2(m_FlowData.points[1][i].y - m_FlowData.points[0][i].y,
				m_FlowData.points[1][i].x - m_FlowData.points[0][i].x);

			if (drawtype == drawAngle) {
				endpoint.x = m_FlowData.points[0][i].x + 25 * cos(angle);
				endpoint.y = m_FlowData.points[0][i].y + 25 * sin(angle);
			}

			ofLine((m_FlowData.points[0][i]).x,(m_FlowData.points[0][i]).y,endpoint.x,endpoint.y);
			
			sum += (m_FlowData.points[0][i].x - m_FlowData.points[1][i].x);
			
			// with head
			drawArrowHead(cvPointFrom32f(m_FlowData.points[0][i]), cvPointFrom32f(endpoint));			
		}
	}
			avg = sum / m_FlowParams.count;
			int vel = avg;
			char reportStr[1024];
			sprintf(reportStr, "velocity = %i", vel);
			ofDrawBitmapString(reportStr,0, 0);
			if (vel < 0){
				ofFill();
				ofCircle( 575,175,25);
				ofDrawBitmapString("Pan Left!",625,175);
			}
			if (vel > 0){
				ofFill();
				ofCircle( 0,175,25);
				ofDrawBitmapString("Pan Right!",-125,175);
			}
	ofDisableAlphaBlending();
}