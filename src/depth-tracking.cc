#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
//#include <zmq.hpp>
#include <algorithm>
#include <cmath>

extern "C" {
    IplImage *freenect_sync_get_depth_cv(int index);
    IplImage *freenect_sync_get_rgb_cv(int index);
}

using namespace std;
using namespace cv;

/*
 * Hardware dependent code for getting image rgb
 * Abstract kinect or asus MI code
 * */
void get_image(Mat &frame) {
    // cvtColor(Mat(freenect_sync_get_rgb_cv(0)), frame, CV_RGB2BGR); // Kinect version
	cvtColor(Mat(freenect_sync_get_rgb_cv(0)), frame, CV_RGB2BGR);
}


/*
 * Hardware dependent code for getting image depth
 * Abstract kinect or asus MI code
 */
void get_depth(Mat &frame) {
    // frame = Mat(freenect_sync_get_depth_cv(0)); // Kinect version
	frame = Mat(freenect_sync_get_depth_cv(0));
	
}


string pos_to_json(int x, int y, int theta, int k) {
    stringstream message_stream;
    message_stream << "[{\"x\": ";
    message_stream << x;
    message_stream << ", \"y\": ";
    message_stream << y;
    message_stream << ", \"theta\": 0, \"k\": 11}]";
    return message_stream.str();
}


//TODO: broadcast feature not implemented
/*
void broadcast_pos_json(zmq::socket_t &s, int x, int y) {
    if (x == 0 && y == 0) {
        string message = "[]";
        zmq::message_t msg(message.size());
        memcpy(msg.data(), message.c_str(), message.size());
        s.send(msg);
    }
    else {
        string message = pos_to_json(x, y, 0, 11);
        cout << message << endl;

        zmq::message_t msg(message.size());
        memcpy(msg.data(), message.c_str(), message.size());
        s.send(msg);
    }
}

void broadcast_pos_bin(zmq::socket_t &s, int x, int y) {
    if (x == 0 && y == 0) return;
    zmq::message_t msg(2*sizeof(int));
    memcpy(msg.data(), &x, sizeof(int));
    memcpy((char *)msg.data()+sizeof(int), &y, sizeof(int));
    s.send(msg);
}

void broadcast_pos(zmq::socket_t &s, int x, int y) {
    broadcast_pos_bin(s, x, y);
}

*/

double depth_to_meters(int raw_depth)
{
    if (raw_depth < 2048)
        return 0.1236 * std::tan(raw_depth/2842.5 + 1.1863);
    return 0;
}

unsigned short *gen_depth(void)
{
    unsigned short *m_gamma = new unsigned short[2048];
    for (int i = 0 ; i < 2048 ; i++) {
        float v = i/2048.0;
        v = std::pow(v, 3)* 6;
        m_gamma[i] = v*6*256;
    }
    return m_gamma;
}

Mat colorize_depth(Mat d)
{
    static unsigned short *m_gamma = gen_depth();
    Mat m(480, 640, CV_8UC3);
    unsigned char *m_buffer_depth = m.data;
    unsigned short *depth = (unsigned short *)d.data;
    for (int i = 0 ; i < 640*480 ; i++) {
        int pval = m_gamma[depth[i]];
        int lb = pval & 0xff;
        switch (pval >> 8) {
            case 0:
                m_buffer_depth[3*i+0] = 255;
                m_buffer_depth[3*i+1] = 255-lb;
                m_buffer_depth[3*i+2] = 255-lb;
                break;
            case 1:
                m_buffer_depth[3*i+0] = 255;
                m_buffer_depth[3*i+1] = lb;
                m_buffer_depth[3*i+2] = 0;
                break;
            case 2:
                m_buffer_depth[3*i+0] = 255-lb;
                m_buffer_depth[3*i+1] = 255;
                m_buffer_depth[3*i+2] = 0;
                break;
            case 3:
                m_buffer_depth[3*i+0] = 0;
                m_buffer_depth[3*i+1] = 255;
                m_buffer_depth[3*i+2] = lb;
                break;
            case 4:
                m_buffer_depth[3*i+0] = 0;
                m_buffer_depth[3*i+1] = 255-lb;
                m_buffer_depth[3*i+2] = 255;
                break;
            case 5:
                m_buffer_depth[3*i+0] = 0;
                m_buffer_depth[3*i+1] = 0;
                m_buffer_depth[3*i+2] = 255-lb;
                break;
            default:
                m_buffer_depth[3*i+0] = 0;
                m_buffer_depth[3*i+1] = 0;
                m_buffer_depth[3*i+2] = 0;
                break;
        }
    }
    return m;
}

int main(int, char **)
{

    //zmq::context_t ctx(1);
    //zmq::socket_t s(ctx, ZMQ_PUB);
    //s.bind("tcp://*:6969");
    //zmq::socket_t s_json(ctx, ZMQ_PUB);
    //s_json.bind("tcp://*:6968");

    namedWindow("rgb", 1);
    //namedWindow("depth", 1);

    Mat camera_matrix, distortion_coefficients, rectification_matrix,
        projection_matrix, undistort_RGB, map1_RGB, map2_RGB,
        undistort_depth, map1_depth, map2_depth;

    double cx_rgb, cy_rgb, fx_rgb, fy_rgb;
    //double cx_d, cy_d, fx_d, fy_d;

    FileStorage calibRGB("calibration_rgb.yaml", FileStorage::READ);
    calibRGB["camera_matrix"] >> camera_matrix;
    calibRGB["distortion_coefficients"] >> distortion_coefficients;
    calibRGB["rectification_matrix"] >> rectification_matrix;
    calibRGB["projection_matrix"] >> projection_matrix;
    Size image_size(calibRGB["image_width"], calibRGB["image_height"]);

    initUndistortRectifyMap(camera_matrix, distortion_coefficients,
            rectification_matrix, undistort_RGB, image_size, CV_16SC2,
            map1_RGB, map2_RGB);

    fx_rgb = camera_matrix.at<double>(0, 0);
    fy_rgb = camera_matrix.at<double>(1, 1);
    cx_rgb = camera_matrix.at<double>(1, 2);
    cy_rgb = camera_matrix.at<double>(1, 3);

    cout << fx_rgb << ' ' << fy_rgb << ' ' << cx_rgb << ' ' << cy_rgb << endl;

    FileStorage calibdepth("calibration_depth.yaml", FileStorage::READ);
    calibdepth["camera_matrix"] >> camera_matrix;
    calibdepth["distortion_coefficients"] >> distortion_coefficients;
    calibdepth["rectification_matrix"] >> rectification_matrix;
    calibdepth["projection_matrix"] >> projection_matrix;
    image_size = Size(calibdepth["image_width"], calibdepth["image_height"]);

    initUndistortRectifyMap(camera_matrix, distortion_coefficients,
            rectification_matrix, undistort_depth, image_size, CV_16SC2,
            map1_depth, map2_depth);

    Mat frame, undistortImg, undistortDepth, depth;
    get_image(frame);
    get_depth(depth);
    Mat labImg(frame.size(), CV_8UC3);
    Mat img = labImg.clone();
    Mat depth_img = depth.clone();

    char c = waitKey(10);

#ifdef DEBUG_FPS
    int ticks, ticksprev;
    double freq = getTickFrequency();
    ticks = getTickCount();
#endif
    Mat l(img.size(), CV_8UC1);
    Mat a(img.size(), CV_8UC1);
    Mat b(img.size(), CV_8UC1);

    //Mat min(img.size(), CV_8UC3, Scalar(16, 32, 128));
    //Mat max(img.size(), CV_8UC3, Scalar(96, 192, 255));
    Mat min(img.size(), CV_8UC3, Scalar(0, 150, 0));
    Mat max(img.size(), CV_8UC3, Scalar(255, 255, 255));

    Mat newMat, out;

    Point minLoc;

    vector<vector<Point> > contours;
    Mat hierarchy;
    Rect r;
    double minDepth;
    int x,y, x_3d, y_3d = 0;

    Mat tmp;
    //int bilaterial_d=3;
    //createTrackbar( "lalala", "Trackbar Demo", &bilaterial_d,10,GaussianBlur );  
    while(c != 'q')
    {
        get_image(frame);
        get_depth(depth);

        img = frame.clone();
        //depth_img = colorize_depth(depth);
        Mat canny, greyDepth;
        // Implicitly threshold the depth image to 256/alpha, alpha being the third arg.
        // 256/0.256 = 1000, a reasonable threshold for the depth image.
        depth.convertTo(greyDepth, CV_8UC1, 0.256, 0);

        // Fill in shadows in the image.
        unsigned char prev = 0;
        for (int i = 0; i < greyDepth.rows; ++i) {
            for (int j = 0; j < greyDepth.cols; ++j) {
                unsigned char cur = greyDepth.at<unsigned char>(i, j);
                if (cur != 0)
                    prev = cur;
                else
                    greyDepth.at<unsigned char>(i, j) = prev;
            }
        }

        // Find sharp edges in the depth image
        cv::Canny(greyDepth, canny, 200, 80, 5, true);

        // Take the useful part of the image
        // (that is, chop off top & sides, where the sensor is more prone to noise.)
        Mat interestingCanny = canny(Rect(0, 120, 620, 260));

        // Blend together some of the edges and get rid of any noise, find contours, smooth images.
        cv::GaussianBlur(interestingCanny, interestingCanny, Size(3, 3), 1);
	//bilateralFilter(interstingCanny, interestingCanny, 3, 30, 30);
      
	findContours(interestingCanny, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 120));

        // Find largest contour in image
        double maxArea = 500, area;
        int maxContour = -1;
        for (unsigned int i = 0; i < contours.size(); i++) {
            area = contourArea(contours[i]);
            if (area > maxArea) {
                r = boundingRect(contours[i]);
                if (r.width > 20 && r.height > 20) {
                    maxContour = i;
                    maxArea = area;
                }
            }
        }
	
        // If we found one, calculate the position
        if (maxContour > 0) {
	    /* Below are added by Bicheng
	    if (maxArea>15000 && i_count==0){
		fix_win=boundingRect(contours[maxContour]);
		i_count+=1;
		
	    }
	    r=CamShift(img,fex_win,trac_criteria);	
		*/
	    
            r = boundingRect(contours[maxContour]);
            r.x += 10;
            r.y += 20;

            rectangle(img, r, Scalar(0, 255, 0), 2);
            r.x -= 10;
            r.y -= 20;
            //rectangle(depth_img, r, Scalar(0, 255, 0), 2);
            minMaxLoc(depth(r), &minDepth, 0, &minLoc);
            x = r.x + r.width/2;
	    y=  r.y + r.height/2;	
            if (depth_to_meters(minDepth) == 0) goto showimg;

            x_3d = (2.5+(x - cx_rgb)*depth_to_meters(minDepth) / fx_rgb)*100;
	    y_3d = (2.5+(y - cy_rgb)*depth_to_meters(minDepth) / fy_rgb)*100;
            //y_3d = (0.4)*(depth_to_meters(minDepth) - 1) * 175 + 0.6*y_3d;
            //y_3d = (1)*(depth_to_meters(minDepth) - 0.2) * 175 + 0.0*y_3d;

			//TODO: broadcast feature temporary disabled
            //broadcast_pos(s, x_3d, y_3d);
            //broadcast_pos_json(s_json, x_3d, y_3d,minDepth,area);
            //broadcast_pos_json(s_json, x_3d, y_3d);
        } else {
			//TODO: broadcast feature temporary disabled
            //broadcast_pos(s, 0, 0);
            //broadcast_pos_json(s_json, 0, 0,0,0);
            //broadcast_pos_json(s_json, 0, 0);
        }

showimg:
        imshow("rgb", img);
        //imshow("depth", depth_img);
        c = waitKey(1);
        if(c == 'p')
        {
            do {
                c = waitKey(0);
            } while(c != 'p');
        }

#ifdef DEBUG_FPS
        ticksprev = ticks;
        ticks = getTickCount();
        cout << freq / (ticks - ticksprev) << endl;
#endif
    }

    return 0;
}
