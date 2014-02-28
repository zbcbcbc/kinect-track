#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <zmq.hpp>
#include <algorithm>

extern "C" {
    IplImage *freenect_sync_get_depth_cv(int index);
    IplImage *freenect_sync_get_rgb_cv(int index);
}

using namespace std;
using namespace cv;

void get_image(Mat &frame) {
    cvtColor(Mat(freenect_sync_get_rgb_cv(0)), frame, CV_RGB2BGR);
}
void get_depth(Mat &frame) {
    frame = Mat(freenect_sync_get_depth_cv(0));
}

void pos_to_json(int x, int y, int theta, int k, string &message) {
    stringstream message_stream;
    message_stream << "[{\"x\": ";
    message_stream << x;
    message_stream << ", \"y\": ";
    message_stream << y;
    message_stream << ", \"theta\": 0, \"k\": 11}]";
    message = message_stream.str();
}

double depth_to_meters(int raw_depth)
{
    if (raw_depth < 1020)
        return 1.0 / (raw_depth * -0.0030711016 + 3.3309495161);
    return 0;
}

int main(int, char **)
{

    zmq::context_t ctx(1);
    zmq::socket_t s(ctx, ZMQ_PUB);
    s.bind("tcp://*:6969");

    namedWindow("rgb", 1);

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
    Mat min(img.size(), CV_8UC3, Scalar(-100, -150, 0));
    Mat max(img.size(), CV_8UC3, Scalar(255, 150, 255));

    Mat newMat, out;

    Point minLoc;

    int fromTo[] = {0, 0, 1, 1, 2, 2};
    Mat lab[] = {l, a, b};

    vector<vector<Point> > contours;
    Mat hierarchy;
    Rect r;
    double minDepth;
    int x, x_3d, y_3d = 0;

    Mat tmp;

    while(c != 'q')
    {
        get_image(frame);
        get_depth(depth);
        img = frame.clone();
        //remap(img, undistortImg, map1_RGB, map2_RGB, INTER_LINEAR);
        cvtColor(img, labImg, CV_BGR2Lab);

        newMat = (labImg > min) & (labImg < max);
        mixChannels(&newMat, 1, lab, 3, fromTo, 3);

        newMat = l & a & b;
        //erode(newMat, out, Mat());
        //dilate(newMat, out, Mat());
        morphologyEx(newMat, out, MORPH_CLOSE, Mat());

        findContours(out, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

        double maxArea = 30, area;
        int maxContour = -1;
        for (unsigned int i = 0; i < contours.size(); i++) {
            area = contourArea(contours[i]);
            if (area > maxArea) {
                r = boundingRect(contours[i]);
                if (r.width > 5 && r.height > 5 && r.y > 200) {
                    maxContour = i;
                    maxArea = area;
                }
            }
        }

        if (maxContour > 0) {
            r = boundingRect(contours[maxContour]);
            int newx = std::max(r.x-r.width/2, 0);
            int newy = std::max(r.y-r.height/2, 0);
            r = Rect(newx, newy,
                    std::min(r.width*2, 640-newx), std::min(r.height*2, 480-newy));
            //if (r.width > 5 && r.height > 5) {
            {
                string message;
                rectangle(img, r, Scalar(0, 255, 0), 2);
                rectangle(depth_img, r, Scalar(0, 255, 0), 2);
                //remap(depth, undistortDepth, map1_depth, map2_depth,
                //        INTER_LINEAR);
                minMaxLoc(depth(r), &minDepth, 0, &minLoc);
                x = r.x + r.width/2;
                //y = r.y + r.height/2;
                if (depth_to_meters(minDepth) == 0) goto showimg;
                //cout << x << ' ' << y << endl;
                //cout << (x - cx_rgb)*depth_to_meters(depth.at<short>(y, x)) / fx_rgb << ' ' <<
                //(y - cy_rgb)*depth_to_meters(depth.at<short>(y, x)) / fy_rgb << ' ' <<
                //depth_to_meters(depth.at<short>(y, x)) << endl;

                x_3d = (2.5+(x - cx_rgb)*depth_to_meters(minDepth) / fx_rgb)*110;
                y_3d = (0.3)*(depth_to_meters(minDepth) - 1) * 110 + 0.7*y_3d;
                if (x_3d > 450 || x_3d < 50) goto showimg;
                pos_to_json(x_3d, y_3d, 0, 11, message);

                zmq::message_t msg(message.size());
                memcpy(msg.data(), message.c_str(), message.size());
                s.send(msg);
                cout << minDepth << " --> ";
                cout << depth_to_meters(minDepth) << " --> ";
                cout << (depth_to_meters(minDepth) - 1) * 100 << endl;
                //cout << (int)((2.5+(x - cx_rgb)*depth_to_meters(depth.at<short>(y, x)) / fx_rgb)*100) << ' ' <<
                //(int)((depth_to_meters(depth.at<short>(y, x)) - 1) * 100) << endl;
                //circle(img, Point(x, y), 2, Scalar(0, 255, 0), 2);
                //cout << depth_to_meters(minDepth) << endl;
            }
        }

showimg:
        imshow("rgb", img);
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
