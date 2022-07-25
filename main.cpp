#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <string>
#include <vector>

#define cascadePath "D:/c++/py_ball_detection/ball_cascade.xml"

namespace py = pybind11;

class Rect {
public:
    int x;
    int y;
    int w;
    int h;
    Rect() {}
    void fromCvRect(const cv::Rect& rect) {
        x = rect.x;
        y = rect.y;
        w = rect.width;
        h = rect.height;
    }
};

static cv::CascadeClassifier cascade;

py::array_t<Rect>* detect(py::array_t<uint8_t>& data) {
    if (cascade.empty()) cascade.load(cascadePath);
    py::buffer_info buf = data.request();
    cv::Mat frame(data.shape(0), data.shape(1), CV_8UC3, (unsigned char*)buf.ptr);
    cv::Mat frame_gray;
    cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> balls;
    cascade.detectMultiScale(frame_gray, balls, 1.1, 5, 8, cv::Size(16, 16));
    auto rez = new py::array_t<Rect>(balls.size());
    for (int i = 0; i < balls.size(); i++) {
        ((Rect&)rez->at(i)).fromCvRect(balls[i]);
    }
    return rez;
}

PYBIND11_MODULE(ball_detector, m) {
    py::class_<Rect>(m, "Rect")
    .def(py::init<>())
    .def_readwrite("x", &Rect::x)
    .def_readwrite("y", &Rect::y)
    .def_readwrite("w", &Rect::w)
    .def_readwrite("h", &Rect::h)
    .def("__repr__", [](const Rect& rect) -> std::string {
        return "Rect";
    });
    PYBIND11_NUMPY_DTYPE(Rect, x, y, w, h);
    m.def("detect", &detect);
}