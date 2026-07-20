#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <algorithm>

namespace py = pybind11;


// this is a cpp implementation of the sum_pixel_values function, which sums pixel values in a given 
//window around specified pixel coordinates in an image. The function takes a 2D numpy array representing the image,
// a dictionary of grid positions mapping to pixel coordinates, the shape of the grid, and an optional window size parameter. 
//It returns a 2D numpy array containing the summed pixel values for each grid position.


// this can be used a drop in for the current pythonic implementation of sum_pixel_values. 

py::array_t<int> sum_pixel_values(
    py::array_t<uint16_t> image, 
    py::dict grid_positions, 
    std::vector<py::ssize_t> grid_shape, 
    int window_size = 10) 
{
    auto img = image.unchecked<2>(); // Access the image data as a 2D array
    py::ssize_t img_h = img.shape(0);
    py::ssize_t img_w = img.shape(1);
    py::ssize_t half_size = window_size / 2; 

    py::array_t<int> result({grid_shape[0], grid_shape[1]});
    std::fill(result.mutable_data(), result.mutable_data() + result.size(), 0); 
    auto res = result.mutable_unchecked<2>();

    for (auto item : grid_positions) {
        auto grid_coord = item.first.cast<std::pair<py::ssize_t, py::ssize_t>>(); 
        auto pixel_coord = item.second.cast<std::pair<py::ssize_t, py::ssize_t>>();
        
        py::ssize_t i = grid_coord.first;
        py::ssize_t j = grid_coord.second;
        py::ssize_t y = pixel_coord.first;
        py::ssize_t x = pixel_coord.second;

        py::ssize_t start_y = std::max<py::ssize_t>(y - half_size, 0);
        py::ssize_t end_y = std::min<py::ssize_t>(y + half_size + 1, img_h);
        py::ssize_t start_x = std::max<py::ssize_t>(x - half_size, 0);
        py::ssize_t end_x = std::min<py::ssize_t>(x + half_size + 1, img_w);

        int sum = 0;
        for (py::ssize_t row = start_y; row < end_y; ++row) {
            for (py::ssize_t col = start_x; col < end_x; ++col) {
                sum += img(row, col);
            }
        }
        res(i, j) = sum;
    }

    return result;
}

PYBIND11_MODULE(sum_pixel_values_cpp, m) {
    m.def("sum_pixel_values", &sum_pixel_values, 
          py::arg("image_array"), 
          py::arg("grid_positions"), 
          py::arg("grid_shape"), 
          py::arg("window_size") = 10);
}