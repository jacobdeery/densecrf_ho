#include <Eigen/Core>

using namespace Eigen;

struct img_size {
    int width;
    int height;
};

const unsigned char legend[22*3] = {
                                   128,0,0,
                                    0,128,0,
                                    128,128,0,
                                    0,0,128,
                                    //horses are ignored 128,0,128,
                                    0,128,128,
                                    128,128,128,
                                    //mountains are also ignored 64,0,0,
                                    192,0,0,
                                    64,128,0,
                                    192,128,0,
                                    64,0,128,
                                    192,0,128,
                                    64,128,128,
                                    192,128,128,
                                    0,64,0,
                                    128,64,0,
                                    0,192,0,
                                    128,64,128,
                                    0,192,128,
                                    128,192,128,
                                    64,64,0,
                                   192,64,0,
                                   0,0,0
                                    };



std::vector<std::string> get_all_split_files(const std::string & path_to_dataset, const std::string & dataset_split);
std::string get_image_path(const std::string & path_to_dataset, const std::string & image_name);
std::string get_unaries_path(const std::string & path_to_dataset, const std::string & image_name);
std::string get_ground_truth_path(const std::string & path_to_dataset, const std::string & image_name);
std::string get_output_path(const std::string & path_to_results_folder, const std::string & image_name);

unsigned char* load_image(const std::string& path_to_image, img_size size);
Matrix<short,Dynamic,1> load_labeling(const std::string & path_to_labels, img_size& size);
MatrixXf load_unary(const std::string & path_to_unary, img_size& size);
void save_map(const MatrixXf & estimates, const img_size &  size, const std::string &  path_to_output);
