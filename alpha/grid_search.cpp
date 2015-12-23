#include "inference.hpp"
#include "evaluation.hpp"
#include "file_storage.hpp"
#include <fstream>

#define NUMLABELS 22

void save_confusion_matrix(const std::vector<int>& confMat, const std::string& filename, const size_t num_labels)
{
    std::ofstream fs(filename.c_str());
    for(int i = 0; i < num_labels * num_labels; ++i)
    {
        fs << confMat[i] << (i % num_labels == (num_labels-1) ? '\n' : ',');
    }
}


int main(int argc, char *argv[])
{
    std::string path_to_dataset = "/home/rudy/datasets/MSRC/";
    std::string dataset_split = "Validation";

    std::vector<std::string> test_images = get_all_split_files(path_to_dataset, dataset_split);

    for (float w1 =1; w1< 11; w1+=0.5) {
        for (float sigma_alpha = 30; sigma_alpha<80; sigma_alpha+=5) {
            for (float sigma_beta=1; sigma_beta<6; sigma_beta+=0.5) {

                std::vector<label_matrix> gts(test_images.size());
                std::vector<label_matrix> res(test_images.size());

#pragma omp parallel for
                for(int i=0; i< test_images.size(); ++i){
                    std::string image_name = test_images[i];
                    std::string image_path = get_image_path(path_to_dataset, image_name);
                    std::string unaries_path = get_unaries_path(path_to_dataset, image_name);
                    std::string gt_path = get_ground_truth_path(path_to_dataset, image_name);

                    label_matrix gt = load_label_matrix(gt_path);
                    label_matrix img_res = minimize_mean_field(image_path, unaries_path, w1, sigma_alpha, sigma_beta);

                    gts[i] = gt;
                    res[i] = img_res;
                }

                std::vector<int> total_conf_mat(NUMLABELS * NUMLABELS, 0);
                for (int i=0; i<gts.size(); i++) {
                    evaluate_segmentation(gts[i], res[i], total_conf_mat, NUMLABELS);
                }

                double score = pixwise_acc_from_confmat(total_conf_mat, NUMLABELS);
                std::ofstream outfile;
                outfile.open("Grid-search.txt", std::ios_base::app);
                outfile << w1 << '\t' << sigma_alpha << '\t' << sigma_beta << ":\t" << score << std::endl;
                outfile.close();

                save_confusion_matrix(total_conf_mat, "conf_mat.csv", NUMLABELS);

            }
        }
    }

    return 0;
}
