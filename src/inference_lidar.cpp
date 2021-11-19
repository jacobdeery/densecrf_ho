#include <chrono>
#include <ctime>
#include <fstream>
#include <vector>
#include <string>
#include "file_storage.hpp"
#include "densecrf.h"


struct sp_params {
    float const_1;
    float norm_1;
};

MatrixXf compute_unaries(const std::vector<double>& points) {
    MatrixXf unaries(2, 3);

    const double eps = 1e-5;  // we can't take log(0)

    // As written: 0 = ground, 1 = non-ground
    // For the hard-coded example:
    // - point 1 is definitely ground
    // - point 2 is uncertain
    // - point 3 is definitely non-ground
    // We expect point 2 to converge to ground

    unaries(0, 1) = -log(1 - eps);
    unaries(0, 2) = -log(0.5);
    unaries(0, 3) = -log(eps);

    unaries(1, 1) = -log(eps);
    unaries(1, 2) = -log(0.5);
    unaries(1, 3) = -log(1 - eps);

    return unaries;
}

void lidar_inference(std::string method,
                     std::string path_to_results,
                     std::string path_to_image,
                     std::string path_to_unaries,
                     float spc_std,
                     float spc_potts,
                     float bil_spcstd,
                     float bil_colstd,
                     float bil_potts,
                     LP_inf_params & lp_params,
                     sp_params params)
{
    MatrixXf Q;

    const int num_classes = 2;

    std::vector<double> points{0, 0, 0, 1, 1, 1, 10, 10, 10};
    MatrixXf unaries = compute_unaries(points);

    DenseCRF2D crf(points.size() / 3, num_classes);
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseLidar(points, new PottsCompatibility(bil_potts));
    crf.addPairwiseGaussian(spc_std, spc_std, new PottsCompatibility(spc_potts));

    Q = crf.unary_init();

    if (method == "lp") {
        std::cout << "---Running tests on proximal LP\r\n";
        Q = crf.lp_inference_prox_super_pixels(Q, lp_params);
    } else if (method == "qp"){ //qp with a non convex energy function, relaxations removed, just run without super pixels
        std::cout << "---Running tests on QP with non convex energy\r\n";
        Q = crf.qp_inference_super_pixels_non_convex(Q);
    } else{
        std::cout << "Unrecognised method.\n Proper error handling would do something but I won't." << '\n';
    }

    double discretized_energy;
    if (method == "qp_sp")
    {
        discretized_energy = crf.assignment_energy_higher_order(crf.currentMap(Q));
    }
    else if (method == "prox_lp_sp")
    {
        discretized_energy = crf.assignment_energy_higher_order(crf.currentMap(Q));
    }
    else
    {
        discretized_energy = crf.assignment_energy_true(crf.currentMap(Q));
    }
    save_map(Q, size, path_to_results, "MSRC");
    std::cout << "# method: " << method << '\t' << discretized_energy << std::endl;
    delete[] img;
}

int main(int argc, char *argv[])
{
    if (argc<2) {
        std::cout << "Usage: ./example_inference method spc_std spc_potts bil_spcstd bil_colstd bil_potts [const_1 const_2 const_3 norm_1 norm_2 norm_3]" << '\n';
        std::cout << "Example: ./example_inference qp_sp [3 5 30 5 10 [10 1000]]" << '\n';
        return 1;
    }

    std::string method = argv[1];
    std::string image = "../data/img.bmp";
    std::string unaries = "../data/img.c_unary";
    std::string output = "../data/seg.png";

    //pairwise params
    float spc_std = 2.0;
    float spc_potts = 3.0;
    float bil_potts = 2.0;
    float bil_spcstd = 30.0;
    float bil_colstd = 8.0;

    if (argc == 7)
    {
        spc_std = std::stof(argv[2]);
        spc_potts = std::stof(argv[3]);
        bil_spcstd = std::stof(argv[4]);
        bil_colstd = std::stof(argv[5]);
        bil_potts = std::stof(argv[6]);
    }

    //params containing higher order terms
    sp_params params = {50, 1000};
    if (argc == 9)
    {
        params = sp_params {std::stof(argv[7]),
                            std::stof(argv[8])};
    }

    // lp inference params
    LP_inf_params lp_params;
    lp_params.prox_energy_tol = lp_params.dual_gap_tol;

    image_inference(method, output, image, unaries, spc_std, spc_potts,
                        bil_spcstd, bil_colstd, bil_potts, lp_params, params);

}
