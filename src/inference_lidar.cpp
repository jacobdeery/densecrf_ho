#include <vector>
#include <string>
#include "densecrf.h"


std::vector<double> load_lidar() {
    return {
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        1, 1, 0,
        0, 2, 0,
        1, 2, 0
    };
}

MatrixXf get_probabilities(const std::vector<double>& points) {
    MatrixXf probabilities(2, points.size() / 3);

    const double eps = 1e-5;  // we can't take log(0)

    // Classes: 0 = ground, 1 = non-ground

    std::vector<double> ground_probs{
        1 - eps,
        1 - eps,
        0.5,
        0.3,
        eps,
        eps
    };

    for (size_t i = 0; i < points.size() / 3; ++i) {
        probabilities(0, i) = ground_probs[i];
        probabilities(1, i) = 1 - ground_probs[i];
    }

    return probabilities;
}

MatrixXf compute_unaries(const MatrixXf& probabilities) {
    MatrixXf unaries(2, probabilities.cols());

    const double eps = 1e-5;  // we can't take log(0)

    // As written: 0 = ground, 1 = non-ground

    std::vector<double> ground_probs{
        1 - eps,
        0.7,
        0.3,
        eps
    };

    for (size_t i = 0; i < probabilities.cols(); ++i) {
        unaries(0, i) = -log(probabilities(0, i));
        unaries(1, i) = -log(probabilities(1, i));
    }

    return unaries;
}

std::vector<int> lidar_inference(std::string method,
                     const std::vector<double>& points,
                     float bil_potts,
                     LP_inf_params & lp_params)
{
    MatrixXf Q;

    const int num_classes = 2;

    MatrixXf probabilities = get_probabilities(points);
    MatrixXf unaries = compute_unaries(probabilities);

    std::cout << "prior probabilities:" << std::endl << probabilities << std::endl;

    DenseCRF2D crf(points.size() / 3, 1, num_classes);
    crf.setUnaryEnergy(unaries);
    crf.addPairwiseLidar(1, points, new PottsCompatibility(bil_potts));

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

    std::vector<int> labeling(points.size());
    for(int i = 0; i < Q.cols(); ++i) {
        int lbl;
        Q.col(i).maxCoeff( &lbl);
        labeling[i] = lbl;
    }

    std::cout << "energy: " << discretized_energy << std::endl;

    return labeling;
}

int main(int argc, char *argv[])
{
    std::string method = "qp";

    //pairwise params
    float spc_std = 2.0;
    float spc_potts = 3.0;
    float bil_potts = 2.0;
    float bil_spcstd = 30.0;
    float bil_colstd = 8.0;

    // lp inference params
    LP_inf_params lp_params;
    lp_params.prox_energy_tol = lp_params.dual_gap_tol;

    std::vector<double> points = load_lidar();

    const auto predictions = lidar_inference(method, points, bil_potts, lp_params);

    std::cout << "predictions: " << std::endl << predictions << std::endl;;
}
