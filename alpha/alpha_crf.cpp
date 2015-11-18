#include "alpha_crf.hpp"
#include "brute_force.hpp"
#include <iostream>
#include <deque>
#include <limits>

void normalize(MatrixXf & in){
    for (int i=0; i<in.cols(); i++) {
        float col_min = in.col(i).minCoeff();
        in.col(i).array() -= col_min;
    }
}


//////////////////////////////////
// Constructors and Destructors //
//////////////////////////////////

AlphaCRF::AlphaCRF(int W, int H, int M, float alpha) : DenseCRF2D(W, H, M), alpha(alpha) {
}
AlphaCRF::~AlphaCRF(){}

// Overload the addition of the pairwise energy so that it adds the
// proxy-term with the proper weight
void AlphaCRF::addPairwiseEnergy(const MatrixXf & features, LabelCompatibility * function, KernelType kernel_type, NormalizationType normalization_type){
    assert(features.cols() == N_);
    VectorXf potts_weight = function->parameters();
    function->setParameters( alpha * potts_weight);
    DenseCRF::addPairwiseEnergy( new PairwisePotential( features, function, kernel_type, normalization_type));
    if (monitor_mode) {
        assert(potts_weight.rows() == 1);
        assert(potts_weight.cols() == 1);
        MatrixXf feat = features;
        pairwise_weights.push_back(potts_weight(0));
        pairwise_features.push_back(feat);
    }
}

void AlphaCRF::keep_track_of_steps(){
    monitor_mode = true;
};

void AlphaCRF::damp_updates(float damping_factor){
    use_damping = true;
    damping_factor = damping_factor;
}


void AlphaCRF::compute_exact_marginals(){
    exact_marginals_mode = true;
}
////////////////////
// Inference Code //
////////////////////

MatrixXf AlphaCRF::inference(){
    D("Starting inference to minimize alpha-divergence.");
    // Q contains our approximation, unary contains the true
    // distribution unary, approx_Q is the meanfield approximation of
    // the proxy-distribution
    MatrixXf Q(M_, N_), unary(M_, N_), approx_Q(M_, N_), new_Q(M_,N_);
// tmp1 and tmp2 are matrix to gather intermediary computations
    MatrixXf tmp1(M_, N_), tmp2(M_, N_);

    std::deque<MatrixXf> previous_Q;
    if(!unary_){
        unary.fill(0);
    } else {
        unary = unary_->get();
    }
    D("Initializing the approximating distribution");
    //expAndNormalize( Q, -unary); // Initialization by the unaries
    Q.fill(1/(float)M_); // Initialization to a uniform distribution
    D("Got initial estimates of the distribution");

    previous_Q.push_back(Q);
    bool continue_minimizing_alpha_div = true;
    float Q_change;
    int nb_approximate_distribution = 0;
    while(continue_minimizing_alpha_div){

        if (monitor_mode) {
            double ad = compute_alpha_divergence(unary, pairwise_features, pairwise_weights, Q, alpha);
            alpha_divergences.push_back(ad);
        }

        D("Constructing proxy distributions");
        // Compute the factors for the approximate distribution
        //Unaries
        MatrixXf true_unary_part = alpha* unary;
        MatrixXf approx_part = -1 * (1-alpha) * Q.array().log();
        proxy_unary = true_unary_part + approx_part;
        //// Pairwise term are created when we set up the CRF because they
        //// are going to remain the same
        // WARNING: numerical trick - we normalize the unaries, which
        // shouldn't change anything.  This consist in making the
        // smallest term 0, so that exp(-unary) isn't already way too
        // big.
        normalize(proxy_unary);
        D("Done constructing the proxy distribution");;

        if (exact_marginals_mode) {
            marginals_bf(approx_Q);
        } else{
            estimate_marginals(approx_Q, tmp1, tmp2);
        }

        D("Estimate the update rule parameters");
        tmp1 = Q.array().pow(alpha-1);
        tmp2 = tmp1.cwiseProduct(approx_Q);
        tmp2 = tmp2.array().pow(1/alpha);
        expAndNormalize(new_Q, tmp2);
        if(use_damping){
            Q = Q.array().pow(damping_factor) * new_Q.array().pow(1-damping_factor);
        } else {
            Q = new_Q;
        }

        float min_Q_change = std::numeric_limits<float>::max();
        for (std::deque<MatrixXf>::reverse_iterator prev = previous_Q.rbegin(); prev != previous_Q.rend(); prev++) {
            Q_change = (*prev - Q).squaredNorm();
            min_Q_change = min_Q_change < Q_change ? min_Q_change : Q_change;
            continue_minimizing_alpha_div = (min_Q_change > 0.001);
            if(not continue_minimizing_alpha_div){
                break;
            }
        }
        std::cout << '\r' <<  min_Q_change;
        std::cout.flush();
        previous_Q.push_back(Q);
        D("Updated our approximation");
        ++nb_approximate_distribution;
    }
    std::cout << '\n'; // Flush the Q_change monitoring line
    D("Done with alpha-divergence minimization");
    if (monitor_mode) {
        double ad = compute_alpha_divergence(unary, pairwise_features, pairwise_weights, Q, alpha);
        alpha_divergences.push_back(ad);
    }

    return Q;
}


// Reuse the same tempvariables at all step.
void AlphaCRF::mf_for_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2) {
    tmp1 = -proxy_unary;

    for (int i=0; i<pairwise_.size(); i++) {
        pairwise_[i]->apply(tmp2, approx_Q);
        tmp1 -= tmp2;
    }

    expAndNormalize(approx_Q, tmp1);
}

void AlphaCRF::estimate_marginals(MatrixXf & approx_Q, MatrixXf & tmp1, MatrixXf & tmp2){
    /**
     * approx_Q is a M_ by N_ matrix containing all our marginals that we want to estimate.
     * approx_Q_old .... that contains the previous marginals estimation so that we can estimate convergences.
     * tmp1 and tmp2 are also of the same size, they are temporary matrix, used to perform computations.
     * We pass all of these so that there is no need to reallocate / deallocate.
     */
    D("Starting to estimate the marginals of the distribution");
    // Starting value.
    //expAndNormalize(approx_Q, -proxy_unary); // Initialization by the unaries
    approx_Q.fill(1/(float)M_);// Uniform initialization

    std::deque<MatrixXf> previous_Q;
    previous_Q.push_back(approx_Q);

    // Setup the checks for convergence.
    bool continue_estimating_marginals = true;
    float marginal_change;
    int nb_marginal_estimation = 0;

    while(continue_estimating_marginals) {
        // Perform one meanfield iteration to update our approximation
        mf_for_marginals(approx_Q, tmp1, tmp2);
        // If we stopped changing a lot, stop the loop and
        // consider we have some good marginals
        float min_Q_change = std::numeric_limits<float>::max();
        for (std::deque<MatrixXf>::reverse_iterator prev = previous_Q.rbegin(); prev != previous_Q.rend(); prev++) {
            marginal_change = (*prev - approx_Q).squaredNorm();
            min_Q_change = min_Q_change < marginal_change ? min_Q_change : marginal_change;
            continue_estimating_marginals = (min_Q_change > 0.001);
            if(not continue_estimating_marginals){
                break;
            }
        }
        previous_Q.push_back(approx_Q);
        ++ nb_marginal_estimation;
    }
    D("Finished MF marginals estimation");
}

void AlphaCRF::marginals_bf(MatrixXf & approx_Q){
    std::vector<float> proxy_weights;
    for (int i = 0; i < pairwise_weights.size(); i++) {
        proxy_weights.push_back(pairwise_weights[i] * alpha);
    }
    approx_Q = brute_force_marginals(proxy_unary, pairwise_features, proxy_weights);

}
