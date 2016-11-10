#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string.h>
#include <math.h>
#include <gsl_rng.h>
#include <gsl_randist.h>
#include <gsl_sf_gamma.h>
#include <omp.h>

using namespace std;

int num_docs = -1;
int max_words = -1;
int num_features = -1;
int doc_start_id = -1;
vector<int> words_in_doc;
double GAMMA = 0.0;
double alpha = 0.0;
double SIGMA = 0.0;
//int vocab_size = 0;
gsl_rng *GSL_RNG;

bool temporal = false;

void split(const string &s, char delim, vector<string> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  split(s, delim, elems);
  return elems;
}

void computeParams(const char* fileName) {
  num_docs = 0;
  ifstream dataFile (fileName, ios::in);

  int frames_in_curr_video = 0;
  int curr_video_id = -1;
  for (string line; getline(dataFile, line); ) {
    vector<string> features = split(line, ' ');
    num_features = features.size() - 2;
    if (curr_video_id == atoi(features[0].c_str())) {
      frames_in_curr_video++;
    } else {
      if (curr_video_id == -1) {
        doc_start_id = atoi(features[0].c_str());
      } else {
        max_words = max(max_words, frames_in_curr_video);
        words_in_doc.push_back(frames_in_curr_video);
      }
      curr_video_id = atoi(features[0].c_str());
      frames_in_curr_video = 1;
      num_docs++;
    }
  }
  max_words = max(max_words, frames_in_curr_video);
  words_in_doc.push_back(frames_in_curr_video);

  printf("num_docs : %d\n", num_docs);
  printf("max_words: %d\n", max_words);
  printf("num_features : %d\n", num_features);
  for(int i=0; i<num_docs; i++) {
    cout << i << " " << words_in_doc[i] << "\n";
  }

  dataFile.close();
}

void readData(const char* fileName, double* W) {
  ifstream dataFile (fileName, ios::in);

  string line;
  for (int d=0; d<num_docs; d++) {
    for (int n=0; n<words_in_doc[d]; n++) {
      getline(dataFile, line);
      vector<string> features = split(line, ' ');
      //cout << features[0].c_str() << " " << d << "\n";
      printf ("d : %d \t n : %d\n", d, n);
      //printf ("doc_start_id : %d f0 : %d\n", doc_start_id, atoi(features[0].c_str()));
      assert (atoi(features[0].c_str()) == doc_start_id + d);
      //assert (atoi(features[1].c_str()) == n);
      for (int f=0; f<num_features; f++) {
        W[d * max_words * num_features + n * num_features + f] = 
            atof(features[f+2].c_str());
      }
    }
  }

  dataFile.close();
}

int indexOf(unsigned int value, unsigned int *arr, int size) {
  for (int i=0; i<size; i++) {
    if (arr[i] == value) {
      return i;
    }
  }
  for (int i=0; i<size; i++) {
    cout << arr[i] << " ";
  }
  cout << "\n";
  assert(false);
}

int generateMultinomialSample(double* prob, int size) {
  unsigned int* sample = new unsigned int[size];
  gsl_ran_multinomial(GSL_RNG, size, 1, prob, sample);
  int result = indexOf(1, sample, size);
  delete [] sample;
  return result;
}

class Model {
  double alpha;
  double GAMMA;
  int num_levels;
  int branching_factor;
  int num_paths;
  int num_nodes;
  int num_internal_nodes;
  double *beta;
  double *best_beta;
  
  void printDocCounts(int doc, int* countsPerDoc) {
    cout << "Doc : " << doc << "\n";
    for (int p=0; p<num_nodes; p++) {
      cout << countsPerDoc[doc * num_nodes + p] << "\t";
    }
    cout << "\n";
  }

  double compute_CRP_likelihood(int *counts) {
    double ll = 0.0;
    int num_nonzeros = 0;
    int sum_counts = 0;
    for (int i=0; i<branching_factor; i++) {
      if (counts[i] == 0) {
        num_nonzeros++;
      } else {
        ll += gsl_sf_lnfact(counts[i] - 1);
        sum_counts += counts[i];
      }
    }
    ll += log(GAMMA) * (num_nonzeros - 1);
    for (int k=1; k<sum_counts; k++) {
      ll -= log(k + GAMMA);
    }
    return ll;
  }

  double compute_NCRP_likelihood(int d, int num_words_in_doc, 
      int* countsPerDoc, int* Z) {
    double ll = 0.0;
    computeDocCounts(d, num_words_in_doc, countsPerDoc, Z);
    for (int start_idx=1; start_idx<num_nodes; start_idx+= branching_factor) {
      ll += compute_CRP_likelihood(countsPerDoc + d * num_nodes);
    }
    return ll;
  }

  double compute_log_likelihood(double* W, int* countsPerDoc, int* Z, double sigma) {
    double ll = 0.0;
    for (int p=0; p<num_paths; p++) {
      for (int f=0; f<num_features; f++) {
        ll += log(gsl_ran_gaussian_pdf(beta[p * num_features + f] - alpha, sigma));
      }
    }
    for (int d=0; d<num_docs; d++) {
      ll += compute_NCRP_likelihood(d, words_in_doc[d], countsPerDoc, Z);
    }
    for (int d=0; d<num_docs; d++) {
      for (int n=0; n<words_in_doc[d]; n++) {
        for (int f=0; f<num_features; f++) {
          ll += log(gsl_ran_gaussian_pdf(W[d * max_words * num_features + 
              n * num_features + f] - beta[Z[d * max_words + n] * num_features + f], sigma));
        }
      }
    }
    return ll;
  }

  public:
  
  Model(float alpha, float GAMMA, int num_levels, int branching_factor, 
      bool isTrain) {
    this->alpha = alpha;
    this->GAMMA = GAMMA;
    this->num_levels = num_levels;
    this->branching_factor = branching_factor;
    this->num_nodes = (pow(branching_factor, num_levels) - 1) / 
        (branching_factor - 1);
    this->num_paths = pow(branching_factor, num_levels - 1);
    this->num_internal_nodes = num_nodes - num_paths;

    cout << "Num levels : " << this->num_levels << "\n";
    cout << "BF : " << this->branching_factor << "\n";
    cout << "Num nodes : " << this->num_nodes << "\n";
    cout << "Num paths : " << this->num_paths << "\n";
    cout << "Num internal nodes : " << this->num_internal_nodes << "\n";

    if (isTrain) {
      beta = new double[num_paths * num_features];

      for (int p=0; p<num_paths; p++) {
        for (int f=0; f<num_features; f++) {
          beta[p * num_features + f] = alpha;
        }
      }

      best_beta = new double[num_paths * num_features];

      memcpy(best_beta, beta, num_paths * num_features * sizeof(double));
    }
  }

  ~Model() {
    delete [] beta;
    delete [] best_beta;
  }

  void computeDocCounts(int doc, int num_words_in_doc, int* countsPerDoc, 
      int* Z) {
    memset(countsPerDoc + doc * num_nodes, 0, num_nodes * sizeof(int));
    for (int n=0; n<num_words_in_doc; n++) {
      countsPerDoc[doc * num_nodes + num_internal_nodes + Z[doc * max_words + n]]++;
    }
    for (int p=num_nodes - 1; p>0; p--) {
      countsPerDoc[doc * num_nodes + (p-1)/branching_factor] += 
          countsPerDoc[doc * num_nodes + p];
    }
    // printDocCounts(doc);
  }

  void sample_beta(double* W, int* Z, double sigma) {
    double* counts = new double[num_paths];
    memset(counts, 0, num_paths * sizeof(double));
    double* sumW = new double[num_paths * num_features];
    memset(sumW, 0, num_paths * num_features * sizeof(double));

    for (int d=0; d<num_docs; d++) {
      for (int n=0; n<words_in_doc[d]; n++) {
        counts[Z[d * max_words + n]] ++;
        for (int f=0; f<num_features; f++) {
          sumW[Z[d * max_words + n] * num_features + f] += W[d * max_words + 
              n * num_features + f];
        }
      }
    }

    for (int p=0; p<num_paths; p++) {
      double sigma_p = sigma / (1 + counts[p]);
      for (int f=0; f<num_features; f++) {
        double mu_pf = (alpha + sumW[p * num_features + f]) / (1.0 + counts[p]);
        beta[p * num_features + f] = mu_pf + gsl_ran_gaussian(GSL_RNG, sigma_p);
      }
    }

    delete [] counts;
    delete [] sumW;
  }

  double likelihood_gaussian(double* W_dn, double* beta_p, double sigma) {
    double result = 0;
    //printf ("Lik. Gaussian : \n");
    for (int f=0; f<num_features; f++) {
      result += pow(beta_p[f], 2.0);
    }
    //printf ("Lik. Gaussian : after loop 1, result : %f\n", result);
    for (int f=0; f<num_features; f++) {
      result -= 2 * beta_p[f] * W_dn[f];
    }
    //printf ("Lik. Gaussian : after loop 2, result : %f\n", result);
    result = (-1.0 / (2.0 * sigma) * result);
    return result;
  }

  void sample_Zdn(double* W_dn, int d, int n, int num_words_in_doc, double temp,
      int* countsPerDoc, int* Z, double sigma) {
    // approximate prior
    double *multinomial_prob = new double[num_paths];
    double* likelihoods = new double[num_paths];
    double max_likelihood = -pow(10, 10);
    for (int p=0; p<num_paths; p++) {
      likelihoods[p] = likelihood_gaussian(W_dn, beta + p * num_features, sigma);
      max_likelihood = max(max_likelihood, likelihoods[p]);
    }
    for (int p=0; p<num_paths; p++) {
      multinomial_prob[p] = 
          pow(((double)countsPerDoc[d * num_nodes + num_internal_nodes + p] + GAMMA) / 
          (num_words_in_doc + num_paths * GAMMA), 1.0/temp) *
          exp((likelihoods[p] - max_likelihood)/temp);
      //multinomial_prob[p] = pow(multinomial_prob[p], 1.0/temp);
      //printf ("p : %d, mult[p]: %f\n", p, multinomial_prob[p]);
      //printf ("countsPerDoc : %d, Gamma : %f, num_words_in_doc : %d, lik_gauss: %f\n", 
      //    countsPerDoc[d * num_nodes + num_internal_nodes + p], GAMMA, num_words_in_doc,
      //    exp(likelihoods[p] - max_likelihood));
    }

    if (temporal) {
      for (int n1=0; n1<n; n1++) {
        int leaf_nodes_in_subtree_start = (Z[d * max_words + n1]) 
            / branching_factor * branching_factor;
        for (int p=leaf_nodes_in_subtree_start; p<Z[d * max_words + n1]; p++) {
          multinomial_prob[p] = 0.0;
        }
      }
    }

    //printf ("Before sampling\n");
    Z[d * max_words + n] = generateMultinomialSample(multinomial_prob, num_paths);
    //printf ("After sampling\n");

    delete [] multinomial_prob;
  }

  void update_best_configuration(double* W, int iter, int* countsPerDoc, int* Z, 
      int* best_Z, double &best_log_likelihood, int &last_update_iter, 
      bool isTrain, double sigma) {
    double curr_log_likelihood = compute_log_likelihood(W, countsPerDoc, Z, sigma);
    //printf ("Curr LL : %f\n", curr_log_likelihood);
    if (curr_log_likelihood > best_log_likelihood) {
      last_update_iter = iter;
      best_log_likelihood = curr_log_likelihood;
      memcpy(best_Z, Z, num_docs * max_words * sizeof(int));
      if (isTrain) {
        memcpy(best_beta, beta, num_paths * num_features * sizeof(double));
      }
    }
  }

  int getNumNodes() {
    return num_nodes;
  }

  void write_model_to_file(const char* prefix, double* W, int* best_Z) {
    char* beta_filename = new char[100];
    sprintf(beta_filename, "%s_beta.out", prefix);
    std::ofstream beta_file (beta_filename, ios::out);
    for (int p=0; p<num_paths; p++) {
      for (int f=0; f<num_features; f++) {
        beta_file << fixed << setprecision(6) 
                  << best_beta[p * num_features + f] << " ";
      }
      beta_file << "\n";
    }
    beta_file.close();
    delete [] beta_filename;

    char* Z_filename = new char[100];
    sprintf(Z_filename, "%s_Z.out", prefix);
    std::ofstream Z_file (Z_filename, ios::out);
    for (int d=0; d<num_docs; d++) {
      for (int n=0; n<max_words; n++) {
        Z_file  << ((W[d * max_words + n] == -1) ? -1 : best_Z[d * max_words + n])
                << " ";
      }
      Z_file << "\n";
    }
    Z_file.close();
    delete [] Z_filename;
  }

  void load_beta_from_file(const char* prefix) {
    char* beta_filename = new char[100];
    sprintf(beta_filename, "%s_beta.out", prefix);
    ifstream dataFile (beta_filename, ios::in);

    bool beta_initialized = false;

    int p=0;
    for (string line; getline(dataFile, line); p++) {
      vector<string> probs = split(line, ' ');
      num_features = probs.size() - 2;
      if (!beta_initialized) {
        beta = new double[num_paths * num_features];
        best_beta = new double[num_paths * num_features];
        beta_initialized = true;
      }
      for (int f=0; f<num_features; f++) {
        beta[p * num_features + f] = max(atof(probs[f].c_str()), 0.00);
      }
      double beta_sum = 0.0;
      for (int f=0; f<num_features; f++) {
        beta_sum += beta[p * num_features + f];
      }
      for (int f=0; f<num_features; f++) {
        beta[p * num_features + f] /= beta_sum;
      }
    }

    for (p=0; p<num_paths; p++) {
      for (int f=0; f<num_features; f++) {
        cout << fixed << setprecision(3) << beta[p * num_features + f] << " ";
      }
      cout << "\n";
    }
  }
};

void runGibbsSampling(double* W, const char* prefix, bool isTrain) {
  int num_levels = 4;
  int branching_factor = 3;
  Model *model = new Model(alpha, GAMMA, num_levels, branching_factor, isTrain);

  const int NUM_STARTS = 1;
  const int MAX_ITER = 10000;

  const double start_temp = 10.0;
  const double end_temp = 0.1;

  double curr_temp = start_temp;

  int* countsPerDoc = new int[num_docs * model->getNumNodes()];
  int* Z = new int[num_docs * max_words];
  int* best_Z = new int[num_docs * max_words];

  double best_log_likelihood = -pow(10, 10);
  int last_update_iter = -1;

  memset(Z, 0, num_docs * max_words * sizeof(int));
  memset(best_Z, 0, num_docs * max_words * sizeof(int));

  if (!isTrain) {
    model->load_beta_from_file(prefix);
  }

  double sigma_variable_comp = num_features;
  printf ("Starting optimization\n");
  for (int start = 0; start < NUM_STARTS; start++) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
      //printf ("Iter : %d -- Sampling Z\n", iter);
      double sigma = SIGMA + sigma_variable_comp;
      // sample Z
      memset(countsPerDoc, 0, num_docs * model->getNumNodes() * sizeof(int));

      for (int d = 0; d < num_docs; d++) {
        model->computeDocCounts(d, words_in_doc[d], countsPerDoc, Z);
        for (int n=0; n < words_in_doc[d]; n++) {
          //printf ("\t d : %d, n : %d\n", d, n);
          model->sample_Zdn(W + d * max_words * num_features + n, d, n, words_in_doc[d], 
              curr_temp, countsPerDoc, Z, sigma);
        }
      }
      //printf ("Iter : %d -- Sampled Z\n", iter);

      //printf ("Iter : %d -- Sampling beta\n", iter);
      // sample beta
      if (isTrain) {
        model->sample_beta(W, Z, sigma);
      }
      //printf ("Iter : %d -- Sampled beta\n", iter);

      //printf ("Updating best config\n", iter);
      // compute likelihood of the new configuration and update best
      model->update_best_configuration(W, iter, countsPerDoc, Z, best_Z,
          best_log_likelihood, last_update_iter, isTrain, sigma);
      //printf ("Updated best config\n", iter);

      cout  << fixed << setprecision(10)
            << "Iter " << iter << "\t" 
            << "Likelihood " << best_log_likelihood << "\t"
            << "Sigma " << sigma << "\t"
            << "Temp " << curr_temp << "\n";

      if (iter % 500 == 0 && isTrain) {
        char* prefix_iter = new char[100];
        sprintf(prefix_iter, "%s_%d", prefix, iter);
        model->write_model_to_file(prefix_iter, W, best_Z);
        delete [] prefix_iter;
      }

      if (iter - last_update_iter > 10) {
        last_update_iter = iter;
        curr_temp *= 0.9;
        sigma_variable_comp *= 0.5;
        if (curr_temp < end_temp) {
          break;
        }
      }
    }
    if (isTrain) { 
      model->write_model_to_file(prefix, W, best_Z);
    }
  }
  delete model;
  delete [] countsPerDoc;
  delete [] Z;
  delete [] best_Z;
}

int main(int argc, char* argv[]) {
  // parse commandline arguments
  if (argc != 7) {
    printf("Run as ./executable corpus alpha sigma GAMMA outfile_prefix train\n");
    printf("Run as ./executable corpus alpha sigma GAMMA infile_prefix test\n");
    exit(1);
  }

  istringstream alpha_str(argv[2]);
  alpha_str >> alpha;
  istringstream sigma_str(argv[3]);
  sigma_str >> SIGMA;
  istringstream gamma_str(argv[4]);
  gamma_str >> GAMMA;

  // read data
  computeParams(argv[1]);
  assert (num_docs > 0);
  assert (max_words > 0);
  double* W = new double[num_docs * max_words * num_features];
  /*
  for (int d=0; d<num_docs; d++) {
    for (int n=0; n<max_words; n++) {
      W[d*max_words + n] = -1;
    }
  }
  */
  readData(argv[1], W);

  printf("Read %d docs.\n", num_docs);
  printf("Read %d words.\n", max_words);

  GSL_RNG = gsl_rng_alloc(gsl_rng_mt19937);
  
  if (strcmp(argv[6], "train") == 0) {
    // run Gibbs sampler
    runGibbsSampling(W, argv[5], true);
  } else if (strcmp(argv[6], "test") == 0) {
    runGibbsSampling(W, argv[5], false);
  } else {
    exit(1);
  }

  delete [] W;
  gsl_rng_free(GSL_RNG);
  return 0;
}
