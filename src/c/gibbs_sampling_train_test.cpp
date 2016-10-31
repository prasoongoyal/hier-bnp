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
double GAMMA = 0.0;
double alpha = 0.0;
int vocab_size = 0;
gsl_rng *GSL_RNG;

bool temporal = true;

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

  for (string line; getline(dataFile, line); ) {
    num_docs++;
    vector<string> words = split(line, ' ');
    if ((int)words.size() > max_words) {
      max_words = words.size();
    }
  }

  dataFile.close();
}

void readData(const char* fileName, int* W) {
  ifstream dataFile (fileName, ios::in);

  int doc = 0;
  for (string line; getline(dataFile, line); doc++) {
    vector<string> words = split(line, ' ');
    int words_size = words.size();
    int i;
    for (i=0; i<words_size; i++) {
      W[doc * max_words + i] = atoi(words[i].c_str());
      vocab_size = max(vocab_size, W[doc * max_words + i]);
    }
    while (i<max_words) {
      W[doc * max_words + i] = -1;
      i++;
    }
  }
  vocab_size++;
}

int getNumWords(int* W_d) {
  int num_words_in_doc = 0;
  for (int n=0; n<max_words; n++) {
    if (W_d[n] == -1) {
      break;
    }
    num_words_in_doc++;
  }
  return num_words_in_doc;
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

  double compute_log_likelihood(int *W, int* countsPerDoc, int* Z) {
    double ll = 0.0;
    for (int p=0; p<num_paths; p++) {
      for (int v=0; v<vocab_size; v++) {
        ll += (alpha - 1) * log(beta[p * vocab_size + v]);
      }
    }
    for (int d=0; d<num_docs; d++) {
      ll += compute_NCRP_likelihood(d, getNumWords(W + d * max_words), 
          countsPerDoc, Z);
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
      beta = new double[num_paths * vocab_size];

      for (int p=0; p<num_paths; p++) {
        for (int v=0; v<vocab_size; v++) {
          beta[p * vocab_size + v] = 1.0 / vocab_size;
        }
      }

      best_beta = new double[num_paths * vocab_size];

      memcpy(best_beta, beta, num_paths * vocab_size * sizeof(double));
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

  void sample_beta(int *W, int* Z) {
    double *counts = new double[num_paths * vocab_size];
    for (int p=0; p<num_paths; p++) {
      for (int v=0; v<vocab_size; v++) {
        counts[p * vocab_size + v] = alpha;
      }
    }
    for (int d=0; d<num_docs; d++) {
      for (int n=0; n<max_words; n++) {
        if (W[d * max_words + n] != -1) {
          counts[Z[d * max_words + n] * vocab_size + W[d * max_words + n]] ++;
        }
      }
    }
    for (int p=0; p<num_paths; p++) {
      gsl_ran_dirichlet(GSL_RNG, vocab_size, counts + p * vocab_size, \
          beta + p * vocab_size);
    }

    delete [] counts;
  }

  void sample_Zdn(int* W_d, int d, int n, int num_words_in_doc, double temp,
      int* countsPerDoc, int* Z) {
    // approximate prior
    double *multinomial_prob = new double[num_paths];
    for (int p=0; p<num_paths; p++) {
      multinomial_prob[p] = 
          ((double)countsPerDoc[d * num_nodes + num_internal_nodes + p] + GAMMA) / 
          (num_words_in_doc + num_paths * GAMMA) *
          beta[p * vocab_size +  W_d[n]];
      multinomial_prob[p] = pow(multinomial_prob[p], 1.0/temp);
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

    unsigned int *sample = new unsigned int[num_paths];
    gsl_ran_multinomial(GSL_RNG, num_paths, 1, multinomial_prob, sample);
    Z[d * max_words + n] = indexOf(1, sample, num_paths);

    delete [] multinomial_prob;
    delete [] sample;
  }

  void update_best_configuration(int *W, int iter, int* countsPerDoc, int* Z, 
      int* best_Z, double &best_log_likelihood, int &last_update_iter, 
      bool isTrain) {
    double curr_log_likelihood = compute_log_likelihood(W, countsPerDoc, Z);
    if (curr_log_likelihood > best_log_likelihood) {
      last_update_iter = iter;
      best_log_likelihood = curr_log_likelihood;
      memcpy(best_Z, Z, num_docs * max_words * sizeof(int));
      if (isTrain) {
        memcpy(best_beta, beta, num_paths * vocab_size * sizeof(double));
      }
    }
  }

  int getNumNodes() {
    return num_nodes;
  }

  void write_model_to_file(const char* prefix, int* W, int* best_Z) {
    char* beta_filename = new char[100];
    sprintf(beta_filename, "%s_beta.out", prefix);
    std::ofstream beta_file (beta_filename, ios::out);
    for (int p=0; p<num_paths; p++) {
      for (int v=0; v<vocab_size; v++) {
        beta_file << fixed << setprecision(6) 
                  << best_beta[p * vocab_size + v] << " ";
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
      vocab_size = probs.size();
      if (!beta_initialized) {
        beta = new double[num_paths * vocab_size];
        best_beta = new double[num_paths * vocab_size];
        beta_initialized = true;
      }
      for (int v=0; v<vocab_size; v++) {
        beta[p * vocab_size + v] = max(atof(probs[v].c_str()), 0.000001);
      }
      double beta_sum = 0.0;
      for (int v=0; v<vocab_size; v++) {
        beta_sum += beta[p * vocab_size + v];
      }
      for (int v=0; v<vocab_size; v++) {
        beta[p * vocab_size + v] /= beta_sum;
      }
    }

    for (p=0; p<num_paths; p++) {
      for (int v=0; v<vocab_size; v++) {
        cout << fixed << setprecision(3) << beta[p * vocab_size + v] << " ";
      }
      cout << "\n";
    }
  }
};

void runGibbsSampling(int* W, const char* prefix, bool isTrain) {
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

  for (int start = 0; start < NUM_STARTS; start++) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
      // sample Z
      memset(countsPerDoc, 0, num_docs * model->getNumNodes() * sizeof(int));
      #pragma omp parallel for
      for (int d = 0; d < num_docs; d++) {
        int num_words_in_doc = getNumWords(W + d * max_words);
        model->computeDocCounts(d, num_words_in_doc, countsPerDoc, Z);
        for (int n=0; n < num_words_in_doc; n++) {
          model->sample_Zdn(W + d * max_words, d, n, num_words_in_doc, 
              curr_temp, countsPerDoc, Z);
        }
      }

      // sample beta
      if (isTrain) {
        model->sample_beta(W, Z);
      }

      // compute likelihood of the new configuration and update best
      model->update_best_configuration(W, iter, countsPerDoc, Z, best_Z,
          best_log_likelihood, last_update_iter, isTrain);

      cout  << fixed << setprecision(10)
            << "Iter " << iter << "\t" 
            << "Likelihood " << best_log_likelihood << "\t"
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
  if (argc != 6) {
    printf("Run as ./executable corpus alpha GAMMA outfile_prefix train\n");
    printf("Run as ./executable corpus alpha GAMMA infile_prefix test\n");
    exit(1);
  }

  istringstream alpha_str(argv[2]);
  alpha_str >> alpha;
  istringstream gamma_str(argv[3]);
  gamma_str >> GAMMA;

  // read data
  computeParams(argv[1]);
  assert (num_docs > 0);
  assert (max_words > 0);
  int *W = new int[num_docs * max_words];
  for (int d=0; d<num_docs; d++) {
    for (int n=0; n<max_words; n++) {
      W[d*max_words + n] = -1;
    }
  }
  readData(argv[1], W);

  printf("Read %d docs.\n", num_docs);
  printf("Read %d words.\n", max_words);

  printf("Vocab size : %d\n", vocab_size);

  GSL_RNG = gsl_rng_alloc(gsl_rng_mt19937);
  
  if (strcmp(argv[5], "train") == 0) {
    // run Gibbs sampler
    runGibbsSampling(W, argv[4], true);
  } else if (strcmp(argv[5], "test") == 0) {
    runGibbsSampling(W, argv[4], false);
  } else {
    exit(1);
  }

  delete [] W;
  gsl_rng_free(GSL_RNG);
  return 0;
}
