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
int num_test_docs = -1;
double GAMMA = 0.0;
double alpha = 0.0;
int vocab_size = 0;
gsl_rng *GSL_RNG;
int branching_factor;
int num_levels;
int num_nodes;
int num_paths;
int num_internal_nodes;
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

void readData(const char* fileName_Z, const char* fileName_beta, 
    const char* fileName_testcorpus, int* Z, double* beta, int* testcorpus) {
  ifstream file_Z (fileName_Z, ios::in);

  int doc = 0;
  for (string line; getline(file_Z, line); doc++) {
    vector<string> words = split(line, ' ');
    assert (words.size() == max_words);
    for (int i=0; i<max_words; i++) {
      Z[doc * max_words + i] = atoi(words[i].c_str());
    }
  }

  file_Z.close();

  ifstream file_beta (fileName_beta, ios::in);

  int path = 0;
  for (string line; getline(file_beta, line); path++) {
    vector<string> vals = split(line, ' ');
    assert (vals.size() == vocab_size);
    for (int v=0; v<vocab_size; v++) {
      beta[path * vocab_size + v] = atof(vals[v].c_str());
    }
  }

  file_beta.close();

  ifstream file_testcorpus (fileName_testcorpus, ios::in);

  doc = 0;
  for (string line; getline(file_testcorpus, line); doc++) {
    vector<string> words = split(line, ' ');
    for (int i=0; i<max_words; i++) {
      testcorpus[doc * max_words + i] = (i < words.size()) 
                                        ? atoi(words[i].c_str())
                                        : -1;
    }
  }

  file_testcorpus.close();
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

int indexOf(int value, unsigned int *arr, int size) {
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

void generate_doc(int* result, double* beta, int* Z, int num_words_in_doc) {
  int* counts = new int[num_nodes];
  memset(counts, 0, num_nodes * sizeof(int));

  for (int n=0; n<num_words_in_doc; n++) {
    int counts_start_idx = 1;
    int selected_node = 0;
    while(counts_start_idx < num_nodes - 1) {
      //cout << n << " " << counts_start_idx << "\n";
      double* multinomial_prob = new double[branching_factor];
      int num_zeros = 0;
      for (int i=0; i<branching_factor; i++) {
        if (counts[counts_start_idx + i] == 0) {
          num_zeros++;
        }
      }
      for (int i=0; i<branching_factor; i++) {
        multinomial_prob[i] = (counts[counts_start_idx + i] == 0) ? GAMMA 
                          / num_zeros 
                          : counts[counts_start_idx + i];
      }
      // sample from multinomial
      unsigned int *sample = new unsigned int[branching_factor];
      gsl_ran_multinomial(GSL_RNG, branching_factor, 1, multinomial_prob, sample);
      selected_node = counts_start_idx + indexOf(1, sample, branching_factor);
      counts[selected_node]++;
      counts_start_idx = counts_start_idx * branching_factor + 1;
      delete [] multinomial_prob;
      delete [] sample;
    }
    // sample from beta[selected_node]
    unsigned int* sample = new unsigned int[vocab_size];
    gsl_ran_multinomial(GSL_RNG, vocab_size, 1, beta + (selected_node  - num_internal_nodes) * vocab_size, sample);
    result[n] = indexOf(1, sample, vocab_size);
    delete [] sample;
  }
}

double estimate_likelihood_corpus(double* beta, int* Z, int* testcorpus) {
  int K = 1000;
  int num_words_in_doc = max_words / 2;
  // generate K documents from trained model
  int* generated_docs = new int[K * num_words_in_doc];
  #pragma omp parallel for
  for (int k=0; k<K; k++) {
    // cout << k << "\n";
    generate_doc(generated_docs + k * num_words_in_doc, beta, Z, num_words_in_doc);
    /*
    for (int n=0; n<num_words_in_doc; n++) {
      cout << generated_docs[k * num_words_in_doc + n] << " ";
    }
    cout << "\n";
    */
  }

  cout << "Generated docs.\n";

  // estimate multinomials over words for generated docs
  double* probs = new double[K * vocab_size];
  for (int k=0; k<K; k++) {
    for (int v=0; v<vocab_size; v++) {
      probs[k * vocab_size + v] = 0.1;
    }
  }
  for (int k=0; k<K; k++) {
    for (int n=0; n<num_words_in_doc; n++) {
      probs[k * vocab_size + generated_docs[k * num_words_in_doc + n]]++;
    }
  }
  for (int k=0; k<K; k++) {
    double sum = 0.0;
    for (int v=0; v<vocab_size; v++) {
      sum += probs[k * vocab_size + v];
    }
    for (int v=0; v<vocab_size; v++) {
      probs[k * vocab_size + v] = log(probs[k * vocab_size + v] / sum);
    }
  }

  cout << "Computed multinomials.\n";

  double testcorpus_log_prob = 0.0;

  for (int d=0; d<num_test_docs; d++) {
    double avg_prob = 0.0;
    for (int k=0; k<K; k++) {
      double doc_log_prob = 0.0;
      int num_words_in_curr_doc = 0;
      for (int n=0; n<max_words; n++) {
        if (testcorpus[d * max_words + n] != -1) {
          double curr_word_prob = 
              probs[k * vocab_size + testcorpus[d * max_words + n]];
          //cout << "Cur word : " << curr_word_prob << " " << log(curr_word_prob) 
          //      << "\n"; 
          doc_log_prob += probs[k * vocab_size + 
              testcorpus[d * max_words + n]];
          num_words_in_curr_doc++;
        } else {
          break;
        }
      }
      avg_prob += exp(doc_log_prob / num_words_in_curr_doc);
    }
    avg_prob /= K;
    // cout << "Avg prob : " << avg_prob << "\n";
    if (!isnan(avg_prob)) {
      testcorpus_log_prob += avg_prob;
    }
  }

  cout << "Test corpus avg prob : " << testcorpus_log_prob / num_test_docs << "\n";

  /*
  for (int k=0; k<K; k++) {
    for (int v=0; v<vocab_size; v++) {
      cout << fixed << setprecision(5) << probs[k * vocab_size + v] << " ";
    }
    cout << "\n";
  }

  double ll = 0.0;
  for (int d=0; d<num_doc_test; d++) {
    for (int k=0; k<K; k++) {
      ll += estimate_likelihood_doc(test_corpus[d], generated_docs[k]);
    }
  }
  ll /= K;
  return ll;
  */ 
}

int main(int argc, char* argv[]) {
  // parse commandline arguments
  if (argc != 12) {
    printf("Run as ./executable Z_file beta_file num_docs max_words alpha GAMMA"
            "num_levels branching_factor vocab_size test_corpus num_test_docs\n");
    exit(1);
  }

  //cout.precision(3);
  //cout << fixed;

  istringstream alpha_str(argv[5]);
  alpha_str >> alpha;
  istringstream gamma_str(argv[6]);
  gamma_str >> GAMMA;
  istringstream num_docs_str(argv[3]);
  num_docs_str >> num_docs;
  istringstream max_words_str(argv[4]);
  max_words_str >> max_words;
  istringstream num_levels_str(argv[7]);
  num_levels_str >> num_levels;
  istringstream branching_factor_str(argv[8]);
  branching_factor_str >> branching_factor;
  istringstream vocab_size_str(argv[9]);
  vocab_size_str >> vocab_size;
  istringstream num_test_docs_str(argv[11]);
  num_test_docs_str >> num_test_docs;

  num_nodes = (pow(branching_factor, num_levels) - 1) / (branching_factor - 1);
  num_paths = pow(branching_factor, num_levels - 1);
  num_internal_nodes = num_nodes - num_paths;

  // read data
  computeParams(argv[1]);
  assert (num_docs > 0);
  assert (max_words > 0);
  int *Z = new int[num_docs * max_words];
  double* beta = new double[num_paths * vocab_size];
  int *test_corpus = new int[num_test_docs * max_words];
  readData(argv[1], argv[2], argv[10], Z, beta, test_corpus);

  printf("Read %d docs.\n", num_docs);
  printf("Read %d words.\n", max_words);

  GSL_RNG = gsl_rng_alloc(gsl_rng_mt19937);
  
  /*
  // print data
  for (int d=0; d<num_docs; d++) {
    for (int n=0; n<max_words; n++) {
      cout << W[d*max_words + n] << " ";
    }
    cout << "\n";
  }
  */

  // int* opt_Z = new int[num_docs * max_words];
  // double *opt_beta = new double[num_paths]

  // run Gibbs sampler
  //Model best_model = runGibbsSampling(W);
  //runGibbsSampling(W, argv[4]);

  estimate_likelihood_corpus(beta, Z, test_corpus);

  gsl_rng_free(GSL_RNG);
  //delete &best_model;
  return 0;

  /*
  int* best_Z = best_model.getBestZ();

  for (int d=0; d<num_docs; d++) {
    for (int n=0; n<getNumWords(W + d * max_words); n++) {
      cout << best_Z[d * max_words + n] << " ";
    }
    cout << "\n";
  }


  return 0;
  */
}
