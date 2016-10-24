#include <iostream>
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

using namespace std;

int num_docs = -1;
int max_words = -1;
double GAMMA = 0.0;
double alpha = 0.0;
int vocab_size = 20;    // TODO : fix this
gsl_rng *GSL_RNG;

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
    for (int i=0; i<words_size; i++) {
      W[doc * max_words + i] = atoi(words[i].c_str());
    }
    //TODO: Add -1s
  }
}

class Model {
  double alpha;
  double GAMMA;
  int num_levels;
  int branching_factor;
  int num_paths;
  int num_nodes;
  int num_internal_nodes;
  float *beta;
  int *Z;
  int *countsPerDoc;
  

  public:
  
  Model(float alpha, float GAMMA, int num_levels, int branching_factor) {
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

    beta = new float[num_paths * vocab_size];
    Z = new int[num_docs * max_words];
    countsPerDoc = new int[num_docs * num_nodes];

    memset(Z, 0, num_docs * max_words * sizeof(int));

    for (int p=0; p<num_paths; p++) {
      for (int v=0; v<vocab_size; v++) {
        beta[p * vocab_size + v] = 1.0 / vocab_size;
      }
    }
  }

  void printDocCounts(int doc) {
    cout << "Doc : " << doc << "\n";
    for (int p=0; p<num_nodes; p++) {
      cout << countsPerDoc[doc * num_nodes + p] << "\t";
    }
    cout << "\n";
  }

  void computeDocCounts(int doc, int num_words_in_doc) {
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

  void normalize_distr(double *distr, int size) {
    double sum = 0.0;
    for (int i=0; i<size; i++) {
      sum += distr[i];
    }
    for (int i=0; i<size; i++) {
      distr[i] /= sum;
    }
  }

  void sample_beta_p() {
  
  }

  void sample_Zdn(int* W_d, int d, int n, int num_words_in_doc) {
    /*
    // remove current Z from counts
    int idx = Z[d * max_words + n] + num_internal_nodes;
    while (idx >= 0) {
      countsPerDoc[d * max_words + idx] --;
      idx = (idx - 1) / branching_factor;
    }

    // compute NCRP priors
    */
    // approximate prior
    double *multinomial_prob = new double[num_paths];
    /*
    cout << "counts per doc: \n";
    for (int p=0; p<num_paths; p++) {
      cout << countsPerDoc[d * num_nodes + num_internal_nodes + p] << " ";
    }
    cout << "\n";
    */
    for (int p=0; p<num_paths; p++) {
      /*
      cout << "p : " << p << " " << 
          ((double)countsPerDoc[d * num_nodes + num_internal_nodes + p] + GAMMA) /
          (num_words_in_doc + num_paths * GAMMA) << " " << 
          beta[p * vocab_size +  W_d[n]] << " " <<
          W_d[n] << "\n";
      */
      multinomial_prob[p] = 
          ((double)countsPerDoc[d * num_nodes + num_internal_nodes + p] + GAMMA) / 
          (num_words_in_doc + num_paths * GAMMA) *
          beta[p * vocab_size +  W_d[n]];
    }

    // normalize_distr(multinomial_prob, num_paths);
    /*
    for (int p=0; p<num_paths; p++) {
      cout << multinomial_prob[p] << " ";
    }
    cout << "\n";
    */
    unsigned int *sample = new unsigned int[num_paths];
    gsl_ran_multinomial(GSL_RNG, num_paths, 1, multinomial_prob, sample);
    Z[d * max_words + n] = indexOf(1, sample, num_paths);

    cout << "Sampled " << d << " " << n << " " << Z[d * max_words + n] << "\n";
  }
};

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

Model runGibbsSampling(int* W) {
  Model *model = new Model(alpha, GAMMA, 4, 3);

  const int NUM_STARTS = 1;
  const int MAX_ITER = 1;

  for (int start = 0; start < NUM_STARTS; start++) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
      // sample Z
      for (int d = 0; d < num_docs; d++) {
        int num_words_in_doc = getNumWords(W + d * max_words);
        cout << "num_words_in_doc : " << num_words_in_doc << "\n";
        model->computeDocCounts(d, num_words_in_doc);
        for (int n=0; n < num_words_in_doc; n++) {
          model->sample_Zdn(W + d * max_words, d, n, num_words_in_doc);
        }
      }

      // sample beta
      model->sample_beta_p();
    }
  }
}

int main(int argc, char* argv[]) {
  // parse commandline arguments
  if (argc != 4) {
    printf("Run as ./executable corpus alpha GAMMA\n");
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

  // run Gibbs sampler
  runGibbsSampling(W);

  return 0;
}
