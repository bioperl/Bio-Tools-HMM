#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include "hmm.h"

static unsigned int hmm_seed;
static void alphaT(HMM *, double *, char *, int);
static void beta1(HMM *, double *, char *, int);
static double * xi(HMM *);
static void alpha_mat(HMM *, double **, char *, int);
static void beta_mat(HMM *, double **, char *, int);
static void HMM_fatal(char *);
static double sumLogProbs(double *, int);
static double sumLogProb(double, double);

/*
   sumLogProbs adds up the log probabilities stored in the array 
   logprobs of size sz by dividing every items with the largest
   item in the array. This prevents underflow such that calculations
   for long sequence are allowed. The function returns the log of the
   sum of log probabilities.
   log(a_0+a_1+...+a_n) = log(a_max)+log(a_0/a_max+a_1/a_max+...+a_n/a_max)
 */  
static double
sumLogProbs(double * logprobs, int sz)
{
   double max = 0.0;
   double p = 0.0;
   int i;
   for (i = 0; i < sz; ++i) 
      if (i == 0 || logprobs[i] > max)
         max = logprobs[i];
   if (max == log(0))
      return max;
   for (i = 0; i < sz; ++i) 
      p += exp(logprobs[i] - max);
   return max + log(p); 
}

/*
   sumLogProb is similar to sumLogProbs except that it is adding up 
   only two log probabilities p1 and p2. It returns the log of the sum
   of p1 and p2.
 */
static double
sumLogProb(double p1, double p2)
{
   if (p1 == log(0) && p2 == log(0))
      return p1;
   if (p1 > p2)
      return p1 + log(1 + exp(p2 - p1));
   else
      return p2 + log(1 + exp(p1 - p2));
}

/*
  HMM_fatal prints the string s to STDERR and then exit
 */
static void
HMM_fatal(char * s)
{
    fprintf(stderr, s);
    exit(-1);
}

/*
   HMM_new initializes an HMM object with string of all possible 
   symbols and string of all possible states. The other parameters
   like initial probabilities (HMM->init), state transition matrix
   (HMM->a_mat) and emission matrix (HMM->e_mat) are initialized
   randomly such that it is ready for training.
 */ 
HMM *
HMM_new(char * symbols, char * states)
{
   HMM * hmm;
   double sum, random;
   int i, j;

   hmm_seed = time(NULL);
   hmm = (HMM *) malloc(sizeof(HMM));
   if (hmm == NULL) 
      HMM_fatal("Can't allocate memory for HMM, die!\n");
   hmm->symbols = symbols;
   hmm->states = states;
   hmm->M = strlen(symbols);
   hmm->N = strlen(states);
   for (i = 0; i < hmm->M; ++i) 
      hmm->omap[symbols[i]] = i;   
   for (i = 0; i < hmm->N; ++i) 
      hmm->smap[states[i]] = i;   
   hmm->init = (double *) malloc(hmm->N*sizeof(double));
   if (hmm->init == NULL) 
      HMM_fatal("Can't allocate memory for init array of HMM, die!\n");
/* initialize the initial state array */
   sum = 0.0;
   for (j = 0; j < hmm->N; ++j) {
      srand(hmm_seed++);
      random = (double) rand();
      hmm->init[j] = random;
      sum += random;
   }
   for (j = 0; j < hmm->N; ++j) 
      hmm->init[j] /= sum;
   hmm->e_mat = (double **) malloc(hmm->N*sizeof(double *));
   if (hmm->e_mat == NULL) 
      HMM_fatal("Can't allocate memory for emission matrix of HMM, die!\n");
   for (i = 0; i < hmm->N; ++i) {
      hmm->e_mat[i] = (double *) malloc(hmm->M*sizeof(double));
      if (hmm->e_mat[i] == NULL)
         HMM_fatal("Can't allocate memory for emission matrix of HMM, die!\n");
   }
   hmm->a_mat = (double **) malloc(hmm->N*sizeof(double *));
   if (hmm->a_mat == NULL) 
      HMM_fatal("Can't allocate memory for transition matrix of HMM, die!\n");
   for (i = 0; i < hmm->N; ++i) {
      hmm->a_mat[i] = (double *) malloc(hmm->N*sizeof(double));
      if (hmm->a_mat[i] == NULL)
         HMM_fatal("Can't allocate memory for transition matrix of HMM, die!\n");

/* randomize the transition matrix */
      sum = 0.0;
      for (j = 0; j < hmm->N; ++j) {
         srand(hmm_seed++);
         random = (double) rand();
         hmm->a_mat[i][j] = random;
         sum += random;
      }
      for (j = 0; j < hmm->N; ++j) 
         hmm->a_mat[i][j] /= sum;
   }
   hmm->e_mat = (double **) malloc(hmm->N*sizeof(double *));
   if (hmm->e_mat == NULL) 
      HMM_fatal("Can't allocate memory for emission matrix of HMM, die!\n");
   for (i = 0; i < hmm->N; ++i) {
      hmm->e_mat[i] = (double *) malloc(hmm->M*sizeof(double));
      if (hmm->e_mat[i] == NULL)
         HMM_fatal("Can't allocate memory for emission matrix of HMM, die!\n");
/* randomize the emission matrix */
      sum = 0.0;
      for (j = 0; j < hmm->M; ++j) {
         srand(hmm_seed++);
         random = (double) rand();
         hmm->e_mat[i][j] = random;
         sum += random;
      }
      for (j = 0; j < hmm->M; ++j) 
         hmm->e_mat[i][j] /= sum;
   }
   return hmm;
}

/*
   init_HMM is similar to HMM_new except that it allows you to initialize
   the HMM with your own initial probabilities (init), your own state
   transition matrix (a_mat) and your own emission matrix (e_mat).
 */
HMM *
init_HMM(char * symbols, char * states, double * init, double ** a_mat, double ** e_mat)
{
   HMM * hmm;
   int i;

   hmm = (HMM *) calloc(1, sizeof(HMM));
   if (hmm == NULL) 
      HMM_fatal("Can't allocate memory for HMM, die!\n");
   hmm->symbols = symbols;
   hmm->states = states;
   hmm->M = strlen(symbols);
   hmm->N = strlen(states);
   for (i = 0; i < hmm->M; ++i) 
      hmm->omap[symbols[i]] = i;   
   for (i = 0; i < hmm->N; ++i) 
      hmm->smap[states[i]] = i;   
   hmm->init = init;
   hmm->a_mat = a_mat;
   hmm->e_mat = e_mat;
   return hmm;
}

/*
   HMM_get_init_entry returns the value of a cell in the initial
   probability array based on the state supplied.
   HMM_get_init_entry is written such that XS can access the C array.
 */
double
HMM_get_init_entry(HMM * hmm, char * state)
{
   return hmm->init[hmm->smap[state[0]]];
}

/*
   HMM_set_init_entry sets the value of a cell in the initial
   probability array based on the val supplied.
   HMM_set_init_entry is written such that XS can modify the C array.
 */
void
HMM_set_init_entry(HMM * hmm, char * state, double val)
{
   hmm->init[hmm->smap[state[0]]] = val;
}

/*
   HMM_get_a_entry returns the value of a cell in the state transition
   matrix based on the from-state (state1) and the to-state (state2)
   supplied.
   HMM_get_a_entry is written such that XS can access the C matrix.
 */
double
HMM_get_a_entry(HMM * hmm, char * state1, char * state2)
{
   return hmm->a_mat[hmm->smap[state1[0]]][hmm->smap[state2[0]]];
}

/*
   HMM_set_a_entry sets the value of a cell in the state transition
   matrix to val based on the from-state (state1) and the to-state 
   (state2) supplied.
   HMM_set_a_entry is written such that XS can modify the C matrix.
 */
void
HMM_set_a_entry(HMM * hmm, char * state1, char * state2, double val)
{
   hmm->a_mat[hmm->smap[state1[0]]][hmm->smap[state2[0]]] = val;
}

/*
   HMM_get_e_entry returns the value of a cell in the emission
   matrix based on the from-state (state1) and the to-state (state2)
   supplied.
   HMM_get_e_entry is written such that XS can access the C matrix.
 */
double
HMM_get_e_entry(HMM * hmm, char * state, char * symbol)
{
   return hmm->e_mat[hmm->smap[state[0]]][hmm->omap[symbol[0]]];
}

/*
   HMM_set_e_entry sets the value of a cell in the emission matrix
   to val based on the from-state (state1) and the to-state (state2)
   supplied.
   HMM_set_e_entry is written such that XS can modify the C matrix.
 */
void
HMM_set_e_entry(HMM * hmm, char * state, char * symbol, double val)
{
   hmm->e_mat[hmm->smap[state[0]]][hmm->omap[symbol[0]]] = val;
}

/*
   omap takes a symbol character and return its index in the state
   transition matrix.
 */
void
omap(HMM * hmm, char * c, int L)
{
   int i;

   for (i = 0; i < L; ++i) 
      c[i] = hmm->omap[c[i]];
}

/*
   smap takes a state character and return its index in the emission
   matrix.
 */
void
smap(HMM * hmm, char * c, int L)
{
   int i;

   for (i = 0; i < L; ++i) 
      c[i] = hmm->smap[c[i]];
}

/*
   alpha function for HMM. It computes the alpha function value
   at the time T which is the length of the observation sequence
   obs for every state. Note that alpha_vec needs to be malloc'd
   before we supply it to alphaT.
   Note that the return vector is in logarithmic form. 
 */
static void
alphaT(HMM * hmm, double * alpha_vec, char * obs, int T)
{
   int N = hmm->N;
   double alpha_vec_prev[N];
   double * init = hmm->init;
   double ** a_mat = hmm->a_mat;
   double ** e_mat = hmm->e_mat;
   int i, j, k;

   if (T <= 0) 
      HMM_fatal("Nonpositive T, die!\n");
   for (i  = 0; i < N; ++i)  
      alpha_vec[i] = log(init[i]) + log(e_mat[i][obs[0]]);
   if (T == 1) 
      return;
   if (T > 1) {
      for (i = 1; i < T; ++i) {
         for (j = 0; j < N; ++j) {
            alpha_vec_prev[j] = alpha_vec[j];
            alpha_vec[j] = log(0);
         }
        
         for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k)
               alpha_vec[j] = sumLogProb(alpha_vec[j], alpha_vec_prev[k] + log(a_mat[k][j]));
            alpha_vec[j] = alpha_vec[j] + log(e_mat[j][obs[i]]);
         }
      }
    }
    return; 
}

/* 
   Palpha computes the probability of an observation sequence
   using the alpha function.
 */
double
Palpha(HMM * hmm, char * obs, int T)
{
   double P = 0.0;
   int i;
   double * alpha_vec;

   alpha_vec = (double *) calloc(hmm->N, sizeof(double));
   if (alpha_vec == NULL) 
      HMM_fatal("Can't allocate memory for alpha vector, die!\n");
   alphaT(hmm, alpha_vec, obs, T);
   P = sumLogProbs(alpha_vec, hmm->N);
   free(alpha_vec);
   return P;
}


/*
   beta function for HMM. It computes the beta function value
   at the time 1 for every state. Note that beta_vec should
   be allocated memory thru malloc before we use it.
   Note that the return vector is in logarithmic form. 
 */
static void
beta1(HMM * hmm, double * beta_vec, char * obs, int T)
{
   int N = hmm->N;
   double beta_vec_next[N];
   double * init = hmm->init;
   double ** a_mat = hmm->a_mat;
   double ** e_mat = hmm->e_mat;
   int i, j, k;

   if (T <= 0) 
      HMM_fatal("Nonpositive T, die!\n");
   for (i  = 0; i < N; ++i)  
      beta_vec[i] = 0;
   if (T == 1) 
      return;
   if (T > 1) {
      for (i = T - 1; i >= 1; --i) {
         for (j = 0; j < N; ++j) {
            beta_vec_next[j] = beta_vec[j];
            beta_vec[j] = log(0);
         }
        
         for (j = 0; j < N; ++j) {
            for (k = 0; k < N; ++k)
               beta_vec[j] = sumLogProb(beta_vec[j], beta_vec_next[k] + log(a_mat[j][k]) + log(e_mat[k][obs[i]]));
         }
      }
    }
    return; 
}

/* 
   Pbeta computes the probability of an observation sequence
   using the beta function.
 */
double
Pbeta(HMM * hmm, char * obs, int T)
{
   double P = log(0);
   int i;
   double * beta_vec;

   beta_vec = (double *) calloc(hmm->N, sizeof(double));
   if (beta_vec == NULL) 
      HMM_fatal("Can't allocate memory for beta vector, die!\n");
   beta1(hmm, beta_vec, obs, T);
   for (i = 0; i < hmm->N; ++i) {
      P = sumLogProb(P, log(hmm->init[i]) + log(hmm->e_mat[i][obs[0]]) + beta_vec[i]);
   }
   free(beta_vec);
   return P;
}

/*
   alpha_mat computes the values for the alpha function in the TxN space 
   where T is the length of the observation sequence and N is the number
   of states. Note that the alpha matrix has to be malloc'd before we
   supply it to alpha_mat. The computed values will be stored in
   alpha.
 */  
static void
alpha_mat(HMM * hmm, double ** alpha, char * obs, int T)
{
   int i, j, t;
   int N = hmm->N;
   double ** a_mat = hmm->a_mat;
   double ** e_mat = hmm->e_mat;
   double * init = hmm->init;

   for (i = 0; i < N; ++i) 
      alpha[i][0] = log(init[i]) + log(e_mat[i][obs[0]]);
   for (t = 1; t < T; ++t) {
      for (i = 0; i < N; ++i) 
         alpha[i][t] = log(0);
      for (i = 0; i < N; ++i) {
         for (j = 0; j < N; ++j) 
            alpha[i][t] = sumLogProb(alpha[i][t], alpha[j][t-1] + log(a_mat[j][i]));
         alpha[i][t] += log(e_mat[i][obs[t]]);
      }
   }
}

/*
   beta_mat computes the values for the alpha function in the TxN space 
   where T is the length of the observation sequence and N is the number
   of states. Note that the beta matrix has to be malloc'd before we
   supply it to beta_mat. The computed values will be stored in
   beta.
 */  
static void
beta_mat(HMM * hmm, double ** beta, char * obs, int T)
{
   int i, j, t;
   int N = hmm->N;
   double ** a_mat = hmm->a_mat;
   double ** e_mat = hmm->e_mat;
   double * init = hmm->init;

   for (i = 0; i < N; ++i) 
      beta[i][T-1] = 0;
   for (t = T-2; t >= 0; --t) {
      for (i = 0; i < N; ++i) 
         beta[i][t] = log(0);
      for (i = 0; i < N; ++i) 
         for (j = 0; j < N; ++j) 
            beta[i][t] = sumLogProb(beta[i][t], beta[j][t+1] + log(a_mat[i][j]) + log(e_mat[j][obs[t+1]]));
   }
}

/*
   baum_welch takes an array of observation sequence strings (obs) and 
   the number of strings in the array (L) to estimate the parameters
   of an HMM using a special case of Expectation Maximization called
   Baum-Welch algorithm. baum_welch doesn't return anything but upon
   its completion, it will set the init array, a_mat matrix, e_mat
   matrix with parameters that maximize the chance of seeing the 
   observation sequences you supplied.
 */
void
baum_welch(HMM * hmm, char ** obs, int * T, int L)
{
   int i, j, l, t;
   int maxL = -1; /* length of longest training sequence */
   double P[L];
   double newP[L];
   double S, newS, diff;
   int N = hmm->N, M = hmm->M;
   double ** alpha;
   double ** beta;
   double A[N][N];
   double E[N][M];
   double init[N];
   double a_mat[N][N];
   double e_mat[N][M];

   for (l = 0; l < L; ++l) {
      P[l] = Palpha(hmm, obs[l], T[l]);
      if (maxL == -1) 
         maxL = T[l];
      else if (T[l] > maxL)
         maxL = T[l];
   }

   alpha = (double **) malloc(hmm->N*sizeof(double *));
   if (alpha == NULL) 
      HMM_fatal("Can't allocate memory for alpha matrix!\n");
   for (i = 0; i < N; ++i) {
      alpha[i] = (double *) malloc(maxL*sizeof(double)); 
      if (alpha[i] == NULL) 
         HMM_fatal("Can't allocate memory for rows of alpha matrix!\n");
   }
   beta = (double **) malloc(hmm->N*sizeof(double *));
   if (beta == NULL) 
      HMM_fatal("Can't allocate memory for beta matrix!\n");
   for (i = 0; i < N; ++i) {
      beta[i] = (double *) malloc(maxL*sizeof(double)); 
      if (beta[i] == NULL) 
         HMM_fatal("Can't allocate memory for rows of beta matrix!\n");
   }

   do { /* do...while loop */

/* initialize the matrices */
      for (i = 0; i < N; ++i)
         init[i] = 0.0;
      for (i = 0; i < N; ++i) 
         for (j = 0; j < N; ++j) 
            A[i][j] = 0.0;
      for (i = 0; i < N; ++i)
         for (j = 0; j < M; ++j)
            E[i][j] = 0.0;

/* use all training sequences to train */
      for (l = 0; l < L; ++l) {
         alpha_mat(hmm, alpha, obs[l], T[l]);
         beta_mat(hmm, beta, obs[l], T[l]);

/* estimate new initial state probability */
         for (i = 0; i < N; ++i) 
            for (j = 0; j < N; ++j) 
               init[i] += exp(alpha[i][0] + log(hmm->a_mat[i][j]) + log(hmm->e_mat[j][obs[l][1]]) + beta[j][1] - P[l]);

/* estimate A matrix */
         for (i = 0; i < N; ++i) 
            for (j = 0; j < N; ++j) 
               for (t = 0; t < T[l]-1; ++t) 
                  A[i][j] += exp(alpha[i][t] + log(hmm->a_mat[i][j]) + log(hmm->e_mat[j][obs[l][t+1]]) + beta[j][t+1] - P[l]);

/* Estimate E matrix */
         for (i = 0; i < N; ++i)
            for (t = 0; t < T[l]; ++t)
               E[i][obs[l][t]] += exp(alpha[i][t] + beta[i][t] - P[l]);
      }

/* Estimate init */
   for (i = 0; i < N; ++i) 
      hmm->init[i] = init[i] / (double) L;

/* Estimate a_mat */
   for (i = 0; i < N; ++i) 
      for (j = 0; j < N; ++j)
         a_mat[i][j] = hmm->a_mat[i][j];

   for (i = 0; i < N; ++i)
      for (j = 0; j < N; ++j) {
         hmm->a_mat[i][j] = 0.0;
         for (t = 0; t < N; ++t)
            hmm->a_mat[i][j] += A[i][t];
         hmm->a_mat[i][j] = A[i][j] / hmm->a_mat[i][j];
      }

/* Estimate e_mat */
   for (i = 0; i < N; ++i)
      for (j = 0; j < M; ++j)
         e_mat[i][j] = hmm->e_mat[i][j];

   for (i = 0; i < N; ++i)
      for (j = 0; j < M; ++j) {
         hmm->e_mat[i][j] = 0.0;
         for (t = 0; t < M; ++t)
            hmm->e_mat[i][j] += E[i][t];
         hmm->e_mat[i][j] = E[i][j] / hmm->e_mat[i][j];
      }

/*
// print parameter estimates for debugging
   for (i = 0; i < N; ++i)
      printf("%.2f\t", hmm->init[i]);
   printf("\n");
   for (i = 0; i < N; ++i) {
      for (j = 0; j < N; ++j)
         printf("%.2f\t", hmm->a_mat[i][j]);
      printf("\n");
   }
   for (i = 0; i < N; ++i) {
      for (j = 0; j < M; ++j)
         printf("%.2f\t", hmm->e_mat[i][j]);
      printf("\n");
   }
*/

/* find new P */
   S = 0.0;
   newS = 0.0;
   for (l = 0; l < L; ++l) {
      S += P[l];
      newP[l] = Palpha(hmm, obs[l], T[l]);
      newS += newP[l];
      P[l] = newP[l];
   }
      diff = newS - S;
//printf("%g(%g)\t%g(%g)\t%g\n", newS, log(newS), S, log(S), diff);
      if (diff < 0 && fabs(diff) >= UPPER_TOL) 
         HMM_fatal("S should be monotonic increasing!\n");
   }
   while (diff >= UPPER_TOL);

   for (i = 0; i < N; ++i) {
      free(alpha[i]);
      free(beta[i]);
   }
   free(alpha);
   free(beta);
}

/*
   viterbi algorithm takes a observation sequence (obs) and the HMM model
   (hmm) and then returns the hidden state sequence (hss) that maximizes
   the probability of seeing obs. The hss argument supplied should be 
   allocated to the same amount of memory as obs.
 */
void
viterbi(HMM * hmm, char * hss, char * obs, int T)
{
   double delta_vec[hmm->N];
   double delta_vec_prev[hmm->N];
   int N = hmm->N;
   double * init = hmm->init;
   double ** a_mat = hmm->a_mat;
   double ** e_mat = hmm->e_mat;
   int i, j, k;
   double max;
   int max_i = -1;
   int phi_vec[T];
   int phi_mat[T][N];

   if (T <= 0) 
      HMM_fatal("Nonpositive T, die!\n");
   for (i = 0; i < N; ++i) {
//      delta_vec[i] = init[i]*e_mat[i][obs[0] - '1'];
//      delta_vec[i] = i == 0 ? log(0.5) : -HUGE_VAL;
      delta_vec[i] = log(init[i]);
      if (max_i < 0) {
         max_i = i;
         max = delta_vec[i];
      }
      else if (delta_vec[i] > max) {
         max_i = i;
         max = delta_vec[i];
      }
   }
   if (T == 1) {
      phi_vec[0] = max_i;
      hss[0] = hmm->states[phi_vec[0]];
      hss[1] = '\0';
      return;
   }
   if (T > 1) {
      for (i = 0; i < T; ++i) {
         for (j = 0; j < N; ++j) {
            delta_vec_prev[j] = delta_vec[j];
            delta_vec[j] = 0;
         }
         for (j = 0; j < N; ++j) {
            max_i = -1;   
            for (k = 0; k < N; ++k) {
               double val = delta_vec_prev[k] + log(a_mat[k][j]);
               if (max_i < 0) {
                  max_i = k;
                  max = val; 
               }
               else if (val > max) {
                  max_i = k;
                  max = val;
               }
            }
            delta_vec[j] = max + log(e_mat[j][obs[i]]);
            phi_mat[i][j] = max_i;
         }
      }
   }
   max_i = -1;
   for (i = 0; i < N; ++i) {
      if (max_i < 0) {
         max_i = i;
         max = delta_vec[i];
      }
      else if (delta_vec[i] > max) {
         max_i = i;
         max = delta_vec[i];
      }
   }
   phi_vec[T-1] = max_i;
// traceback
   for (i = T-2; i >= 0; --i) 
      phi_vec[i] = phi_mat[i+1][phi_vec[i+1]];
   for (i = 0; i < T; ++i) 
      hss[i] = hmm->states[phi_vec[i]];
   hss[T] = '\0';
   return;
}

/* 
   state_est takes an array of observation sequence strings (obs), 
   an array of the corresponding hidden state sequence strings (sta)
   and the number of strings (L) to find the HMM parameters that
   maximize the chance of seeing the supplied observation sequences
   and their corresponding hidden state sequences. At the completion
   of this function, init, a_mat and e_mat will be set to the 
   proper values.
 */
void
state_est(HMM * hmm, char ** obs, char ** sta, int * T, int L)
{
   int N = hmm->N, M = hmm->M;
   int i, j, k, l;
   double * init = hmm->init;
   double ** a_mat = hmm->a_mat;
   double ** e_mat = hmm->e_mat;
   int pi[N];
   int A[N][N];
   int E[N][M];

/* initialize the matrices */
   for (i = 0; i < N; ++i) 
      pi[i] = 0;
   for (i = 0; i < N; ++i)
      for (j = 0; j < N; ++j) {
         A[i][j] = 0;
         a_mat[i][j] = 0.0;
      }
   for (i = 0; i < N; ++i)
      for (j = 0; j < M; ++j) {
         E[i][j] = 0;
         e_mat[i][j] = 0.0;
      }

/* count occurrences */
   for (l = 0; l < L; ++l) {
      ++pi[sta[l][0]];

      for (i = 0; i < T[l] - 1; ++i)
         ++A[sta[l][i]][sta[l][i+1]];
      for (i = 0; i < T[l]; ++i)
         ++E[sta[l][i]][obs[l][i]];
   }

/* Estimate initial state probability */
   for (i = 0; i < N; ++i) 
      init[i] = (double) pi[i] / (double) L;

/* Estimate state transition probability matrix */
   for (i = 0; i < N; ++i)
      for (j = 0; j < N; ++j) {
         for (k = 0; k < N; ++k)
            a_mat[i][j] += A[i][k];
         a_mat[i][j] = A[i][j] / a_mat[i][j];
      }

/* Estimate Symbol Emission matrix */
   for (i = 0; i < N; ++i)
      for (j = 0; j < M; ++j) {
         for (k = 0; k < M; ++k)
            e_mat[i][j] += E[i][k];
         e_mat[i][j] = E[i][j] / e_mat[i][j];
      }
}
