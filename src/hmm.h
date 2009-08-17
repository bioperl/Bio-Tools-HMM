#ifndef _HMM_H_
#define _HMM_H_

typedef struct HMM
{
   int M;
   int N;
   char * symbols;
   unsigned char omap[256];
   char * states;
   unsigned char smap[256];
   double * init;
   double ** a_mat;
   double ** e_mat;
} HMM;

#define UPPER_TOL 0.000001
#define LOWER_TOL 1e-100

HMM * init_HMM(char *, char *, double *, double **, double **); 
HMM * HMM_new(char *, char *);
double HMM_get_init_entry(HMM *, char *);
void HMM_set_init_entry(HMM *, char *, double);
double HMM_get_a_entry(HMM *, char *, char *);
void HMM_set_a_entry(HMM *, char *, char *, double);
double HMM_get_e_entry(HMM *, char *, char *);
void HMM_set_e_entry(HMM *, char *, char *, double);
void omap(HMM *, char *, int);
void smap(HMM *, char *, int);
double Palpha(HMM *, char *, int);
double Pbeta(HMM *, char *, int);
void viterbi(HMM *, char *, char *, int);
void state_est(HMM *, char **, char **, int *, int);
void baum_welch(HMM *, char **, int *, int);

#endif 
