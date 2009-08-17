
#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

#include "hmm.h"

MODULE = Bio::Ext::HMM PACKAGE = Bio::Ext::HMM

void
HMM_statistical_training(class, hmm, obs, hs)
        char * class
        HMM * hmm
        SV * obs
        SV * hs
        CODE:
        AV * obs_av = (AV *) SvRV(obs);
        AV * hs_av = (AV *) SvRV(hs);
        int i;
        int avlen = av_len(obs_av);
        char ** obs_ar = (char **) malloc(avlen*sizeof(char *));
        char ** hs_ar = (char **) malloc(avlen*sizeof(char *));
        int * obs_len = (int *) malloc(avlen*sizeof(int));
        if (obs_ar == NULL || hs_ar == NULL)
           croak("Can't allocate memory for observation and/or state arrays!\n");
        if (obs_len == NULL)
           croak("Can't allocate memory for observation length array!\n");
        for (i = 0; i < avlen; ++i) {
           obs_ar[i] = (char *) SvPV(*av_fetch(obs_av, i, 0), PL_na);   
           obs_len[i] = strlen(obs_ar[i]);
           obs_ar[i] = (char *) malloc((obs_len[i]+1)*sizeof(char));
           if (obs_ar[i] == NULL)
              croak("Can't allocate memory for observation array!\n");
           strcpy(obs_ar[i], (char *) SvPV(*av_fetch(obs_av, i, 0), PL_na));
           hs_ar[i] = (char *) malloc((obs_len[i]+1)*sizeof(char));
           if (hs_ar[i] == NULL)
              croak("Can't allocate memory for state array!\n");
           strcpy(hs_ar[i], (char *) SvPV(*av_fetch(hs_av, i, 0), PL_na));
           omap(hmm, obs_ar[i], obs_len[i]);   
           smap(hmm, hs_ar[i], obs_len[i]);   
        }
        state_est(hmm, obs_ar, hs_ar, obs_len, avlen);
        for (i = 0; i < avlen; ++i) { 
           free(obs_ar[i]);
           free(hs_ar[i]);
        }
        free(obs_ar);
        free(hs_ar);
        free(obs_len);

double
HMM_likelihood(class, hmm, seq)
        char * class
        HMM * hmm
        char * seq
        CODE:
        int T = strlen(seq);
        char obs[T+1];
        strcpy(obs, seq);
        omap(hmm, obs, T);
        RETVAL = Palpha(hmm, obs, T);
        OUTPUT:
        RETVAL

void
HMM_baum_welch_training(class, hmm, obs)
        char * class
        HMM * hmm
        SV * obs
        PPCODE:
        AV * obs_av = (AV *) SvRV(obs);
        int i;
        int avlen = av_len(obs_av);
        char ** obs_ar = (char **) malloc(avlen*sizeof(char *));
        int * obs_len = (int *) malloc(avlen*sizeof(int));
        if (obs_ar == NULL)
           croak("Can't allocate memory for observation arrays!\n");
        if (obs_len == NULL)
           croak("Can't allocate memory for observation length array!\n");
        for (i = 0; i < avlen; ++i) {
           obs_ar[i] = (char *) SvPV(*av_fetch(obs_av, i, 0), PL_na);
           obs_len[i] = strlen(obs_ar[i]);
           obs_ar[i] = (char *) malloc((obs_len[i]+1)*sizeof(char));
           if (obs_ar[i] == NULL)
              croak("Can't allocate memory for observation array!\n");
           strcpy(obs_ar[i], (char *) SvPV(*av_fetch(obs_av, i, 0), PL_na));
           omap(hmm, obs_ar[i], obs_len[i]);   
        }
        baum_welch(hmm, obs_ar, obs_len, avlen);
        for (i = 0; i < avlen; ++i) 
           free(obs_ar[i]);
        free(obs_ar);

SV *
HMM_viterbi(class, hmm, seq)
        char * class
        HMM * hmm
        char * seq	 
        PPCODE:
        SV * sv;
        int T = strlen(seq);
        char * hss = (char *) malloc((T+1)*sizeof(char));
        char obs[T+1];
        if (hss == NULL)
           croak("Can't allocate memory for hidden state sequence!\n");
        strcpy(obs, seq);
        omap(hmm, obs, T);
        viterbi(hmm, hss, obs, T);
        sv = newSVpv(hss, strlen(hss));
        free(hss);
        PUSHs(sv_2mortal(sv));
        

MODULE = Bio::Ext::HMM PACKAGE = Bio::Ext::HMM::HMM

HMM *
new(class, symbols, states)
        char * class
        char * symbols
        char * states
        CODE:
        RETVAL = HMM_new(symbols, states);
        OUTPUT:
        RETVAL

double
get_init_entry(class, hmm, state)
        char * class
        HMM * hmm
        char * state
        CODE:
        RETVAL = HMM_get_init_entry(hmm, state);
        OUTPUT:
        RETVAL

void
set_init_entry(class, hmm, state, val)
        char * class
        HMM * hmm
        char * state
        double val
        CODE:
        HMM_set_init_entry(hmm, state, val);

double
get_a_entry(class, hmm, state1, state2)
        char * class
        HMM * hmm
        char * state1
        char * state2
        CODE:
        RETVAL = HMM_get_a_entry(hmm, state1, state2);
        OUTPUT:
        RETVAL

void
set_a_entry(class, hmm, state1, state2, val)
        char * class
        HMM * hmm
        char * state1
        char * state2
        double val
        CODE:
        HMM_set_a_entry(hmm, state1, state2, val);

double
get_e_entry(class, hmm, state, symbol)
        char * class
        HMM * hmm
        char * state
        char * symbol 
        CODE:
        RETVAL = HMM_get_e_entry(hmm, state, symbol);
        OUTPUT:
        RETVAL

void
set_e_entry(class, hmm, state, symbol, val)
        char * class
        HMM * hmm
        char * state
        char * symbol
        double val 
        CODE:
        HMM_set_e_entry(hmm, state, symbol, val);

void
DESTROY(obj)
        HMM * obj
        PPCODE:
        int i;
        free(obj->init);
        for (i = 0; i < obj->N; ++i)
           free(obj->a_mat[i]);
        free(obj->a_mat);
        for (i = 0; i < obj->N; ++i)
           free(obj->e_mat[i]);
        free(obj->e_mat);

