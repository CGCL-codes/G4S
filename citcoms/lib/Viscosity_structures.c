/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/* Functions relating to the determination of viscosity field either
   as a function of the run, as an initial condition or as specified from
   a previous file */


#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"


void myerror(struct All_variables *,char *);

void allocate_visc_vars(struct All_variables *E);
void read_visc_layer_file(struct All_variables *E);
void read_visc_param_from_file(struct All_variables *E,
                               const char *param, float *var,
                               FILE *fp);
static void apply_low_visc_wedge_channel(struct All_variables *E, float **evisc);
static void low_viscosity_channel_factor(struct All_variables *E, float *F);
static void low_viscosity_wedge_factor(struct All_variables *E, float *F);
void parallel_process_termination();
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
#include "anisotropic_viscosity.h"
#endif


void viscosity_system_input(struct All_variables *E)
{
    int m=E->parallel.me;
    int i;


    input_boolean("VISC_UPDATE",&(E->viscosity.update_allowed),"on",m);

    input_boolean("visc_layer_control",&(E->viscosity.layer_control),"off",m);
    input_string("visc_layer_file", E->viscosity.layer_file,"visc.dat",m);


    input_int("allow_anisotropic_viscosity",&(E->viscosity.allow_anisotropic_viscosity),"0",m);
#ifndef CITCOM_ALLOW_ANISOTROPIC_VISC 
    if(E->viscosity.allow_anisotropic_viscosity){ /* error */
      fprintf(stderr,"error: allow_anisotropic_viscosity is not zero, but code not compiled with CITCOM_ALLOW_ANISOTROPIC_VISC\n");
      parallel_process_termination();
    }
#else
    if(E->viscosity.allow_anisotropic_viscosity){ /* read additional
						     parameters for
						     anisotropic
						     viscosity */
      input_int("anisotropic_init",&(E->viscosity.anisotropic_init),"0",m); /* 0: isotropic
									       1: random
									       2: read in director orientation
									       and log10(eta_s/eta) 
									       3: align with velocity, use ani_vis2_factor for eta_s/eta
									       4: align with ISA, use ani_vis2_factor for eta_s/eta
									       5: align mixed depending on deformation state, use ani_vis2_factor for eta_s/eta
									       6: use radially aligned director and taper eta_s/eta from base (1) to top of layer (ani_vis2_factor)
									       
									    */
      input_string("anisotropic_init_dir",(E->viscosity.anisotropic_init_dir),"",m); /* directory
											for
											ggrd
											type
											init */
      input_int("anivisc_layer",&(E->viscosity.anivisc_layer),"1",m); /* >0: assign to layers on top of anivisc_layer
									 <0: assign to layer = anivisc_layer
								      */
      if((E->viscosity.anisotropic_init == 6) && (E->viscosity.anivisc_layer >= 0))
	myerror(E,"anisotropic init mode 6 requires selection of layer where anisotropy applies");
      
      input_boolean("anivisc_start_from_iso",
		    &(E->viscosity.anivisc_start_from_iso),"on",m); /* start
								       from
								       isotropic
								       solution? */
      if(!E->viscosity.anivisc_start_from_iso)
	if(E->viscosity.anisotropic_init == 3){
	  if(E->parallel.me == 0)fprintf(stderr,"WARNING: overriding anisotropic first step for ani init mode\n");
	  E->viscosity.anivisc_start_from_iso = TRUE;
	}
      /* ratio between weak and strong direction */
      input_double("ani_vis2_factor",&(E->viscosity.ani_vis2_factor),"1.0",m);
      
      
    }
#endif
    /* allocate memory here */
    allocate_visc_vars(E);

    for(i=0;i < E->viscosity.num_mat;i++)
        E->viscosity.N0[i]=1.0;
    input_float_vector("visc0",E->viscosity.num_mat,(E->viscosity.N0),m);

    input_boolean("TDEPV",&(E->viscosity.TDEPV),"on",m);
    input_int("rheol",&(E->viscosity.RHEOL),"3",m);
    if (E->viscosity.TDEPV) {
        for(i=0;i < E->viscosity.num_mat;i++) {
            E->viscosity.T[i] = 0.0;
            E->viscosity.Z[i] = 0.0;
            E->viscosity.E[i] = 0.0;
        }

        input_float_vector("viscT",E->viscosity.num_mat,(E->viscosity.T),m);
        input_float_vector("viscE",E->viscosity.num_mat,(E->viscosity.E),m);
        input_float_vector("viscZ",E->viscosity.num_mat,(E->viscosity.Z),m);

        /* for viscosity 8 */
        input_float("T_sol0",&(E->viscosity.T_sol0),"0.6",m);
        input_float("ET_red",&(E->viscosity.ET_red),"0.1",m);
    }


    E->viscosity.sdepv_misfit = 1.0;
    input_boolean("SDEPV",&(E->viscosity.SDEPV),"off",m);
    if (E->viscosity.SDEPV) {
      E->viscosity.sdepv_visited = 0;
      input_float_vector("sdepv_expt",E->viscosity.num_mat,(E->viscosity.sdepv_expt),m);
    }


    input_boolean("PDEPV",&(E->viscosity.PDEPV),"off",m); /* plasticity addition by TWB */
    if (E->viscosity.PDEPV) {
      E->viscosity.pdepv_visited = 0;
      for(i=0;i < E->viscosity.num_mat;i++) {
          E->viscosity.pdepv_a[i] = 1.e20; /* \sigma_y = min(a + b * (1-r),y) */
          E->viscosity.pdepv_b[i] = 0.0;
          E->viscosity.pdepv_y[i] = 1.e20;
      }

      input_boolean("psrw",&(E->viscosity.psrw),"off",m); /* SRW? else regular plasiticity */
      input_boolean("pdepv_eff",&(E->viscosity.pdepv_eff),"on",m); /* effective
								      or
								      min/max
								      approach */
      input_float_vector("pdepv_a",E->viscosity.num_mat,(E->viscosity.pdepv_a),m);
      input_float_vector("pdepv_b",E->viscosity.num_mat,(E->viscosity.pdepv_b),m);
      input_float_vector("pdepv_y",E->viscosity.num_mat,(E->viscosity.pdepv_y),m);

      input_float("pdepv_offset",&(E->viscosity.pdepv_offset),"0.0",m);
    }
    input_float("sdepv_misfit",&(E->viscosity.sdepv_misfit),"0.001",m);	/* there should be no harm in having 
									   this parameter read in regardless of 
									   rheology (activated it for anisotropic viscosity)
									*/

    // moved to Composition related, for init purposes
    //input_boolean("CDEPV",&(E->viscosity.CDEPV),"off",m);
    for(i=0;i<10;i++)
      E->viscosity.cdepv_ff[i] = 1.0; /* flavor factors for CDEPV */
    if(E->viscosity.CDEPV){
      /* compositional viscosity */
      if(E->control.tracer < 1){
	fprintf(stderr,"error: CDEPV requires tracers, but tracer is off\n");
	parallel_process_termination();
      }
      if(E->trace.nflavors > 10)
	myerror(E,"error: too many flavors for CDEPV");
      /* read in flavor factors */
      input_float_vector("cdepv_ff",E->trace.nflavors,
			 (E->viscosity.cdepv_ff),m);
      /* and take the log because we're using a geometric avg */
      for(i=0;i<E->trace.nflavors;i++)
	E->viscosity.cdepv_ff[i] = log(E->viscosity.cdepv_ff[i]);
    }


    input_boolean("low_visc_channel",&(E->viscosity.channel),"off",m);
    input_boolean("low_visc_wedge",&(E->viscosity.wedge),"off",m);

    input_float("lv_min_radius",&(E->viscosity.lv_min_radius),"0.9764",m);
    input_float("lv_max_radius",&(E->viscosity.lv_max_radius),"0.9921",m);
    input_float("lv_channel_thickness",&(E->viscosity.lv_channel_thickness),"0.0047",m);
    input_float("lv_reduction",&(E->viscosity.lv_reduction),"0.5",m);

    input_boolean("use_ne_visc_smooth",&(E->viscosity.SMOOTH),"off",m);

    input_boolean("VMAX",&(E->viscosity.MAX),"off",m);
    if (E->viscosity.MAX)
        input_float("visc_max",&(E->viscosity.max_value),"1e22,1,nomax",m);

    input_boolean("VMIN",&(E->viscosity.MIN),"off",m);
    if (E->viscosity.MIN)
        input_float("visc_min",&(E->viscosity.min_value),"1e20",m);

    return;
}


void viscosity_input(struct All_variables *E)
{
    int m = E->parallel.me;

    input_string("Viscosity",E->viscosity.STRUCTURE,"system",m);
    input_int ("visc_smooth_method",&(E->viscosity.smooth_cycles),"0",m);

    if ( strcmp(E->viscosity.STRUCTURE,"system") == 0)
        E->viscosity.FROM_SYSTEM = 1;
    else
        E->viscosity.FROM_SYSTEM = 0;

    if (E->viscosity.FROM_SYSTEM)
        viscosity_system_input(E);

    return;
}


void allocate_visc_vars(struct All_variables *E)
{
  int i,j,k,lim,nel,nno;

  if(E->viscosity.layer_control) {
    /* noz is already defined, but elz is undefined yet */
    i = E->mesh.noz - 1;
  } else {
    i = E->viscosity.num_mat;
  }
  
  E->viscosity.N0 = (float*) malloc(i*sizeof(float));
  E->viscosity.E = (float*) malloc(i*sizeof(float));
  E->viscosity.T = (float*) malloc(i*sizeof(float));
  E->viscosity.Z = (float*) malloc(i*sizeof(float));
  E->viscosity.pdepv_a = (float*) malloc(i*sizeof(float));
  E->viscosity.pdepv_b = (float*) malloc(i*sizeof(float));
  E->viscosity.pdepv_y = (float*) malloc(i*sizeof(float));
  E->viscosity.sdepv_expt = (float*) malloc(i*sizeof(float));
  
  if(E->viscosity.N0 == NULL ||
     E->viscosity.E == NULL ||
     E->viscosity.T == NULL ||
     E->viscosity.Z == NULL ||
     E->viscosity.pdepv_a == NULL ||
     E->viscosity.pdepv_b == NULL ||
     E->viscosity.pdepv_y == NULL ||
     E->viscosity.sdepv_expt == NULL) {
    fprintf(stderr, "Error: Cannot allocate visc memory, rank=%d\n",
	    E->parallel.me);
    parallel_process_termination();
  }

}


/* ============================================ */
/* 

   when called from Drive_solvers:
   
   evisc = E->EVI[E->mesh.levmax]
   visc  = E->VI[E->mesh.levmax]

 */
void get_system_viscosity(E,propogate,evisc,visc)
     struct All_variables *E;
     int propogate;
     float **evisc,**visc;
{
    void visc_from_mat();
    void visc_from_T();
    void visc_from_S();

    void visc_from_P();
    void visc_from_C();

    void apply_viscosity_smoother();
    void visc_from_gint_to_nodes();
    void  visc_from_nodes_to_gint();


    int i,j,m;
    float temp1,temp2,*vvvis;
    double *TG;

    const int vpts = vpoints[E->mesh.nsd];
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
    if(E->viscosity.allow_anisotropic_viscosity){
      if(!E->viscosity.anisotropic_viscosity_init)
	set_anisotropic_viscosity_at_element_level(E,1);
      else
	set_anisotropic_viscosity_at_element_level(E,0);
    }
#endif


    if(E->viscosity.TDEPV)
        visc_from_T(E,evisc,propogate);
    else
        visc_from_mat(E,evisc);

    if(E->viscosity.CDEPV)	/* compositional prefactor */
      visc_from_C(E,evisc);

    if(E->viscosity.SDEPV)
      visc_from_S(E,evisc,propogate);

    if(E->viscosity.PDEPV)	/* "plasticity" */
      visc_from_P(E,evisc);


    /* i think this should me placed differently i.e.  before the
       stress dependence but I won't change it because it's by
       someone else

       TWB
    */
    if(E->viscosity.channel || E->viscosity.wedge)
        apply_low_visc_wedge_channel(E, evisc);


    /* min/max cut-off */

    if(E->viscosity.MAX) {
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.nel;i++)
                for(j=1;j<=vpts;j++)
                    if(evisc[m][(i-1)*vpts + j] > E->viscosity.max_value)
                        evisc[m][(i-1)*vpts + j] = E->viscosity.max_value;
    }

    if(E->viscosity.MIN) {
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=E->lmesh.nel;i++)
                for(j=1;j<=vpts;j++)
                    if(evisc[m][(i-1)*vpts + j] < E->viscosity.min_value)
                        evisc[m][(i-1)*vpts + j] = E->viscosity.min_value;
    }

    if (E->control.verbose)  {
      fprintf(E->fp_out,"output_evisc \n");
      for(m=1;m<=E->sphere.caps_per_proc;m++) {
        fprintf(E->fp_out,"output_evisc for cap %d\n",E->sphere.capid[m]);
      for(i=1;i<=E->lmesh.nel;i++)
          fprintf(E->fp_out,"%d %d %f %f\n",i,E->mat[m][i],evisc[m][(i-1)*vpts+1],evisc[m][(i-1)*vpts+7]);
      }
      fflush(E->fp_out);
    }
    /* interpolate from gauss quadrature points to node points for output */
    visc_from_gint_to_nodes(E,evisc,visc,E->mesh.levmax);

    if(E->viscosity.SMOOTH){ /* go the other way, for
					    smoothing */
      visc_from_nodes_to_gint(E,visc,evisc,E->mesh.levmax);
    }

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC /* allow for anisotropy */
    if(E->viscosity.allow_anisotropic_viscosity){
      visc_from_gint_to_nodes(E,E->EVI2[E->mesh.levmax], E->VI2[E->mesh.levmax],E->mesh.levmax);
      visc_from_gint_to_nodes(E,E->EVIn1[E->mesh.levmax], E->VIn1[E->mesh.levmax],E->mesh.levmax);
      visc_from_gint_to_nodes(E,E->EVIn2[E->mesh.levmax], E->VIn2[E->mesh.levmax],E->mesh.levmax);
      visc_from_gint_to_nodes(E,E->EVIn3[E->mesh.levmax], E->VIn3[E->mesh.levmax],E->mesh.levmax);
      normalize_director_at_nodes(E,E->VIn1[E->mesh.levmax],E->VIn2[E->mesh.levmax],E->VIn3[E->mesh.levmax],E->mesh.levmax);
      
      if(E->viscosity.SMOOTH){ 
	if(E->parallel.me == 0)fprintf(stderr,"WARNING: smoothing anisotropic viscosity, perhaps not a good idea\n");
	visc_from_nodes_to_gint(E,E->VI2[E->mesh.levmax],E->EVI2[E->mesh.levmax],E->mesh.levmax);
	visc_from_nodes_to_gint(E,E->VIn1[E->mesh.levmax],E->EVIn1[E->mesh.levmax],E->mesh.levmax);
	visc_from_nodes_to_gint(E,E->VIn2[E->mesh.levmax],E->EVIn2[E->mesh.levmax],E->mesh.levmax);
	visc_from_nodes_to_gint(E,E->VIn3[E->mesh.levmax],E->EVIn3[E->mesh.levmax],E->mesh.levmax);
	normalize_director_at_gint(E,E->EVIn1[E->mesh.levmax],E->EVIn2[E->mesh.levmax],E->EVIn3[E->mesh.levmax],E->mesh.levmax);
    
      }
    }
#endif
    return;
}



void initial_viscosity(struct All_variables *E)
{
    void report(struct All_variables*, char*);

    report(E,"Initialize viscosity field");

    if (E->viscosity.FROM_SYSTEM)
        get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);

    return;
}


void visc_from_mat(E,EEta)
     struct All_variables *E;
     float **EEta;
{

    int i,m,jj;
    if(E->control.mat_control){	/* use pre-factor even without temperature dependent viscosity */
      for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(i=1;i<=E->lmesh.nel;i++)
	  for(jj=1;jj<=vpoints[E->mesh.nsd];jj++)
	    EEta[m][ (i-1)*vpoints[E->mesh.nsd]+jj ] = E->viscosity.N0[E->mat[m][i]-1]*E->VIP[m][i];
     }else{
      for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(i=1;i<=E->lmesh.nel;i++)
	  for(jj=1;jj<=vpoints[E->mesh.nsd];jj++)
	    EEta[m][ (i-1)*vpoints[E->mesh.nsd]+jj ] = E->viscosity.N0[E->mat[m][i]-1];
    }

    return;
}


void read_visc_layer_file(struct All_variables *E)
{
    int i;
    FILE *fp;
    char junk[256];

    fp = fopen(E->viscosity.layer_file, "r");
    if (fp == NULL) {
        fprintf(E->fp, "(Viscosity_structures #1) Cannot open %s\n", E->viscosity.layer_file);
        exit(8);
    }


    /* default value */
    for(i=0; i<E->mesh.elz; i++) {
        E->viscosity.N0[i] =
            E->viscosity.E[i] =
            E->viscosity.T[i] =
            E->viscosity.Z[i] =
            E->viscosity.pdepv_a[i] =
            E->viscosity.pdepv_b[i] =
            E->viscosity.pdepv_y[i] =
            E->viscosity.sdepv_expt[i] = 0;
    }

    read_visc_param_from_file(E, "visc0", E->viscosity.N0, fp);
    if(E->viscosity.TDEPV) {
        read_visc_param_from_file(E, "viscE", E->viscosity.E, fp);
        read_visc_param_from_file(E, "viscT", E->viscosity.T, fp);
        read_visc_param_from_file(E, "viscZ", E->viscosity.Z, fp);
    }

    if(E->viscosity.SDEPV) {
        read_visc_param_from_file(E, "sdepv_expt", E->viscosity.sdepv_expt, fp);
    }

    if(E->viscosity.PDEPV) {
        read_visc_param_from_file(E, "pdepv_a", E->viscosity.pdepv_a, fp);
        read_visc_param_from_file(E, "pdepv_b", E->viscosity.pdepv_b, fp);
        read_visc_param_from_file(E, "pdepv_y", E->viscosity.pdepv_y, fp);
    }

    return;
}


void visc_from_T(E,EEta,propogate)
     struct All_variables *E;
     float **EEta;
     int propogate;
{
    int m,i,k,l,z,jj,kk;
    float zero,one,eta0,temp,tempa,TT[9];
    float zzz,zz[9],dr;
    float visc1, visc2;
    const int vpts = vpoints[E->mesh.nsd];
    const int ends = enodes[E->mesh.nsd];
    const int nel = E->lmesh.nel;

    one = 1.0;
    zero = 0.0;

    /* consistent handling : l is (material number - 1) to allow
       addressing viscosity arrays, which are all 0...n-1  */
    switch (E->viscosity.RHEOL)   {
    case 1:
        /* eta = N_0 exp( E * (T_0 - T))  */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;

                if(E->control.mat_control==0)
                    tempa = E->viscosity.N0[l];
                else 
                    tempa = E->viscosity.N0[l]*E->VIP[m][i];

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    EEta[m][ (i-1)*vpts + jj ] = tempa*
                        exp( E->viscosity.E[l] * (E->viscosity.T[l] - temp));

                }
            }
        break;

    case 2:
        /* eta = N_0 exp(-T/T_0) */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;

                if(E->control.mat_control==0)
                    tempa = E->viscosity.N0[l];
                else 
                    tempa = E->viscosity.N0[l]*E->VIP[m][i];

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    EEta[m][ (i-1)*vpts + jj ] = tempa*
                        exp( -temp / E->viscosity.T[l]);

                }
            }
        break;

    case 3:
        /* eta = N_0 exp(E/(T+T_0) - E/(1+T_0)) 

	   where T is normalized to be within 0...1

	 */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;
		if(E->control.mat_control) /* switch moved up here TWB */
		  tempa = E->viscosity.N0[l] * E->VIP[m][i];
		else
		  tempa = E->viscosity.N0[l];

                for(kk=1;kk<=ends;kk++) {
		  TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)   {	/* took out
						   computation of
						   depth, not needed
						   TWB */
		      TT[kk]=max(TT[kk],zero);
		      temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
                    }
		    EEta[m][ (i-1)*vpts + jj ] = tempa*
		      exp( E->viscosity.E[l]/(temp+E->viscosity.T[l])
			   - E->viscosity.E[l]/(one +E->viscosity.T[l]) );
                }
            }
        break;

    case 4:
        /* eta = N_0 exp( (E + (1-z)Z_0) / (T+T_0) ) */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;
		if(E->control.mat_control) /* moved this up here TWB */
		  tempa = E->viscosity.N0[l] * E->VIP[m][i];
		else
		  tempa = E->viscosity.N0[l];

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                    zz[kk] = (1.-E->sx[m][3][E->ien[m][i].node[kk]]);
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        TT[kk]=max(TT[kk],zero);
                        temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
                        zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }


		    EEta[m][ (i-1)*vpts + jj ] = tempa*
		      exp( (E->viscosity.E[l] +  E->viscosity.Z[l]*zzz )
			   / (E->viscosity.T[l]+temp) );

                }
            }
        break;


    case 5:

        /* when mat_control=0, same as rheol 3,
           when mat_control=1, applying viscosity cut-off before mat_control */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;
                tempa = E->viscosity.N0[l];
                /* fprintf(stderr,"\nINSIDE visc_from_T, l=%d, tempa=%g",l+1,tempa);*/
                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        TT[kk]=max(TT[kk],zero);
                        temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
                    }

                    if(E->control.mat_control==0){
                        EEta[m][ (i-1)*vpts + jj ] = tempa*
                            exp( E->viscosity.E[l]/(temp+E->viscosity.T[l])
                                 - E->viscosity.E[l]/(one +E->viscosity.T[l]) );
		    }else{
                       visc2 = tempa*
	               exp( E->viscosity.E[l]/(temp+E->viscosity.T[l])
		          - E->viscosity.E[l]/(one +E->viscosity.T[l]) );
                       if(E->viscosity.MAX) {
                           if(visc2 > E->viscosity.max_value)
                               visc2 = E->viscosity.max_value;
                         }
                       if(E->viscosity.MIN) {
                           if(visc2 < E->viscosity.min_value)
                               visc2 = E->viscosity.min_value;
                         }
                       EEta[m][ (i-1)*vpts + jj ] = E->VIP[m][i]*visc2;
                      }

                }
            }
        break;


    case 6:
        /* like case 1, but allowing for depth-dependence if Z_0 != 0
           eta = N_0 exp(E(T_0-T) + (1-z) Z_0 )
        */

        for(m=1;m <= E->sphere.caps_per_proc;m++)
	  for(i=1;i <= nel;i++)   {

	    l = E->mat[m][i] - 1;

	    if(E->control.mat_control)
	      tempa = E->viscosity.N0[l] * E->VIP[m][i];
	    else
	      tempa = E->viscosity.N0[l];

	    for(kk=1;kk<=ends;kk++) {
	      TT[kk] = E->T[m][E->ien[m][i].node[kk]];
	      zz[kk] = (1.0 - E->sx[m][3][E->ien[m][i].node[kk]]);
	    }

	    for(jj=1;jj <= vpts;jj++) {
	      temp=0.0;zzz=0.0;
	      for(kk=1;kk <= ends;kk++)   {
		TT[kk]=max(TT[kk],zero);
		temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)];
		zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
	      }
	      EEta[m][ (i-1)*vpts + jj ] = tempa*
		exp( E->viscosity.E[l]*(E->viscosity.T[l] - temp) +
		     zzz *  E->viscosity.Z[l]);
	      /*
               if(E->parallel.me == 0)
	         fprintf(stderr,"z %11g km mat %i N0 %11g T %11g T0 %11g E %11g Z %11g mat: %i log10(eta): %11g\n",
                        zzz *E->data.radius_km ,l+1,
                        tempa,temp,E->viscosity.T[l],E->viscosity.E[l], E->viscosity.Z[l],l+1,log10(EEta[m][ (i-1)*vpts + jj ]));
              */
	    }
	  }
        break;


    case 7:
        /* The viscosity formulation (dimensional) is:
           visc=visc0*exp[(Ea+p*Va)/(R*T)]

           Typical values for dry upper mantle are:
           Ea = 300 KJ/mol ; Va = 1.e-5 m^3/mol

           T=DT*(T0+T');
           where DT - temperature contrast (from Rayleigh number)
           T' - nondimensional temperature;
           T0 - nondimensional surface tempereture;

           =>
           visc = visc0 * exp{(Ea+p*Va) / [R*DT*(T0 + T')]}
                = visc0 * exp{[Ea/(R*DT) + p*Va/(R*DT)] / (T0 + T')}

           so:
           E->viscosity.E = Ea/(R*DT);
           (1-r) = p/(rho*g);
           E->viscosity.Z = Va*rho*g/(R*DT);
           E->viscosity.T = T0;

           after normalizing visc=1 at T'=1 and r=r_CMB:
           visc = visc0*exp{ [viscE + (1-r)*viscZ] / (viscT+T')
                - [viscE + (1-r_CMB)*viscZ] / (viscT+1) }
        */

        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
	      l = E->mat[m][i] - 1;

		if(E->control.mat_control)
		  tempa = E->viscosity.N0[l] * E->VIP[m][i];
		else
		  tempa = E->viscosity.N0[l];

                for(kk=1;kk<=ends;kk++) {
                    TT[kk] = E->T[m][E->ien[m][i].node[kk]];
                    zz[kk] = (1.-E->sx[m][3][E->ien[m][i].node[kk]]);
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {
                        temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                        zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
                    }


                    EEta[m][ (i-1)*vpts + jj ] = tempa*
                        exp( (E->viscosity.E[l] +  E->viscosity.Z[l]*zzz )
                             / (E->viscosity.T[l] + temp)
                             - (E->viscosity.E[l] +
                                E->viscosity.Z[l]*(E->sphere.ro-E->sphere.ri) )
                             / (E->viscosity.T[l] + one) );
                }
            }
        break;

    case 8:
        /*
          eta0 = N_0 exp(E/(T+T_0) - E/(1+T_0))

          eta =        eta0 if T  < T_sol0 + 2(1-z)
          eta = ET_red*eta0 if T >= T_sol0 + 2(1-z)

	  T is limited to lie between 0 and 1

          where z is normalized by layer
          thickness, and T_sol0 is something
          like 0.6, and ET_red = 0.1

          (same as case 3, but for viscosity reduction)
        */

        dr = E->sphere.ro - E->sphere.ri;
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;
		if(E->control.mat_control) 
		  tempa = E->viscosity.N0[l] * E->VIP[m][i];
		else
		  tempa = E->viscosity.N0[l];

                for(kk=1;kk<=ends;kk++) {
		  TT[kk] = E->T[m][E->ien[m][i].node[kk]];
		  zz[kk] = E->sx[m][3][E->ien[m][i].node[kk]]; /* radius */
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {	
		      TT[kk]=max(TT[kk],zero);
		      temp += min(TT[kk],one) * E->N.vpt[GNVINDEX(kk,jj)]; /* mean temp */
		      zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];/* mean r */
                    }
		    /* convert to z, as defined to be unity at surface
		       and zero at CMB */
		    zzz = (zzz - E->sphere.ri)/dr;
		    visc1 = tempa* exp( E->viscosity.E[l]/(temp+E->viscosity.T[l]) 
				  - E->viscosity.E[l]/(one +E->viscosity.T[l]) );
		    if(temp < E->viscosity.T_sol0 + 2.*(1.-zzz))
		      EEta[m][ (i-1)*vpts + jj ] = visc1;
		    else
		      EEta[m][ (i-1)*vpts + jj ] = visc1 * E->viscosity.ET_red;
                }
            }
        break;
    case 9:
        /* eta = N_0 exp(E/(T+T_0) - E/(1+T_0)) 

	   like option 3, but T is allow to vary beyond 1 

	 */
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;
		if(E->control.mat_control) /* switch moved up here TWB */
		  tempa = E->viscosity.N0[l] * E->VIP[m][i];
		else
		  tempa = E->viscosity.N0[l];
                for(kk=1;kk<=ends;kk++) 
		  TT[kk] = E->T[m][E->ien[m][i].node[kk]];
		
                for(jj=1;jj<=vpts;jj++) {
                    temp=0.0;
                    for(kk=1;kk<=ends;kk++)
		      temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)];
		    EEta[m][ (i-1)*vpts + jj ] = tempa*
		      exp( E->viscosity.E[l]/(temp+E->viscosity.T[l])
			   - E->viscosity.E[l]/(one +E->viscosity.T[l]) );
                }
            }
        break;
    case 10:
        /*
          eta0 = N_0 exp(E/(T+T_0) - E/(1+T_0))

          eta =        eta0 if T  < T_sol0 + 2(1-z)
          eta = ET_red*eta0 if T >= T_sol0 + 2(1-z)

	  like rheol == 8, but T is not limited to lie between 0 and 1

        */

        dr = E->sphere.ro - E->sphere.ri;
        for(m=1;m<=E->sphere.caps_per_proc;m++)
            for(i=1;i<=nel;i++)   {
                l = E->mat[m][i] - 1;
		if(E->control.mat_control) 
		  tempa = E->viscosity.N0[l] * E->VIP[m][i];
		else
		  tempa = E->viscosity.N0[l];

                for(kk=1;kk<=ends;kk++) {
		  TT[kk] = E->T[m][E->ien[m][i].node[kk]];
		  zz[kk] = E->sx[m][3][E->ien[m][i].node[kk]]; /* radius */
                }

                for(jj=1;jj<=vpts;jj++) {
                    temp=zzz=0.0;
                    for(kk=1;kk<=ends;kk++)   {	
		      temp += TT[kk] * E->N.vpt[GNVINDEX(kk,jj)]; /* mean temp */
		      zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];/* mean r */
                    }
		    zzz = (zzz - E->sphere.ri)/dr;
		    visc1 = tempa* exp( E->viscosity.E[l]/(temp+E->viscosity.T[l]) 
				  - E->viscosity.E[l]/(one +E->viscosity.T[l]) );
		    if(temp < E->viscosity.T_sol0 + 2.*(1.-zzz))
		      EEta[m][ (i-1)*vpts + jj ] = visc1;
		    else
		      EEta[m][ (i-1)*vpts + jj ] = visc1 * E->viscosity.ET_red;
                }
            }
        break;
	

    case 100:
        /* user-defined viscosity law goes here */
        fprintf(stderr, "Need user definition for viscosity law: 'rheol=%d'\n",
                E->viscosity.RHEOL);
        parallel_process_termination();
        break;

    default:
        /* unknown option */
        fprintf(stderr, "Invalid value of 'rheol=%d'\n", E->viscosity.RHEOL);

        parallel_process_termination();
        break;
    }


    return;
}


void visc_from_S(E,EEta,propogate)
     struct All_variables *E;
     float **EEta;
     int propogate;
{
    float one,two,scale,stress_magnitude,depth,exponent1;
    float *eedot;

    void strain_rate_2_inv();
    int m,e,l,z,jj,kk;

    const int vpts = vpoints[E->mesh.nsd];
    const int nel = E->lmesh.nel;

    eedot = (float *) malloc((2+nel)*sizeof(float));
    one = 1.0;
    two = 2.0;

    for(m=1;m<=E->sphere.caps_per_proc;m++)  {
      if(E->viscosity.sdepv_visited){
	
        /* get second invariant for all elements */
        strain_rate_2_inv(E,m,eedot,1);
      }else{
	for(e=1;e<=nel;e++)	/* initialize with unity if no velocities around */
	  eedot[e] = 1.0;
	E->viscosity.sdepv_visited = 1;

      }
        /* eedot cannot be too small, or the viscosity will go to inf */
	for(e=1;e<=nel;e++){
	  eedot[e] = max(eedot[e], 1.0e-16);
	}

        for(e=1;e<=nel;e++)   {
            exponent1= one/E->viscosity.sdepv_expt[E->mat[m][e]-1];
            scale=pow(eedot[e],exponent1-one);
            for(jj=1;jj<=vpts;jj++)
                EEta[m][(e-1)*vpts + jj] = scale*pow(EEta[m][(e-1)*vpts+jj],exponent1);
        }
    }

    free ((void *)eedot);
    return;
}

void visc_from_P(E,EEta) /* "plasticity" implementation


			 psrw = FALSE

			 viscosity will be limited by a yield stress
			 
			 \sigma_y  = min(a + b * (1-r), y)

			 where a,b,y are parameters input via pdepv_a,b,y

			 and

			 \eta_y = \sigma_y / (2 \eps_II)

			 where \eps_II is the second invariant. Then

			 \eta_eff = (\eta_0 \eta_y)/(\eta_0 + \eta_y)

			 for pdepv_eff = 1

			 or

			 \eta_eff = min(\eta_0,\eta_y)

			 for pdepv_eff = 0

			 where \eta_0 is the regular viscosity


			 psrw = 1

			 a strain-rate weakening rheology applies
			 based on a steady state stress-strain
			 relationship following, e.g., Tackley 1998

			 \eta_ett = (\sigma_y^2 \eta_0)/(\sigma_y^2 + \eta_0^2 (2 \eps_II^2))

			 where sigma_y is defined as above

			 where the 2\eps_II arises because our \eps_II has the 1/2 factor in it

			 TWB

			 */
     struct All_variables *E;
     float **EEta;
{
  float *eedot,zz[9],zzz,tau,eta_p,eta_new,tau2,eta_old,eta_old2;
  int m,e,l,z,jj,kk;

  const int vpts = vpoints[E->mesh.nsd];
  const int nel = E->lmesh.nel;
  const int ends = enodes[E->mesh.nsd];
  
  void strain_rate_2_inv();
  
  
  eedot = (float *) malloc((2+nel)*sizeof(float));
  
  for(m=1;m<=E->sphere.caps_per_proc;m++)  {
    
    if(E->viscosity.pdepv_visited){
      if(E->viscosity.psrw)
	strain_rate_2_inv(E,m,eedot,0);	/* get second invariant for all elements, don't take sqrt */
      else
	strain_rate_2_inv(E,m,eedot,1);	/* get second invariant for all elements */
    }else{
      for(e=1;e<=nel;e++)	/* initialize with unity if no velocities around */
	eedot[e] = 1.0;
      if(m == E->sphere.caps_per_proc)
	E->viscosity.pdepv_visited = 1;
      if((E->parallel.me == 0)&&(E->control.verbose)){
	fprintf(stderr,"num mat: %i a: %g b: %g y: %g %s\n",
		e,E->viscosity.pdepv_a[e],E->viscosity.pdepv_b[e],E->viscosity.pdepv_y[e],
		(E->viscosity.psrw)?(" -- SRW"):(""));
      }
    }
    if(!E->viscosity.psrw){
      /* 
	 regular plasticity
      */
      for(e=1;e <= nel;e++)   {	/* loop through all elements */
	
	l = E->mat[m][e] -1 ;	/* material of this element */
	
	for(kk=1;kk <= ends;kk++) /* nodal depths */
	  zz[kk] = (1.0 - E->sx[m][3][E->ien[m][e].node[kk]]); /* for depth, zz = 1 - r */
	
	for(jj=1;jj <= vpts;jj++){ /* loop through integration points */
	  
	  zzz = 0.0;		/* get mean depth of integration point */
	  for(kk=1;kk<=ends;kk++)
	    zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
	  
	  /* depth dependent yield stress */
	  tau = E->viscosity.pdepv_a[l] + zzz * E->viscosity.pdepv_b[l];
	  
	  /* min of depth dep. and constant yield stress */
	  tau = min(tau,  E->viscosity.pdepv_y[l]);
	  
	  /* yield viscosity */
	  eta_p = tau/(2.0 * eedot[e] + 1e-7) + E->viscosity.pdepv_offset;
	  if(E->viscosity.pdepv_eff){
	    /* two dashpots in series */
	    eta_new  = 1.0/(1.0/EEta[m][ (e-1)*vpts + jj ] + 1.0/eta_p);
	  }else{
	    /* min viscosities*/
	    eta_new  = min(EEta[m][ (e-1)*vpts + jj ], eta_p);
	  }
	  //fprintf(stderr,"z: %11g mat: %i a: %11g b: %11g y: %11g ee: %11g tau: %11g eta_p: %11g eta_new: %11g eta_old: %11g\n",
	  //	  zzz,l,E->viscosity.pdepv_a[l], E->viscosity.pdepv_b[l],E->viscosity.pdepv_y[l],
	  //	  eedot[e],tau,eta_p,eta_new,EEta[m][(e-1)*vpts + jj]);
	  EEta[m][(e-1)*vpts + jj] = eta_new;
	} /* end integration point loop */
      }	/* end element loop */
    }else{
      /* strain-rate weakening, steady state solution */
      for(e=1;e <= nel;e++)   {	/* loop through all elements */
	
	l = E->mat[m][e] -1 ;	
	for(kk=1;kk <= ends;kk++)
	  zz[kk] = (1.0 - E->sx[m][3][E->ien[m][e].node[kk]]); 
	for(jj=1;jj <= vpts;jj++){ 
	  zzz = 0.0;
	  for(kk=1;kk<=ends;kk++)
	    zzz += zz[kk] * E->N.vpt[GNVINDEX(kk,jj)];
	  /* compute sigma_y as above */
	  tau = E->viscosity.pdepv_a[l] + zzz * E->viscosity.pdepv_b[l];
	  tau = min(tau,  E->viscosity.pdepv_y[l]);
	  tau2 = tau * tau;
	  if(tau < 1e10){
	    /*  */
	    eta_old = EEta[m][ (e-1)*vpts + jj ];
	    eta_old2 = eta_old * eta_old;
	    /* effectiev viscosity */
	    eta_new = (tau2 * eta_old)/(tau2 + 2.0 * eta_old2 * eedot[e]);
	    //fprintf(stderr,"SRW: a %11g b %11g y %11g z %11g sy: %11g e2: %11g eold: %11g enew: %11g logr: %.3f\n",
	    //	    E->viscosity.pdepv_a[l],E->viscosity.pdepv_b[l],E->viscosity.pdepv_y[l],zzz,tau,eedot[e],eta_old,eta_new,
	    //	    log10(eta_new/eta_old));
	    EEta[m][(e-1)*vpts + jj] = eta_new;
	  }
	}
      }
    }
  } /* end caps loop */
  free ((void *)eedot);
  return;
}

/*

multiply with compositional factor which is determined by a geometric
mean average from the tracer composition, assuming two flavors and
compositions between zero and unity

*/
void visc_from_C( E, EEta)
     struct All_variables *E;
     float **EEta;
{
  double vmean,cc_loc[10],CC[10][9],cbackground;
  int m,l,z,jj,kk,i,p,q;


  const int vpts = vpoints[E->mesh.nsd];
  const int nel = E->lmesh.nel;
  const int ends = enodes[E->mesh.nsd];

  for(m=1;m <= E->sphere.caps_per_proc;m++)  {
    for(i = 1; i <= nel; i++){
      /* determine composition of each of the nodes of the
	 element */
        for(p=0; p<E->composition.ncomp; p++) {
            for(kk = 1; kk <= ends; kk++){
                CC[p][kk] = E->composition.comp_node[m][p][E->ien[m][i].node[kk]];
                if(CC[p][kk] < 0)CC[p][kk]=0.0;
                if(CC[p][kk] > 1)CC[p][kk]=1.0;
            }
        }
        for(jj = 1; jj <= vpts; jj++) {
            /* concentration of background material */
            cbackground = 1;
            for(p=0; p<E->composition.ncomp; p++) {
                /* compute mean composition  */
                cc_loc[p] = 0.0;
                for(kk = 1; kk <= ends; kk++) {
                    cc_loc[p] += CC[p][kk] * E->N.vpt[GNVINDEX(kk, jj)];
                }
                cbackground -= cc_loc[p];
            }

            /* geometric mean of viscosity */
            vmean = cbackground * E->viscosity.cdepv_ff[0];
            for(p=0; p<E->composition.ncomp; p++) {
                vmean += cc_loc[p] * E->viscosity.cdepv_ff[p+1];
            }
            vmean = exp(vmean);

            /* multiply the viscosity with this prefactor */
            EEta[m][ (i-1)*vpts + jj ] *= vmean;

        } /* end jj loop */
    } /* end el loop */
  } /* end cap */
}

void strain_rate_2_inv(E,m,EEDOT,SQRT)
     struct All_variables *E;
     float *EEDOT;
     int m,SQRT;
{
    void get_rtf_at_ppts();
    void velo_from_element();
    void construct_c3x3matrix_el();
    void get_ba_p();

    struct Shape_function_dx *GNx;

    double edot[4][4], rtf[4][9];
    double theta;
    double ba[9][9][4][7];
    float VV[4][9], Vxyz[7][9], dilation[9];
    
    int e, i, j, p, q, n;

    const int nel = E->lmesh.nel;
    const int dims = E->mesh.nsd;
    const int ends = enodes[dims];
    const int lev = E->mesh.levmax;
    const int ppts = ppoints[dims];
    const int sphere_key = 1;

    for(e=1; e<=nel; e++) {

        get_rtf_at_ppts(E, m, lev, e, rtf); /* pressure evaluation
					       points */
        velo_from_element(E, VV, m, e, sphere_key);
        GNx = &(E->gNX[m][e]);

        theta = rtf[1][1];


        /* Vxyz is the strain rate vector, whose relationship with
         * the strain rate tensor (e) is that:
         *    Vxyz[1] = e11
         *    Vxyz[2] = e22
         *    Vxyz[3] = e33
         *    Vxyz[4] = 2*e12
         *    Vxyz[5] = 2*e13
         *    Vxyz[6] = 2*e23
         * where 1 is theta, 2 is phi, and 3 is r
         */
        for(j=1; j<=ppts; j++) {
            Vxyz[1][j] = 0.0;
            Vxyz[2][j] = 0.0;
            Vxyz[3][j] = 0.0;
            Vxyz[4][j] = 0.0;
            Vxyz[5][j] = 0.0;
            Vxyz[6][j] = 0.0;
            dilation[j] = 0.0;
        }

        if ((E->control.precise_strain_rate) || (theta < 0.09) || (theta > 3.05)) {
            /* When the element is close to the poles, use a more
             * precise method to compute the strain rate. 
	     
	     if precise_strain_rate=on, will always choose this option

	    */

            if ((e-1)%E->lmesh.elz==0) {
                construct_c3x3matrix_el(E,e,&E->element_Cc,&E->element_Ccx,lev,m,1);
            }

            get_ba_p(&(E->N), GNx, &E->element_Cc, &E->element_Ccx,
                     rtf, E->mesh.nsd, ba);

            for(j=1;j<=ppts;j++)
                for(p=1;p<=6;p++)
                    for(i=1;i<=ends;i++)
                        for(q=1;q<=dims;q++) {
                            Vxyz[p][j] += ba[i][j][q][p] * VV[q][i];
                        }

        }
        else {
            for(j=1; j<=ppts; j++) {
                for(i=1; i<=ends; i++) {
                    Vxyz[1][j] += (VV[1][i] * GNx->ppt[GNPXINDEX(0, i, j)]
                                   + VV[3][i] * E->N.ppt[GNPINDEX(i, j)])
                        * rtf[3][j];
                    Vxyz[2][j] += ((VV[2][i] * GNx->ppt[GNPXINDEX(1, i, j)]
                                    + VV[1][i] * E->N.ppt[GNPINDEX(i, j)]
                                    * cos(rtf[1][j])) / sin(rtf[1][j])
                                   + VV[3][i] * E->N.ppt[GNPINDEX(i, j)])
                        * rtf[3][j];
                    Vxyz[3][j] += VV[3][i] * GNx->ppt[GNPXINDEX(2, i, j)];

                    Vxyz[4][j] += ((VV[1][i] * GNx->ppt[GNPXINDEX(1, i, j)]
                                    - VV[2][i] * E->N.ppt[GNPINDEX(i, j)]
                                    * cos(rtf[1][j])) / sin(rtf[1][j])
                                   + VV[2][i] * GNx->ppt[GNPXINDEX(0, i, j)])
                        * rtf[3][j];
                    Vxyz[5][j] += VV[1][i] * GNx->ppt[GNPXINDEX(2, i, j)]
                        + rtf[3][j] * (VV[3][i] * GNx->ppt[GNPXINDEX(0, i, j)]
                                       - VV[1][i] * E->N.ppt[GNPINDEX(i, j)]);
                    Vxyz[6][j] += VV[2][i] * GNx->ppt[GNPXINDEX(2, i, j)]
                        + rtf[3][j] * (VV[3][i]
                                       * GNx->ppt[GNPXINDEX(1, i, j)]
                                       / sin(rtf[1][j])
                                       - VV[2][i] * E->N.ppt[GNPINDEX(i, j)]);
                }
            }
        } /* end of fast, imprecise strain-rate computation */

        if(E->control.inv_gruneisen != 0) {
            for(j=1; j<=ppts; j++)
                dilation[j] = (Vxyz[1][j] + Vxyz[2][j] + Vxyz[3][j]) / 3.0;
        }

        edot[1][1] = edot[2][2] = edot[3][3] = 0;
        edot[1][2] = edot[1][3] = edot[2][3] = 0;

        /* edot is 2 * (the deviatoric strain rate tensor) */
        for(j=1; j <= ppts; j++) {
            edot[1][1] += 2.0 * (Vxyz[1][j] - dilation[j]);
            edot[2][2] += 2.0 * (Vxyz[2][j] - dilation[j]);
            edot[3][3] += 2.0 * (Vxyz[3][j] - dilation[j]);
            edot[1][2] += Vxyz[4][j];
            edot[1][3] += Vxyz[5][j];
            edot[2][3] += Vxyz[6][j];
        }

        EEDOT[e] = edot[1][1] * edot[1][1]
            + edot[1][2] * edot[1][2] * 2.0
            + edot[2][2] * edot[2][2]
            + edot[2][3] * edot[2][3] * 2.0
            + edot[3][3] * edot[3][3]
            + edot[1][3] * edot[1][3] * 2.0;
    }

    if(SQRT)
	for(e=1;e<=nel;e++)
	    EEDOT[e] =  sqrt(0.5 *EEDOT[e]);
    else
	for(e=1;e<=nel;e++)
	    EEDOT[e] *=  0.5;

    return;
}


static void apply_low_visc_wedge_channel(struct All_variables *E, float **evisc)
{
    void parallel_process_termination();

    int i,j,m;
    const int vpts = vpoints[E->mesh.nsd];
    float *F;

    /* low viscosity channel/wedge require tracers to work */
    if(E->control.tracer == 0) {
        if(E->parallel.me == 0) {
            fprintf(stderr, "Error: low viscosity channel/wedge is turned on, "
                   "but tracer is off!\n");
            fprintf(E->fp, "Error: low viscosity channel/wedge is turned on, "
                   "but tracer is off!\n");
            fflush(E->fp);
        }
        parallel_process_termination();
    }


    F = (float *)malloc((E->lmesh.nel+1)*sizeof(float));
    for(i=1 ; i<=E->lmesh.nel ; i++)
        F[i] = 0.0;

    /* if low viscosity channel ... */
    if(E->viscosity.channel)
        low_viscosity_channel_factor(E, F);


    /* if low viscosity wedge ... */
    if(E->viscosity.wedge)
        low_viscosity_wedge_factor(E, F);


    for(i=1 ; i<=E->lmesh.nel ; i++) {
        if (F[i] != 0.0)
            for(m = 1 ; m <= E->sphere.caps_per_proc ; m++) {
                for(j=1;j<=vpts;j++) {
                    evisc[m][(i-1)*vpts + j] = F[i];
            }
        }
    }


    free(F);

    return;
}




static void low_viscosity_channel_factor(struct All_variables *E, float *F)
{
    int i, ii, k, m, e, ee;
    int nz_min[NCS], nz_max[NCS];
    const int flavor = 0;
    double rad_mean, rr;

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        /* find index of radius corresponding to lv_min_radius */
        for(e=1; e<=E->lmesh.elz; e++) {
            rad_mean = 0.5 * (E->sx[m][3][E->ien[m][e].node[1]] +
                              E->sx[m][3][E->ien[m][e].node[8]]);
            if(rad_mean >= E->viscosity.lv_min_radius) break;
        }
        nz_min[m] = e;

        /* find index of radius corresponding to lv_max_radius */
        for(e=E->lmesh.elz; e>=1; e--) {
            rad_mean = 0.5 * (E->sx[m][3][E->ien[m][e].node[1]] +
                              E->sx[m][3][E->ien[m][e].node[8]]);
            if(rad_mean <= E->viscosity.lv_max_radius) break;
        }
        nz_max[m] = e;
    }



    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(k=1; k<=E->lmesh.elx*E->lmesh.ely; k++) {
            for(i=nz_min[m]; i<=nz_max[m]; i++) {
                e = (k-1)*E->lmesh.elz + i;

                rad_mean = 0.5 * (E->sx[m][3][E->ien[m][e].node[1]] +
                                  E->sx[m][3][E->ien[m][e].node[8]]);

                /* loop over elements below e */
                for(ii=i; ii>=nz_min[m]; ii--) {
                    ee = (k-1)*E->lmesh.elz + ii;

                    rr = 0.5 * (E->sx[m][3][E->ien[m][ee].node[1]] +
                                E->sx[m][3][E->ien[m][ee].node[8]]);

                    /* if ee has tracers in it and is within the channel */
                    if((E->trace.ntracer_flavor[m][flavor][ee] > 0) &&
                       (rad_mean <= rr + E->viscosity.lv_channel_thickness)) {
                           F[e] = E->viscosity.lv_reduction;
                           break;
                       }
                }
            }
        }
    }


    /** debug **
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(e=1; e<=E->lmesh.nel; e++)
            fprintf(stderr, "lv_reduction: %d %e\n", e, F[e]);
    */

    return;
}


static void low_viscosity_wedge_factor(struct All_variables *E, float *F)
{
    int i, ii, k, m, e, ee;
    int nz_min[NCS], nz_max[NCS];
    const int flavor = 0;
    double rad_mean, rr;

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        /* find index of radius corresponding to lv_min_radius */
        for(e=1; e<=E->lmesh.elz; e++) {
            rad_mean = 0.5 * (E->sx[m][3][E->ien[m][e].node[1]] +
                              E->sx[m][3][E->ien[m][e].node[8]]);
            if(rad_mean >= E->viscosity.lv_min_radius) break;
        }
        nz_min[m] = e;

        /* find index of radius corresponding to lv_max_radius */
        for(e=E->lmesh.elz; e>=1; e--) {
            rad_mean = 0.5 * (E->sx[m][3][E->ien[m][e].node[1]] +
                              E->sx[m][3][E->ien[m][e].node[8]]);
            if(rad_mean <= E->viscosity.lv_max_radius) break;
        }
        nz_max[m] = e;
    }



    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(k=1; k<=E->lmesh.elx*E->lmesh.ely; k++) {
            for(i=nz_min[m]; i<=nz_max[m]; i++) {
                e = (k-1)*E->lmesh.elz + i;

                rad_mean = 0.5 * (E->sx[m][3][E->ien[m][e].node[1]] +
                                  E->sx[m][3][E->ien[m][e].node[8]]);

                /* loop over elements below e */
                for(ii=i; ii>=nz_min[m]; ii--) {
                    ee = (k-1)*E->lmesh.elz + ii;

                    /* if ee has tracers in it */
                    if(E->trace.ntracer_flavor[m][flavor][ee] > 0) {
                        F[e] = E->viscosity.lv_reduction;
                        break;
                    }
                }
            }
        }
    }


    /** debug **
    for(m=1; m<=E->sphere.caps_per_proc; m++)
        for(e=1; e<=E->lmesh.nel; e++)
            fprintf(stderr, "lv_reduction: %d %e\n", e, F[e]);
    */

    return;
}
/* compute second invariant from a strain-rate tensor in 0,...2 format

 */
double second_invariant_from_3x3(double e[3][3])
{
  return(sqrt(0.5*
	      (e[0][0] * e[0][0] + 
	       e[0][1] * e[0][1] * 2.0 + 
	       e[1][1] * e[1][1] + 
	       e[1][2] * e[1][2] * 2.0 + 
	       e[2][2] * e[2][2] + 
	       e[0][2] * e[0][2] * 2.0)));
}
void calc_strain_from_vgm(double l[3][3], double d[3][3])
{
  int i,j;
  for(i=0;i < 3;i++)
    for(j=0;j < 3;j++)
      d[i][j] = 0.5 * (l[i][j] + l[j][i]);
}
void calc_strain_from_vgm9(double *l9, double d[3][3])
{
  double l[3][3];
  get_3x3_from_9vec(l, l9);
  calc_strain_from_vgm(l, d);
}

/* 

   given a 3x3 velocity gradient matrix l, compute a rotation matrix

*/

void calc_rot_from_vgm(double l[3][3], double r[3][3])
{
  int i,j;
  for(i=0;i < 3;i++)
    for(j=0;j < 3;j++)
      r[i][j] = 0.5 * (l[i][j] - l[j][i]);
}

/* 

   get velocity gradient matrix at element, and also compute the
   average velocity in this element
   

*/

void get_vgm_p(double VV[4][9],struct Shape_function *N,
	       struct Shape_function_dx *GNx,
	       struct CC *cc, struct CCX *ccx, double rtf[4][9],
	       int dims,int ppts, int ends, int spherical,
	       double l[3][3], double v[3])
{

  int i,k,j,a;
  double ra[9], si[9], ct[9];
  const double one = 1.0;
  const double two = 2.0;
  double vgm[3][3];
  double shp, cc1, cc2, cc3,d_t,d_r,d_p,up,ur,ut;
  /* init L matrix */
  for(i=0;i < 3;i++){
    v[i] = 0.0;
    for(j=0;j < 3; j++)
      l[i][j] = 0.0;
  }
  /* mean velocity at pressure integration point */
  for(a=1;a <= ends;a++){
    v[0] += N->ppt[GNPINDEX(a, 1)] * VV[1][a];
    v[1] += N->ppt[GNPINDEX(a, 1)] * VV[2][a];
    v[2] += N->ppt[GNPINDEX(a, 1)] * VV[3][a];
  }
  if(spherical){
    for(k = 1; k <= ppts; k++){
      ra[k] = rtf[3][k];	      /* 1/r */
      si[k] = one / sin(rtf[1][k]); /* 1/sin(t) */
      ct[k] = cos(rtf[1][k]) * si[k]; /* cot(t) */
    }
    for(a = 1; a <= ends; a++){
      for(k = 1; k <= ppts; k++){
	d_t = GNx->ppt[GNPXINDEX(0, a, k)]; /* d_t */
	d_p = GNx->ppt[GNPXINDEX(1, a, k)]; /* d_p */
	d_r = GNx->ppt[GNPXINDEX(2, a, k)]; /* d_r */
	shp = N->ppt[GNPINDEX(a, k)];
	for(i = 1; i <= dims; i++){
	  ut = cc->ppt[BPINDEX(1, i, a, k)]; /* ut */
	  up = cc->ppt[BPINDEX(2, i, a, k)]; /* up */
	  ur = cc->ppt[BPINDEX(3, i, a, k)]; /* ur */
	  
	  /* velocity gradient matrix is transpose of grad v, using Citcom sort t, p, r
	
	     | d_t(vt) d_p(vt) d_r(vt) |
	     | d_t(vp) d_p(vp) d_r(vp) |
	     | d_t(vr) d_p(vr) d_r(vr) |

	  */

	  /* d_t vt = 1/r (d_t vt + vr) */
	  vgm[0][0] =  ((d_t * ut + shp * ccx->ppt[BPXINDEX(1, i, 1, a, k)]) + 
			shp * ur) * ra[k];
	  /* d_p vt = 1/r (1/sin(t) d_p vt -vp/tan(t)) */
	  vgm[0][1] =  ((d_p * ut + shp * ccx->ppt[BPXINDEX(1, i, 2, a, k)]) * si[k] - 
			shp * up * ct[k]) * ra[k];
	  /* d_r vt = d_r v_t */
	  vgm[0][2] = d_r * ut;
	  /* d_t vp = 1/r d_t v_p*/
	  vgm[1][0] = (d_t * up + shp * ccx->ppt[BPXINDEX(2, i, 1, a, k)]) * ra[k];
	  /* d_p vp = 1/r((d_p vp)/sin(t) + vt/tan(t) + vr) */
	  vgm[1][1] = ((d_p * up + shp * ccx->ppt[BPXINDEX(2, i, 2, a, k)]) * si[k] + 
		       shp * ut * ct[k] + shp * ur) * ra[k];
	  /* d_r vp = d_r v_p */
	  vgm[1][2] =  d_r * up;
	  /* d_t vr = 1/r(d_t vr - vt) */
	  vgm[2][0] = ((d_t * ur + shp * ccx->ppt[BPXINDEX(3, i, 1, a, k)]) -
		       shp * ut) * ra[k];
	  /* d_p vr =  1/r(1/sin(t) d_p vr - vp) */
	  vgm[2][1] = (( d_p * ur + shp * ccx->ppt[BPXINDEX(3,i, 2,a,k)] ) * si[k] -
		       shp * up ) * ra[k];
	  /* d_r vr = d_r vr */
	  vgm[2][2] = d_r * ur;


	  l[0][0] += vgm[0][0] * VV[i][a];
	  l[0][1] += vgm[0][1] * VV[i][a];
	  l[0][2] += vgm[0][2] * VV[1][a];
	  
	  l[1][0] += vgm[1][0] * VV[i][a];
	  l[1][1] += vgm[1][1] * VV[i][a];
	  l[1][2] += vgm[1][2] * VV[i][a];
	  
	  l[2][0] += vgm[2][0] * VV[i][a];
	  l[2][1] += vgm[2][1] * VV[i][a];
	  l[2][2] += vgm[2][2] * VV[i][a];
	  
	}
      }
    }
  }else{		
    /* cartesian */
    for(k = 1; k <= ppts; k++){
      for(a = 1; a <= ends; a++){
	/* velocity gradient matrix is transpose of grad v
	
	     | d_x(vx) d_y(vx) d_z(vx) |
	     | d_x(vy) d_y(vy) d_z(vy) |
	     | d_x(vz) d_y(vz) d_z(vz) |
	*/
	l[0][0] += GNx->ppt[GNPXINDEX(0, a, k)] * VV[1][a]; /* other contributions are zero */
	l[0][1] += GNx->ppt[GNPXINDEX(1, a, k)] * VV[1][a];
	l[0][2] += GNx->ppt[GNPXINDEX(2, a, k)] * VV[1][a];

	l[1][0] += GNx->ppt[GNPXINDEX(0, a, k)] * VV[2][a];
	l[1][1] += GNx->ppt[GNPXINDEX(1, a, k)] * VV[2][a];
	l[1][2] += GNx->ppt[GNPXINDEX(2, a, k)] * VV[2][a];

	l[2][0] += GNx->ppt[GNPXINDEX(0, a, k)] * VV[3][a];
	l[2][1] += GNx->ppt[GNPXINDEX(1, a, k)] * VV[3][a];
	l[2][2] += GNx->ppt[GNPXINDEX(2, a, k)] * VV[3][a];

      }
    }
  }
  if(ppts != 1){
    for(i=0;i<3;i++)
      for(j=0;j<3;j++)
	l[i][j] /= (float)ppts;
  }

}


