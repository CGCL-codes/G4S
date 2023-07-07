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
#include <math.h>
#include <sys/types.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "drive_solvers.h"

double global_vdot();
double vnorm_nonnewt();
int need_visc_update(struct All_variables *);
int need_to_iterate(struct All_variables *);


/************************************************************/

void general_stokes_solver_setup(struct All_variables *E)
{
  int i, m;
  void construct_node_maps();

  if (E->control.NMULTIGRID || E->control.NASSEMBLE)
    construct_node_maps(E);
  else
    for (i=E->mesh.gridmin;i<=E->mesh.gridmax;i++)
      for (m=1;m<=E->sphere.caps_per_proc;m++){
	      E->elt_k[i][m]=(double **)malloc((E->lmesh.NEL[i]+1)*sizeof(double *));
        for(int j = 0; j < E->lmesh.NEL[i]+1; ++j){
          E->elt_k[i][m][j] = (double *)malloc(24 * 24 * sizeof(double));
        }
      }

  return;
}




void general_stokes_solver(struct All_variables *E)
{
  void solve_constrained_flow_iterative();
  void construct_stiffness_B_matrix();
  void velocities_conform_bcs();
  void assemble_forces();
  void sphere_harmonics_layer();
  void get_system_viscosity();
  void remove_rigid_rot();

  double Udot_mag, dUdot_mag;
  int m,i;

  double *oldU[NCS], *delta_U[NCS];

  const int neq = E->lmesh.neq;

  E->monitor.visc_iter_count = 0; /* first solution */

  velocities_conform_bcs(E,E->U);

  assemble_forces(E,0);
  if(need_visc_update(E)){
    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
    construct_stiffness_B_matrix(E);
  } 
  
  solve_constrained_flow_iterative(E);

  
  if (need_to_iterate(E)) {
    /* outer iterations for velocity dependent viscosity */

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
      // double tmp_1[neq]={0};
      delta_U[m] = (double *)malloc(neq*sizeof(double));
      oldU[m] = (double *)malloc(neq*sizeof(double));
      // oldU[m] = &tmp_1;
      for(i=0;i<neq;i++)
        oldU[m][i]=0.0;
    }

    Udot_mag=dUdot_mag=0.0;

    E->monitor.visc_iter_count++;
    while (1) {    
     

      for (m=1;m<=E->sphere.caps_per_proc;m++)
	for (i=0;i<neq;i++) {
	  delta_U[m][i] = E->U[m][i] - oldU[m][i];
	  oldU[m][i] = E->U[m][i];
	}

      Udot_mag  = sqrt(global_vdot(E,oldU,oldU,E->mesh.levmax));
      dUdot_mag = vnorm_nonnewt(E,delta_U,oldU,E->mesh.levmax);


      if(E->parallel.me==0){
	fprintf(stderr,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
		dUdot_mag,Udot_mag,E->monitor.visc_iter_count);
	fprintf(E->fp,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
		dUdot_mag,Udot_mag,E->monitor.visc_iter_count);
	fflush(E->fp);
      }
      if ((E->monitor.visc_iter_count > 50) || 
	  (dUdot_mag < E->viscosity.sdepv_misfit))
	break;
      
      get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
      construct_stiffness_B_matrix(E);
      solve_constrained_flow_iterative(E);
      
      E->monitor.visc_iter_count++;

    } /*end while*/

    for (m=1;m<=E->sphere.caps_per_proc;m++)  {
      free((void *) oldU[m]);
      free((void *) delta_U[m]);
    }

  } /*end if we need iterations */

  /* remove the rigid rotation component from the velocity solution */
  if((E->sphere.caps == 12) &&
     (E->control.remove_rigid_rotation || E->control.remove_angular_momentum)) {
      remove_rigid_rot(E);
  }

  return;
}

int need_visc_update(struct All_variables *E)
{
  if(E->viscosity.update_allowed){
    /* always update */
    return 1;
  }else{
    /* deal with first time called */
    if(E->control.restart){	
      /* restart step - when this function is called, the cycle has
	 already been incremented */
      if(E->monitor.solution_cycles ==  E->monitor.solution_cycles_init + 1)
	return 1;
      else
	return 0;
    }else{
      /* regular step */
      if(E->monitor.solution_cycles == 0)
	return 1;
      else
	return 0;
    }
  }
}
int need_to_iterate(struct All_variables *E){
  if((E->control.force_iteration) && (E->monitor.solution_cycles == 0)){
    return 1;
  }else{
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  /* anisotropic viscosity */
  if(E->viscosity.allow_anisotropic_viscosity){
    if(E->viscosity.anivisc_start_from_iso) /* first step will be
					       solved isotropically at
					       first  */
      return 1;
    else
      return (E->viscosity.SDEPV || E->viscosity.PDEPV)?(1):(0);
  }else{
#endif
  /* regular operation */
  return ((E->viscosity.SDEPV || E->viscosity.PDEPV)?(1):(0));
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
  }
#endif
  }
}
void general_stokes_solver_pseudo_surf(struct All_variables *E)
{
  void solve_constrained_flow_iterative();
  void construct_stiffness_B_matrix();
  void velocities_conform_bcs();
  void assemble_forces();
  void get_system_viscosity();
  void std_timestep();
  void remove_rigid_rot();
  void get_STD_freesurf(struct All_variables *, float**);

  double Udot_mag, dUdot_mag;
  int m,count,i;

  double *oldU[NCS], *delta_U[NCS];

  const int neq = E->lmesh.neq;

  velocities_conform_bcs(E,E->U);

  E->monitor.stop_topo_loop = 0;
  E->monitor.topo_loop = 0;
  if(E->monitor.solution_cycles==0) std_timestep(E);
  while(E->monitor.stop_topo_loop == 0) {
	  assemble_forces(E,0);
	  if(need_visc_update(E)){
	    get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
	    construct_stiffness_B_matrix(E);
	  }
	  solve_constrained_flow_iterative(E);

	  if (E->viscosity.SDEPV || E->viscosity.PDEPV) {

		  for (m=1;m<=E->sphere.caps_per_proc;m++)  {
			  delta_U[m] = (double *)malloc(neq*sizeof(double));
			  oldU[m] = (double *)malloc(neq*sizeof(double));
			  for(i=0;i<neq;i++)
				  oldU[m][i]=0.0;
		  }

		  Udot_mag=dUdot_mag=0.0;
		  count=1;

		  while (1) {

			  for (m=1;m<=E->sphere.caps_per_proc;m++)
				  for (i=0;i<neq;i++) {
					  delta_U[m][i] = E->U[m][i] - oldU[m][i];
					  oldU[m][i] = E->U[m][i];
				  }

			  Udot_mag  = sqrt(global_vdot(E,oldU,oldU,E->mesh.levmax));
			  dUdot_mag = vnorm_nonnewt(E,delta_U,oldU,E->mesh.levmax);

			  if(E->parallel.me==0){
                              fprintf(stderr,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
                                      dUdot_mag,Udot_mag,count);
                              fprintf(E->fp,"Stress dep. visc./plast.: DUdot = %.4e (%.4e) for iteration %d\n",
                                      dUdot_mag,Udot_mag,count);
				  fflush(E->fp);
			  }

			  if (count>50 || dUdot_mag<E->viscosity.sdepv_misfit)
				  break;

			  get_system_viscosity(E,1,E->EVI[E->mesh.levmax],E->VI[E->mesh.levmax]);
			  construct_stiffness_B_matrix(E);
			  solve_constrained_flow_iterative(E);

			  count++;

		  } /*end while */
		  for (m=1;m<=E->sphere.caps_per_proc;m++)  {
			  free((void *) oldU[m]);
			  free((void *) delta_U[m]);
		  }

	  } /*end if SDEPV or PDEPV */
	  E->monitor.topo_loop++;
  }

  /* remove the rigid rotation component from the velocity solution */
  if((E->sphere.caps == 12) &&
     (E->control.remove_rigid_rotation || E->control.remove_angular_momentum)) {
      remove_rigid_rot(E);
  }

  get_STD_freesurf(E,E->slice.freesurf);

  return;
}
