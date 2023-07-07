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
/* Functions to assemble the element k matrices and the element f vector.
   Note that for the regular grid case the calculation of k becomes repetitive
   to the point of redundancy. */

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "material_properties.h"
#include <immintrin.h>

#define AVX512_DOU __m512d
#define AVX512_INT __m512i

double _mm512_reduce_add_pd(AVX512_DOU ans)
{
    AVX512_INT t1 = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
    AVX512_INT t2 = _mm512_setr_epi64(2, 3, 0, 1, 0, 1, 2, 3);
    AVX512_INT t3 = _mm512_setr_epi64(1, 0, 6, 7, 0, 1, 2, 3);
    ans = _mm512_add_pd(_mm512_permutexvar_pd(t1, ans), ans);
    ans = _mm512_add_pd(_mm512_permutexvar_pd(t2, ans), ans);
    ans = _mm512_add_pd(_mm512_permutexvar_pd(t3, ans), ans);
    return ans[0];
}


/* else, PGI would complain */
void construct_side_c3x3matrix_el(struct All_variables *,int ,
				  struct CC *,struct CCX *,
				  int ,int ,int ,int );
void construct_c3x3matrix(struct All_variables *);
void construct_c3x3matrix_el (struct All_variables *,int ,struct CC *,
			      struct CCX *,int ,int ,int );
void assemble_div_u(struct All_variables *,
                    double **, double **, int );
static void get_elt_tr(struct All_variables *, int , int , double [24], int );
static void get_elt_tr_pseudo_surf(struct All_variables *, int , int , double [24], int );

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
#include "anisotropic_viscosity.h"
#endif

static void add_force(struct All_variables *E, int e, double elt_f[24], int m)
{
  const int dims=E->mesh.nsd;
  const int ends=enodes[E->mesh.nsd];
  int a, a1, a2, a3, p, node;

  for(a=1;a<=ends;a++)          {
    node = E->ien[m][e].node[a];
    p=(a-1)*dims;
    a1=E->id[m][node].doff[1];
    E->F[m][a1] += elt_f[p];
    a2=E->id[m][node].doff[2];
    E->F[m][a2] += elt_f[p+1];
    a3=E->id[m][node].doff[3];
    E->F[m][a3] += elt_f[p+2];
  }
}



/* ================================================================
   Function to assemble the global  F vector.
                     +
   Function to get the global H vector (mixed method driving terms)
   ================================================================ */

void assemble_forces(E,penalty)
     struct All_variables *E;
     int penalty;
{
  double elt_f[24];
  int m,a,e,i;

  void get_buoyancy();
  void get_elt_f();
  void strip_bcs_from_residual();
  double global_vdot();

  const int neq=E->lmesh.neq;
  const int nel=E->lmesh.nel;
  const int lev=E->mesh.levmax;

  get_buoyancy(E,E->buoyancy);

  for(m=1;m<=E->sphere.caps_per_proc;m++)    {

    for(a=0;a<neq;a++)
      E->F[m][a] = 0.0;

    for (e=1;e<=nel;e++)  {
      get_elt_f(E,e,elt_f,1,m);
      add_force(E, e, elt_f, m);
    }

    /* for traction bc */
    for(i=1; i<=E->boundary.nel; i++) {
      e = E->boundary.element[m][i];

      for(a=0;a<24;a++) elt_f[a] = 0.0;
      for(a=SIDE_BEGIN; a<=SIDE_END; a++) {
          if(E->control.pseudo_free_surf)
              get_elt_tr_pseudo_surf(E, i, a, elt_f, m);
          else
              get_elt_tr(E, i, a, elt_f, m);
      }
      add_force(E, e, elt_f, m);
    }
  }       /* end for m */

  (E->solver.exchange_id_d)(E, E->F, lev);
  strip_bcs_from_residual(E,E->F,lev);

  /* compute the norm of E->F */
  E->monitor.fdotf = sqrt(global_vdot(E, E->F, E->F, lev));

  if(E->parallel.me==0) {
      fprintf(stderr, "Momentum equation force %.9e\n",
              E->monitor.fdotf);
      fprintf(E->fp, "Momentum equation force %.9e\n",
              E->monitor.fdotf);
  }

  return;
}


/*==============================================================
  Function to supply the element strain-displacement matrix Ba at velocity
  quadrature points, which is used to compute element stiffness matrix
  ==============================================================  */

void get_ba(struct Shape_function *N, struct Shape_function_dx *GNx,
       struct CC *cc, struct CCX *ccx, double rtf[4][9],
       int dims, double ba[9][9][4][7])
{
    int k, a, n;
    const int vpts = VPOINTS3D;
    const int ends = ENODES3D;

    double ra[9], isi[9], ct[9];
    double gnx0, gnx1, gnx2, shp, cc1, cc2, cc3;

    for(k=1;k<=vpts;k++) {
    ra[k] = rtf[3][k];
    isi[k] = 1.0 / sin(rtf[1][k]);
    ct[k] = cos(rtf[1][k]) * isi[k];
    }

    for(a=1;a<=ends;a++)
        for(k=1;k<=vpts;k++) {
            gnx0 = GNx->vpt[GNVXINDEX(0,a,k)];
            gnx1 = GNx->vpt[GNVXINDEX(1,a,k)];
            gnx2 = GNx->vpt[GNVXINDEX(2,a,k)];
            shp  = N->vpt[GNVINDEX(a,k)];
            for(n=1;n<=dims;n++) {
                cc1 = cc->vpt[BVINDEX(1,n,a,k)];
                cc2 = cc->vpt[BVINDEX(2,n,a,k)];
                cc3 = cc->vpt[BVINDEX(3,n,a,k)];

        ba[a][k][n][1] = ( gnx0 * cc1
                                   + shp * ccx->vpt[BVXINDEX(1,n,1,a,k)]
                                   + shp * cc3 ) * ra[k];

        ba[a][k][n][2] = ( shp * cc1 * ct[k]
                                   + shp * cc3
                                   + ( gnx1 * cc2
                                       + shp * ccx->vpt[BVXINDEX(2,n,2,a,k)] )
                                   * isi[k] ) * ra[k];

        ba[a][k][n][3] = gnx2 * cc3;

        ba[a][k][n][4] = ( gnx0 * cc2
                                   + shp * ccx->vpt[BVXINDEX(2,n,1,a,k)]
                                   - shp * cc2 * ct[k]
                                   + ( gnx1 * cc1
                                       + shp * ccx->vpt[BVXINDEX(1,n,2,a,k)] )
                                   * isi[k] ) * ra[k];

        ba[a][k][n][5] = gnx2 * cc1
                               + ( gnx0 * cc3
                                   + shp * ( ccx->vpt[BVXINDEX(3,n,1,a,k)]
                                             - cc1 ) ) * ra[k];

        ba[a][k][n][6] = gnx2 * cc2
                               - ra[k] * shp * cc2
                               + ( gnx1 * cc3
                                   + shp * ccx->vpt[BVXINDEX(3,n,2,a,k)] )
                               * isi[k] * ra[k];
        }
        }

    return;
}


/*==============================================================
  Function to supply the element strain-displacement matrix Ba at pressure
  quadrature points, which is used to compute strain rate
  ==============================================================  */

void get_ba_p(struct Shape_function *N, struct Shape_function_dx *GNx,
              struct CC *cc, struct CCX *ccx, double rtf[4][9],
              int dims, double ba[9][9][4][7])
{
    int k, a, n;
    const int ppts = PPOINTS3D;
    const int ends = ENODES3D;

    double ra[9], isi[9], ct[9];
    double gnx0, gnx1, gnx2, shp, cc1, cc2, cc3;

    for(k=1;k<=ppts;k++) {
        ra[k] = rtf[3][k];
        isi[k] = 1.0 / sin(rtf[1][k]);
        ct[k] = cos(rtf[1][k]) * isi[k];
    }

    for(k=1;k<=ppts;k++)
        for(a=1;a<=ends;a++) {
            gnx0 = GNx->ppt[GNPXINDEX(0,a,k)];
            gnx1 = GNx->ppt[GNPXINDEX(1,a,k)];
            gnx2 = GNx->ppt[GNPXINDEX(2,a,k)];
            shp  = N->ppt[GNPINDEX(a,k)];
            for(n=1;n<=dims;n++) {
                cc1 = cc->ppt[BPINDEX(1,n,a,k)];
                cc2 = cc->ppt[BPINDEX(2,n,a,k)];
                cc3 = cc->ppt[BPINDEX(3,n,a,k)];

        ba[a][k][n][1] = ( gnx0 * cc1
                           + shp * ccx->ppt[BPXINDEX(1,n,1,a,k)]
                           + shp * cc3 ) * ra[k];

        ba[a][k][n][2] = ( shp * cc1 * ct[k]
                           + shp * cc3
                           + ( gnx1 * cc2
                               + shp * ccx->ppt[BPXINDEX(2,n,2,a,k)] )
                           * isi[k] ) * ra[k];

        ba[a][k][n][3] = gnx2 * cc3;

        ba[a][k][n][4] = ( gnx0 * cc2
                           + shp * ccx->ppt[BPXINDEX(2,n,1,a,k)]
                           - shp * cc2 * ct[k]
                           + ( gnx1 * cc1
                               + shp * ccx->ppt[BPXINDEX(1,n,2,a,k)] )
                           * isi[k] ) * ra[k];

        ba[a][k][n][5] = gnx2 * cc1
                       + ( gnx0 * cc3
                           + shp * ( ccx->ppt[BPXINDEX(3,n,1,a,k)]
                                     - cc1 ) ) * ra[k];

        ba[a][k][n][6] = gnx2 * cc2
                       - ra[k] * shp * cc2
                       + ( gnx1 * cc3
                           + shp * ccx->ppt[BPXINDEX(3,n,2,a,k)] )
                       * isi[k] * ra[k];
            }
        }
    return;
}



/*==============================================================
  Function to supply the element k matrix for a given element e.
  ==============================================================  */

void get_elt_k(E,el,elt_k,lev,m,iconv)
     struct All_variables *E;
     int el,m;
     double elt_k[24*24];
     int lev, iconv;
{
    double bdbmu[4][4];
    int pn,qn,ad,bd,off;

    int a,b,i,j,i1,j1,k;
    double rtf[4][9],W[9];

    const double two = 2.0;
    const double two_thirds = 2.0/3.0;

    void get_rtf_at_vpts();

    double ba[9][9][4][7]; /* integration points,node,3x6 matrix */

    const int nn=loc_mat_size[E->mesh.nsd];
    const int vpts = VPOINTS3D;
    const int ends = ENODES3D;
    const int dims=E->mesh.nsd;

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
    double D[VPOINTS3D+1][6][6],btmp[6];
    int l1,l2;
#endif

    get_rtf_at_vpts(E, m, lev, el, rtf);

    if (iconv || (el-1)%E->lmesh.ELZ[lev]==0)
      construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,lev,m,0);
    
    /* Note N[a].gauss_pt[n] is the value of shape fn a at the nth gaussian
       quadrature point. Nx[d] is the derivative wrt x[d]. */
    
    for(k=1;k<=vpts;k++) {
      off = (el-1)*vpts+k;
      W[k]=g_point[k].weight[dims-1]*E->GDA[lev][m][el].vpt[k]*E->EVI[lev][m][off];
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
      if(E->viscosity.allow_anisotropic_viscosity){
	/* allow for a possibly anisotropic viscosity */
	get_constitutive(D[k],rtf[1][k],rtf[2][k],TRUE,
			 E->EVIn1[lev][m][off], E->EVIn2[lev][m][off], 
			 E->EVIn3[lev][m][off],
			 E->EVI2[lev][m][off],E->avmode[lev][m][off],
			 E);
      }
#endif
    }
    /*  */
    get_ba(&(E->N), &(E->GNX[lev][m][el]), &E->element_Cc, &E->element_Ccx,
           rtf, E->mesh.nsd, ba);

    for(a=1;a<=ends;a++)	/* loop over element nodes */
      for(b=a;b<=ends;b++)   {

	bdbmu[1][1]=bdbmu[1][2]=bdbmu[1][3]=
	  bdbmu[2][1]=bdbmu[2][2]=bdbmu[2][3]=
	  bdbmu[3][1]=bdbmu[3][2]=bdbmu[3][3]=0.0;

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
	if(E->viscosity.allow_anisotropic_viscosity){
	  for(i=1;i<=dims;i++)
	    for(j=1;j<=dims;j++)
	      for(k=1;k<=vpts;k++){ /*  */

		/* note that D is in 0,...,N-1 convention */
		for(l1=0;l1 < 6;l1++) /* compute D*B */
		  for(btmp[l1]=0.0,l2=0;l2<6;l2++)
		    btmp[l1] += D[k][l1][l2] * ba[b][k][j][l2+1];
		/* compute B^T (D*B) */
		bdbmu[i][j] += W[k] * ( ba[a][k][i][1]*btmp[0]+ba[a][k][i][2]*btmp[1]+ba[a][k][i][3]*btmp[2]+
					ba[a][k][i][4]*btmp[3]+ba[a][k][i][5]*btmp[4]+ba[a][k][i][6]*btmp[5]);
	      }
	  if(E->control.inv_gruneisen != 0)
	    for(i=1;i<=dims;i++)
	      for(j=1;j<=dims;j++)
		for(k=1;k<=VPOINTS3D;k++)
		  bdbmu[i][j] -= W[k] * two_thirds *
		    ( ba[a][k][i][1] + ba[a][k][i][2] + ba[a][k][i][3] ) *
		    ( ba[b][k][j][1] + ba[b][k][j][2] + ba[b][k][j][3] );
	}else{
#endif	/* isotropic branch */
	  for(i=1;i<=dims;i++)
	    for(j=1;j<=dims;j++)
	      for(k=1;k<=VPOINTS3D;k++)
		bdbmu[i][j] += W[k] * ( two * ( ba[a][k][i][1]*ba[b][k][j][1] +
						ba[a][k][i][2]*ba[b][k][j][2] +
						ba[a][k][i][3]*ba[b][k][j][3] ) +
					ba[a][k][i][4]*ba[b][k][j][4] +
					ba[a][k][i][5]*ba[b][k][j][5] +
					ba[a][k][i][6]*ba[b][k][j][6] );
	  
	  if(E->control.inv_gruneisen != 0)
	    for(i=1;i<=dims;i++)
	      for(j=1;j<=dims;j++)
		for(k=1;k<=VPOINTS3D;k++)
		  bdbmu[i][j] -= W[k] * two_thirds *
		    ( ba[a][k][i][1] + ba[a][k][i][2] + ba[a][k][i][3] ) *
		    ( ba[b][k][j][1] + ba[b][k][j][2] + ba[b][k][j][3] );
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
	}
#endif
	
	/**/
	ad=dims*(a-1);
	bd=dims*(b-1);
	
	pn=ad*nn+bd;
	qn=bd*nn+ad;
	
	elt_k[pn       ] = bdbmu[1][1] ; /* above */
	elt_k[pn+1     ] = bdbmu[1][2] ;
	elt_k[pn+2     ] = bdbmu[1][3] ;
	elt_k[pn+nn    ] = bdbmu[2][1] ;
	elt_k[pn+nn+1  ] = bdbmu[2][2] ;
	elt_k[pn+nn+2  ] = bdbmu[2][3] ;
	elt_k[pn+2*nn  ] = bdbmu[3][1] ;
	elt_k[pn+2*nn+1] = bdbmu[3][2] ;
	elt_k[pn+2*nn+2] = bdbmu[3][3] ;
	
	elt_k[qn       ] = bdbmu[1][1] ; /* below diag */
	elt_k[qn+1     ] = bdbmu[2][1] ;
	elt_k[qn+2     ] = bdbmu[3][1] ;
	elt_k[qn+nn    ] = bdbmu[1][2] ;
	elt_k[qn+nn+1  ] = bdbmu[2][2] ;
	elt_k[qn+nn+2  ] = bdbmu[3][2] ;
	elt_k[qn+2*nn  ] = bdbmu[1][3] ;
	elt_k[qn+2*nn+1] = bdbmu[2][3] ;
	elt_k[qn+2*nn+2] = bdbmu[3][3] ;
	/**/
	
      } /*  Sum over all the a,b's to obtain full  elt_k matrix */
    
    return;
}


/* =============================================
   General calling function for del_squared:
   according to whether it should be element by
   element or node by node.
   ============================================= */

void assemble_del2_u(E,u,Au,level,strip_bcs)
     struct All_variables *E;
     double **u,**Au;
     int level;
     int strip_bcs;
{
  void e_assemble_del2_u();
  void n_assemble_del2_u();

  if(E->control.NMULTIGRID||E->control.NASSEMBLE)
    n_assemble_del2_u(E,u,Au,level,strip_bcs);
  else
    e_assemble_del2_u(E,u,Au,level,strip_bcs);

  return;
}















/* ======================================
   Assemble del_squared_u vector el by el
   ======================================   */

void e_assemble_del2_u(E,u,Au,level,strip_bcs)
  struct All_variables *E;
  double **u,**Au;
  int level;
  int strip_bcs;
{
  int  e,i,a,b,a1,a2,a3,ii,m,nodeb;
  void strip_bcs_from_residual();

  const int n=loc_mat_size[E->mesh.nsd];
  const int ends=enodes[E->mesh.nsd];
  const int dims=E->mesh.nsd;
  const int nel=E->lmesh.NEL[level];
  const int neq=E->lmesh.NEQ[level];

  for (m=1;m<=E->sphere.caps_per_proc;m++){
    for(i=0;i<neq;i++)
      Au[m][i] = 0.0;
    for(e=1;e<=nel;e++)   {
      int ends_bound = (ends & 7);
      for(a=1;a<=ends_bound;a++) {
	      ii = E->IEN[level][m][e].node[a];
	      a1 = E->ID[level][m][ii].doff[1];
	      a2 = E->ID[level][m][ii].doff[2];
	      a3 = E->ID[level][m][ii].doff[3];
          double Au_a1 = Au[m][a1];
          double Au_a2 = Au[m][a2];
          double Au_a3 = Au[m][a3];
	      for(b=1;b<=ends;b++) {
	        nodeb = E->IEN[level][m][e].node[b];
	        ii = (a*n+b)*dims-(dims*n+dims);

            double udoff1 = u[m][E->ID[level][m][nodeb].doff[1]];
            double udoff2 = u[m][E->ID[level][m][nodeb].doff[2]];
            double udoff3 = u[m][E->ID[level][m][nodeb].doff[3]];
            /* i=1, j=1,2 */
                    /* i=1, j=1,2,3 */
            Au_a1 +=
                     E->elt_k[level][m][e].k[ii]   * udoff1
                   + E->elt_k[level][m][e].k[ii+1] * udoff2
                   + E->elt_k[level][m][e].k[ii+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a2 +=
                    E->elt_k[level][m][e].k[ii+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a3 +=
                    E->elt_k[level][m][e].k[ii+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii+n+n+2] * udoff3;
        }/* end for loop b */
        Au[m][a1] = Au_a1;
        Au[m][a2] = Au_a2;
        Au[m][a3] = Au_a3;
      }       /* end for loop a */

      for(;a<=ends;a+=8) {
	      int ii1 = E->IEN[level][m][e].node[a];
          int ii2 = E->IEN[level][m][e].node[a+1];
          int ii3 = E->IEN[level][m][e].node[a+2];
          int ii4 = E->IEN[level][m][e].node[a+3];
          int ii5 = E->IEN[level][m][e].node[a+4];
          int ii6 = E->IEN[level][m][e].node[a+5];
          int ii7 = E->IEN[level][m][e].node[a+6];
	      int ii8 = E->IEN[level][m][e].node[a+7];

          int a11 = E->ID[level][m][ii1].doff[1];
          int a21 = E->ID[level][m][ii2].doff[1];
          int a31 = E->ID[level][m][ii3].doff[1];
          int a41 = E->ID[level][m][ii4].doff[1];
          int a51 = E->ID[level][m][ii5].doff[1];
          int a61 = E->ID[level][m][ii6].doff[1];
          int a71 = E->ID[level][m][ii7].doff[1];
          int a81 = E->ID[level][m][ii8].doff[1];

	      int a12 = E->ID[level][m][ii1].doff[2];
          int a22 = E->ID[level][m][ii2].doff[2];
          int a32 = E->ID[level][m][ii3].doff[2];
          int a42 = E->ID[level][m][ii4].doff[2];
          int a52 = E->ID[level][m][ii5].doff[2];
          int a62 = E->ID[level][m][ii6].doff[2];
          int a72 = E->ID[level][m][ii7].doff[2];
          int a82 = E->ID[level][m][ii8].doff[2];

	      int a13 = E->ID[level][m][ii1].doff[3];
          int a23 = E->ID[level][m][ii2].doff[3];
          int a33 = E->ID[level][m][ii3].doff[3];
          int a43 = E->ID[level][m][ii4].doff[3];
          int a53 = E->ID[level][m][ii5].doff[3];
          int a63 = E->ID[level][m][ii6].doff[3];
          int a73 = E->ID[level][m][ii7].doff[3];
          int a83 = E->ID[level][m][ii8].doff[3];

          double Au_a11 = Au[m][a11];
          double Au_a12 = Au[m][a12];
          double Au_a13 = Au[m][a13];

          double Au_a21 = Au[m][a21];
          double Au_a22 = Au[m][a22];
          double Au_a23 = Au[m][a23];

          double Au_a31 = Au[m][a31];
          double Au_a32 = Au[m][a32];
          double Au_a33 = Au[m][a33];

          double Au_a41 = Au[m][a41];
          double Au_a42 = Au[m][a42];
          double Au_a43 = Au[m][a43];

          double Au_a51 = Au[m][a51];
          double Au_a52 = Au[m][a52];
          double Au_a53 = Au[m][a53];

          double Au_a61 = Au[m][a61];
          double Au_a62 = Au[m][a62];
          double Au_a63 = Au[m][a63];

          double Au_a71 = Au[m][a71];
          double Au_a72 = Au[m][a72];
          double Au_a73 = Au[m][a73];

          double Au_a81 = Au[m][a81];
          double Au_a82 = Au[m][a82];
          double Au_a83 = Au[m][a83];

          for(b=1;b<=ends_bound;b++) {

	        nodeb = E->IEN[level][m][e].node[b];

            ii1 = ((a  )*n+b)*dims-(dims*n+dims);
            ii2 = ((a+1)*n+b)*dims-(dims*n+dims);
            ii3 = ((a+2)*n+b)*dims-(dims*n+dims);
            ii4 = ((a+3)*n+b)*dims-(dims*n+dims);
            ii5 = ((a+4)*n+b)*dims-(dims*n+dims);
            ii6 = ((a+5)*n+b)*dims-(dims*n+dims);
            ii7 = ((a+6)*n+b)*dims-(dims*n+dims);
            ii8 = ((a+7)*n+b)*dims-(dims*n+dims);

            double udoff1 = u[m][E->ID[level][m][nodeb].doff[1]];
            double udoff2 = u[m][E->ID[level][m][nodeb].doff[2]];
            double udoff3 = u[m][E->ID[level][m][nodeb].doff[3]];

            // 1

            /* i=1, j=1,2 */
                    /* i=1, j=1,2,3 */
            Au_a11 +=
                     E->elt_k[level][m][e].k[ii1]   * udoff1
                   + E->elt_k[level][m][e].k[ii1+1] * udoff2
                   + E->elt_k[level][m][e].k[ii1+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a12 +=
                    E->elt_k[level][m][e].k[ii1+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii1+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii1+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a13 +=
                    E->elt_k[level][m][e].k[ii1+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii1+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii1+n+n+2] * udoff3;

            // 2

            Au_a21 +=
                     E->elt_k[level][m][e].k[ii2]   * udoff1
                   + E->elt_k[level][m][e].k[ii2+1] * udoff2
                   + E->elt_k[level][m][e].k[ii2+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a22 +=
                    E->elt_k[level][m][e].k[ii2+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii2+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii2+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a23 +=
                    E->elt_k[level][m][e].k[ii2+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii2+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii2+n+n+2] * udoff3;

            // 3

            Au_a31 +=
                     E->elt_k[level][m][e].k[ii3]   * udoff1
                   + E->elt_k[level][m][e].k[ii3+1] * udoff2
                   + E->elt_k[level][m][e].k[ii3+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a32 +=
                    E->elt_k[level][m][e].k[ii3+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii3+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii3+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a33 +=
                    E->elt_k[level][m][e].k[ii3+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii3+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii3+n+n+2] * udoff3;

            // 4
            Au_a41 +=
                     E->elt_k[level][m][e].k[ii4]   * udoff1
                   + E->elt_k[level][m][e].k[ii4+1] * udoff2
                   + E->elt_k[level][m][e].k[ii4+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a42 +=
                    E->elt_k[level][m][e].k[ii4+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii4+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii4+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a43 +=
                    E->elt_k[level][m][e].k[ii4+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii4+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii4+n+n+2] * udoff3;


            // 5
            Au_a51 +=
                     E->elt_k[level][m][e].k[ii5]   * udoff1
                   + E->elt_k[level][m][e].k[ii5+1] * udoff2
                   + E->elt_k[level][m][e].k[ii5+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a52 +=
                    E->elt_k[level][m][e].k[ii5+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii5+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii5+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a53 +=
                    E->elt_k[level][m][e].k[ii5+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii5+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii5+n+n+2] * udoff3;


            // 6
            Au_a61 +=
                     E->elt_k[level][m][e].k[ii6]   * udoff1
                   + E->elt_k[level][m][e].k[ii6+1] * udoff2
                   + E->elt_k[level][m][e].k[ii6+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a62 +=
                    E->elt_k[level][m][e].k[ii6+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii6+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii6+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a63 +=
                    E->elt_k[level][m][e].k[ii6+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii6+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii6+n+n+2] * udoff3;


            // 7

            Au_a71 +=
                     E->elt_k[level][m][e].k[ii7]   * udoff1
                   + E->elt_k[level][m][e].k[ii7+1] * udoff2
                   + E->elt_k[level][m][e].k[ii7+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a72 +=
                    E->elt_k[level][m][e].k[ii7+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii7+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii7+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a73 +=
                    E->elt_k[level][m][e].k[ii7+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii7+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii7+n+n+2] * udoff3;


            // 8
            Au_a81 +=
                     E->elt_k[level][m][e].k[ii8]   * udoff1
                   + E->elt_k[level][m][e].k[ii8+1] * udoff2
                   + E->elt_k[level][m][e].k[ii8+2] * udoff3;
            /* i=2, j=1,2,3 */
            Au_a82 +=
                    E->elt_k[level][m][e].k[ii8+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii8+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii8+n+2] * udoff3;
            /* i=3, j=1,2,3 */
            Au_a83 +=
                    E->elt_k[level][m][e].k[ii8+n+n]   * udoff1
                    + E->elt_k[level][m][e].k[ii8+n+n+1] * udoff2
                    + E->elt_k[level][m][e].k[ii8+n+n+2] * udoff3;
        }/* end for loop b */

            
          AVX512_DOU vec_ans11 = _mm512_setzero_pd();
          AVX512_DOU vec_ans12 = _mm512_setzero_pd();
          AVX512_DOU vec_ans13 = _mm512_setzero_pd();
          AVX512_DOU vec_ans21 = _mm512_setzero_pd();
          AVX512_DOU vec_ans22 = _mm512_setzero_pd();
          AVX512_DOU vec_ans23 = _mm512_setzero_pd();
          AVX512_DOU vec_ans31 = _mm512_setzero_pd();
          AVX512_DOU vec_ans32 = _mm512_setzero_pd();
          AVX512_DOU vec_ans33 = _mm512_setzero_pd();
          AVX512_DOU vec_ans41 = _mm512_setzero_pd();
          AVX512_DOU vec_ans42 = _mm512_setzero_pd();
          AVX512_DOU vec_ans43 = _mm512_setzero_pd();
          AVX512_DOU vec_ans51 = _mm512_setzero_pd();
          AVX512_DOU vec_ans52 = _mm512_setzero_pd();
          AVX512_DOU vec_ans53 = _mm512_setzero_pd();
          AVX512_DOU vec_ans61 = _mm512_setzero_pd();
          AVX512_DOU vec_ans62 = _mm512_setzero_pd();
          AVX512_DOU vec_ans63 = _mm512_setzero_pd();
          AVX512_DOU vec_ans71 = _mm512_setzero_pd();
          AVX512_DOU vec_ans72 = _mm512_setzero_pd();
          AVX512_DOU vec_ans73 = _mm512_setzero_pd();
          AVX512_DOU vec_ans81 = _mm512_setzero_pd();
          AVX512_DOU vec_ans82 = _mm512_setzero_pd();
          AVX512_DOU vec_ans83 = _mm512_setzero_pd();

	      for(;b<=ends;b+=8) {

	        int nodeb1 = E->IEN[level][m][e].node[b];
            int nodeb2 = E->IEN[level][m][e].node[b+1];
            int nodeb3 = E->IEN[level][m][e].node[b+2];
            int nodeb4 = E->IEN[level][m][e].node[b+3];
            int nodeb5 = E->IEN[level][m][e].node[b+4];
            int nodeb6 = E->IEN[level][m][e].node[b+5];
            int nodeb7 = E->IEN[level][m][e].node[b+6];
            int nodeb8 = E->IEN[level][m][e].node[b+7];

            ii1 = ((a  )*n+b)*dims-(dims*n+dims);
            ii2 = ii1 + n * 3;
            ii3 = ii1 + n * 6;
            ii4 = ii1 + n * 9;
            ii5 = ii1 + n * 12;
            ii6 = ii1 + n * 15;
            ii7 = ii1 + n * 18;
            ii8 = ii1 + n * 21;

            double udoff11 = u[m][E->ID[level][m][nodeb1].doff[1]];
            double udoff12 = u[m][E->ID[level][m][nodeb1].doff[2]];
            double udoff13 = u[m][E->ID[level][m][nodeb1].doff[3]];

            double udoff21 = u[m][E->ID[level][m][nodeb2].doff[1]];
            double udoff22 = u[m][E->ID[level][m][nodeb2].doff[2]];
            double udoff23 = u[m][E->ID[level][m][nodeb2].doff[3]];

            double udoff31 = u[m][E->ID[level][m][nodeb3].doff[1]];
            double udoff32 = u[m][E->ID[level][m][nodeb3].doff[2]];
            double udoff33 = u[m][E->ID[level][m][nodeb3].doff[3]];

            double udoff41 = u[m][E->ID[level][m][nodeb4].doff[1]];
            double udoff42 = u[m][E->ID[level][m][nodeb4].doff[2]];
            double udoff43 = u[m][E->ID[level][m][nodeb4].doff[3]];

            double udoff51 = u[m][E->ID[level][m][nodeb5].doff[1]];
            double udoff52 = u[m][E->ID[level][m][nodeb5].doff[2]];
            double udoff53 = u[m][E->ID[level][m][nodeb5].doff[3]];

            double udoff61 = u[m][E->ID[level][m][nodeb6].doff[1]];
            double udoff62 = u[m][E->ID[level][m][nodeb6].doff[2]];
            double udoff63 = u[m][E->ID[level][m][nodeb6].doff[3]];

            double udoff71 = u[m][E->ID[level][m][nodeb7].doff[1]];
            double udoff72 = u[m][E->ID[level][m][nodeb7].doff[2]];
            double udoff73 = u[m][E->ID[level][m][nodeb7].doff[3]];

            double udoff81 = u[m][E->ID[level][m][nodeb8].doff[1]];
            double udoff82 = u[m][E->ID[level][m][nodeb8].doff[2]];
            double udoff83 = u[m][E->ID[level][m][nodeb8].doff[3]];

            AVX512_DOU vec_doff1 = _mm512_setr_pd(udoff11, udoff12, udoff13, udoff21, udoff22, udoff23, udoff31, udoff32);
            AVX512_DOU vec_doff2 = _mm512_setr_pd(udoff33, udoff41, udoff42, udoff43, udoff51, udoff52, udoff53, udoff61);
            AVX512_DOU vec_doff3 = _mm512_setr_pd(udoff62, udoff63, udoff71, udoff72, udoff73, udoff81, udoff82, udoff83);

            // 1

            vec_ans11 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1]), vec_doff1, vec_ans11);
            vec_ans11 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+8]), vec_doff2, vec_ans11);
            vec_ans11 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+16]), vec_doff3, vec_ans11);

            vec_ans12 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+n]), vec_doff1, vec_ans12);
            vec_ans12 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+n+8]), vec_doff2, vec_ans12);
            vec_ans12 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+n+16]), vec_doff3, vec_ans12);

            vec_ans13 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+n+n]), vec_doff1, vec_ans13);
            vec_ans13 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+n+n+8]), vec_doff2, vec_ans13);
            vec_ans13 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii1+n+n+16]), vec_doff3, vec_ans13);



            // 2

            vec_ans21 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2]), vec_doff1, vec_ans21);
            vec_ans21 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+8]), vec_doff2, vec_ans21);
            vec_ans21 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+16]), vec_doff3, vec_ans21);

            vec_ans22 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+n]), vec_doff1, vec_ans22);
            vec_ans22 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+n+8]), vec_doff2, vec_ans22);
            vec_ans22 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+n+16]), vec_doff3, vec_ans22);
            
            vec_ans23 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+n+n]), vec_doff1, vec_ans23);
            vec_ans23 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+n+n+8]), vec_doff2, vec_ans23);
            vec_ans23 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii2+n+n+16]), vec_doff3, vec_ans23);


             // 3

            vec_ans31 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3]), vec_doff1, vec_ans31);
            vec_ans31 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+8]), vec_doff2, vec_ans31);
            vec_ans31 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+16]), vec_doff3, vec_ans31);

            vec_ans32 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+n]), vec_doff1, vec_ans32);
            vec_ans32 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+n+8]), vec_doff2, vec_ans32);
            vec_ans32 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+n+16]), vec_doff3, vec_ans32);
            
            vec_ans33 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+n+n]), vec_doff1, vec_ans33);
            vec_ans33 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+n+n+8]), vec_doff2, vec_ans33);
            vec_ans33 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii3+n+n+16]), vec_doff3, vec_ans33);


             // 4

            vec_ans41 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4]), vec_doff1, vec_ans41);
            vec_ans41 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+8]), vec_doff2, vec_ans41);
            vec_ans41 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+16]), vec_doff3, vec_ans41);

            vec_ans42 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+n]), vec_doff1, vec_ans42);
            vec_ans42 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+n+8]), vec_doff2, vec_ans42);
            vec_ans42 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+n+16]), vec_doff3, vec_ans42);
            
            vec_ans43 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+n+n]), vec_doff1, vec_ans43);
            vec_ans43 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+n+n+8]), vec_doff2, vec_ans43);
            vec_ans43 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii4+n+n+16]), vec_doff3, vec_ans43);

            
             // 5

            vec_ans51 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5]), vec_doff1, vec_ans51);
            vec_ans51 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+8]), vec_doff2, vec_ans51);
            vec_ans51 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+16]), vec_doff3, vec_ans51);

            vec_ans52 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+n]), vec_doff1, vec_ans52);
            vec_ans52 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+n+8]), vec_doff2, vec_ans52);
            vec_ans52 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+n+16]), vec_doff3, vec_ans52);
            
            vec_ans53 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+n+n]), vec_doff1, vec_ans53);
            vec_ans53 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+n+n+8]), vec_doff2, vec_ans53);
            vec_ans53 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii5+n+n+16]), vec_doff3, vec_ans53);

             // 6

            vec_ans61 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6]), vec_doff1, vec_ans61);
            vec_ans61 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+8]), vec_doff2, vec_ans61);
            vec_ans61 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+16]), vec_doff3, vec_ans61);

            vec_ans62 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+n]), vec_doff1, vec_ans62);
            vec_ans62 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+n+8]), vec_doff2, vec_ans62);
            vec_ans62 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+n+16]), vec_doff3, vec_ans62);
            
            vec_ans63 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+n+n]), vec_doff1, vec_ans63);
            vec_ans63 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+n+n+8]), vec_doff2, vec_ans63);
            vec_ans63 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii6+n+n+16]), vec_doff3, vec_ans63);

             // 7

            vec_ans71 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7]), vec_doff1, vec_ans71);
            vec_ans71 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+8]), vec_doff2, vec_ans71);
            vec_ans71 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+16]), vec_doff3, vec_ans71);

            vec_ans72 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+n]), vec_doff1, vec_ans72);
            vec_ans72 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+n+8]), vec_doff2, vec_ans72);
            vec_ans72 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+n+16]), vec_doff3, vec_ans72);
            
            vec_ans73 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+n+n]), vec_doff1, vec_ans73);
            vec_ans73 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+n+n+8]), vec_doff2, vec_ans73);
            vec_ans73 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii7+n+n+16]), vec_doff3, vec_ans73);

             // 8

            vec_ans81 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8]), vec_doff1, vec_ans81);
            vec_ans81 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+8]), vec_doff2, vec_ans81);
            vec_ans81 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+16]), vec_doff3, vec_ans81);

            vec_ans82 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+n]), vec_doff1, vec_ans82);
            vec_ans82 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+n+8]), vec_doff2, vec_ans82);
            vec_ans82 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+n+16]), vec_doff3, vec_ans82);
            
            vec_ans83 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+n+n]), vec_doff1, vec_ans83);
            vec_ans83 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+n+n+8]), vec_doff2, vec_ans83);
            vec_ans83 = _mm512_fmadd_pd(_mm512_loadu_pd(&E->elt_k[level][m][e].k[ii8+n+n+16]), vec_doff3, vec_ans83);


        }/* end for loop b */
        Au[m][a11] = Au_a11 + _mm512_reduce_add_pd(vec_ans11);
        Au[m][a12] = Au_a12 + _mm512_reduce_add_pd(vec_ans12);
        Au[m][a13] = Au_a13 + _mm512_reduce_add_pd(vec_ans13);

        Au[m][a21] = Au_a21 + _mm512_reduce_add_pd(vec_ans21);
        Au[m][a22] = Au_a22 + _mm512_reduce_add_pd(vec_ans22);
        Au[m][a23] = Au_a23 + _mm512_reduce_add_pd(vec_ans23);

        Au[m][a31] = Au_a31 + _mm512_reduce_add_pd(vec_ans31);
        Au[m][a32] = Au_a32 + _mm512_reduce_add_pd(vec_ans32);
        Au[m][a33] = Au_a33 + _mm512_reduce_add_pd(vec_ans33);

        Au[m][a41] = Au_a41 + _mm512_reduce_add_pd(vec_ans41);
        Au[m][a42] = Au_a42 + _mm512_reduce_add_pd(vec_ans42);
        Au[m][a43] = Au_a43 + _mm512_reduce_add_pd(vec_ans43);

        Au[m][a51] = Au_a51 + _mm512_reduce_add_pd(vec_ans51);
        Au[m][a52] = Au_a52 + _mm512_reduce_add_pd(vec_ans52);
        Au[m][a53] = Au_a53 + _mm512_reduce_add_pd(vec_ans53);

        Au[m][a61] = Au_a61 + _mm512_reduce_add_pd(vec_ans61);
        Au[m][a62] = Au_a62 + _mm512_reduce_add_pd(vec_ans62);
        Au[m][a63] = Au_a63 + _mm512_reduce_add_pd(vec_ans63);

        Au[m][a71] = Au_a71 + _mm512_reduce_add_pd(vec_ans71);
        Au[m][a72] = Au_a72 + _mm512_reduce_add_pd(vec_ans72);
        Au[m][a73] = Au_a73 + _mm512_reduce_add_pd(vec_ans73);

        Au[m][a81] = Au_a81 + _mm512_reduce_add_pd(vec_ans81);
        Au[m][a82] = Au_a82 + _mm512_reduce_add_pd(vec_ans82);
        Au[m][a83] = Au_a83 + _mm512_reduce_add_pd(vec_ans83);
      }       /* end for loop a */
    }          /* end for e */
  }         /* end for m  */

    (E->solver.exchange_id_d)(E, Au, level);

  if(strip_bcs)
     strip_bcs_from_residual(E,Au,level);

  return; 
}























/* ======================================================
   Assemble Au using stored, nodal coefficients.
   ====================================================== */

void n_assemble_del2_u(E,u,Au,level,strip_bcs)
     struct All_variables *E;
     double **u,**Au;
     int level;
     int strip_bcs;
{
    int m, e,i;
    int eqn1,eqn2,eqn3;

    double UU,U1,U2,U3;
    void strip_bcs_from_residual();

    int *C;
    higher_precision *B1,*B2,*B3;

    const int neq=E->lmesh.NEQ[level];
    const int nno=E->lmesh.NNO[level];
    const int dims=E->mesh.nsd;
    const int max_eqn = dims*14;


  for (m=1;m<=E->sphere.caps_per_proc;m++)  {

     for(e=0;e<=neq;e++)
	Au[m][e]=0.0;

     u[m][neq] = 0.0;

     for(e=1;e<=nno;e++)     {

       eqn1=E->ID[level][m][e].doff[1];
       eqn2=E->ID[level][m][e].doff[2];
       eqn3=E->ID[level][m][e].doff[3];

       U1 = u[m][eqn1];
       U2 = u[m][eqn2];
       U3 = u[m][eqn3];

       C=E->Node_map[level][m] + (e-1)*max_eqn;
       B1=E->Eqn_k1[level][m]+(e-1)*max_eqn;
       B2=E->Eqn_k2[level][m]+(e-1)*max_eqn;
       B3=E->Eqn_k3[level][m]+(e-1)*max_eqn;

       for(i=3;i<max_eqn;i++)  {
	  UU = u[m][C[i]];
  	  Au[m][eqn1] += B1[i]*UU;
  	  Au[m][eqn2] += B2[i]*UU;
  	  Au[m][eqn3] += B3[i]*UU;
       }
       for(i=0;i<max_eqn;i++)
          Au[m][C[i]] += B1[i]*U1+B2[i]*U2+B3[i]*U3;

       }     /* end for e */
     }     /* end for m */

     (E->solver.exchange_id_d)(E, Au, level);

    if (strip_bcs)
	strip_bcs_from_residual(E,Au,level);

    return;
}


void build_diagonal_of_K(E,el,elt_k,level,m)
     struct All_variables *E;
     int level,el,m;
     double elt_k[24*24];

{
    int a,a1,a2,p,node;

    const int n=loc_mat_size[E->mesh.nsd];
    const int dims=E->mesh.nsd;
    const int ends=enodes[E->mesh.nsd];

    for(a=1;a<=ends;a++) {
	    node=E->IEN[level][m][el].node[a];
	    /* dirn 1 */
	    a1 = E->ID[level][m][node].doff[1];
	    p=(a-1)*dims;
	    E->BI[level][m][a1] += elt_k[p*n+p];

	    /* dirn 2 */
	    a2 = E->ID[level][m][node].doff[2];
	    p=(a-1)*dims+1;
	    E->BI[level][m][a2] += elt_k[p*n+p];

	    /* dirn 3 */
	    a1 = E->ID[level][m][node].doff[3];
	    p=(a-1)*dims+2;
	    E->BI[level][m][a1] += elt_k[p*n+p];
            }

  return;
}

void build_diagonal_of_Ahat(E)
    struct All_variables *E;
{
    double assemble_dAhatp_entry();

    double BU;
    int m,e,npno,level;

 for (level=E->mesh.gridmin;level<=E->mesh.gridmax;level++)

   for (m=1;m<=E->sphere.caps_per_proc;m++)    {

     npno = E->lmesh.NPNO[level];
     //neq=E->lmesh.NEQ[level];

     for(e=1;e<=npno;e++)
	E->BPI[level][m][e]=1.0;

     if(!E->control.precondition)
	return;

     for(e=1;e<=npno;e++)  {
	BU=assemble_dAhatp_entry(E,e,level,m);
	if(BU != 0.0)
	    E->BPI[level][m][e] = 1.0/BU;
	else
	    E->BPI[level][m][e] = 1.0;
        }
     }

    return;
}


/* =====================================================
   Assemble grad(rho_ref*ez)*V element by element.
   Note that the storage is not zero'd before assembling.
   =====================================================  */

void assemble_c_u(struct All_variables *E,
                  double **U, double **result, int level)
{
    int e,j1,j2,j3,p,a,b,m;

    const int nel = E->lmesh.NEL[level];
    const int ends = enodes[E->mesh.nsd];
    const int dims = E->mesh.nsd;
    const int npno = E->lmesh.NPNO[level];

    for(m=1;m<=E->sphere.caps_per_proc;m++)
        for(a=1;a<=ends;a++) {
            p = (a-1)*dims;
            for(e=1;e<=nel;e++) {
                b = E->IEN[level][m][e].node[a];
                j1= E->ID[level][m][b].doff[1];
                j2= E->ID[level][m][b].doff[2];
                j3= E->ID[level][m][b].doff[3];
                result[m][e] += E->elt_c[level][m][e].c[p  ][0] * U[m][j1]
                              + E->elt_c[level][m][e].c[p+1][0] * U[m][j2]
                              + E->elt_c[level][m][e].c[p+2][0] * U[m][j3];
            }
        }

    return;
}



/* =====================================================
   Assemble div(rho_ref*V) = div(V) + grad(rho_ref*ez)*V
   element by element
   =====================================================  */

void assemble_div_rho_u(struct All_variables *E,
                        double **U, double **result, int level)
{
    void assemble_div_u();
    assemble_div_u(E, U, result, level);
    assemble_c_u(E, U, result, level);

    return;
}


/* ==========================================
   Assemble a div_u vector element by element
   ==========================================  */

void assemble_div_u(struct All_variables *E,
                    double **U, double **divU, int level)
{
    int e,j1,j2,j3,p,a,b,m;

    const int nel=E->lmesh.NEL[level];
    const int ends=enodes[E->mesh.nsd];
    const int dims=E->mesh.nsd;
    const int npno=E->lmesh.NPNO[level];

  for(m=1;m<=E->sphere.caps_per_proc;m++)
    for(e=1;e<=npno;e++)
	divU[m][e] = 0.0;

  for(m=1;m<=E->sphere.caps_per_proc;m++)
       for(a=1;a<=ends;a++)   {
	  p = (a-1)*dims;
          for(e=1;e<=nel;e++) {
	    b = E->IEN[level][m][e].node[a];
	    j1= E->ID[level][m][b].doff[1];
            j2= E->ID[level][m][b].doff[2];
	    j3= E->ID[level][m][b].doff[3];
	    divU[m][e] += E->elt_del[level][m][e].g[p  ][0] * U[m][j1]
	                + E->elt_del[level][m][e].g[p+1][0] * U[m][j2]
	                + E->elt_del[level][m][e].g[p+2][0] * U[m][j3];
	    }
	 }

    return;
}


/* ==========================================
   Assemble a grad_P vector element by element
   ==========================================  */

void assemble_grad_p(E,P,gradP,lev)
     struct All_variables *E;
     double **P,**gradP;
     int lev;

{
  int m,e,i,j1,j2,j3,p,a,b,nel,neq;
  void strip_bcs_from_residual();

  const int ends=enodes[E->mesh.nsd];
  const int dims=E->mesh.nsd;

  for(m=1;m<=E->sphere.caps_per_proc;m++)  {

    nel=E->lmesh.NEL[lev];
    neq=E->lmesh.NEQ[lev];

    for(i=0;i<neq;i++)
      gradP[m][i] = 0.0;

    for(e=1;e<=nel;e++) {

	if(0.0==P[m][e])
	    continue;

	for(a=1;a<=ends;a++)       {
	     p = (a-1)*dims;
	     b = E->IEN[lev][m][e].node[a];
	     j1= E->ID[lev][m][b].doff[1];
	     j2= E->ID[lev][m][b].doff[2];
	     j3= E->ID[lev][m][b].doff[3];
		        /*for(b=0;b<ploc_mat_size[E->mesh.nsd];b++)  */
             gradP[m][j1] += E->elt_del[lev][m][e].g[p  ][0] * P[m][e];
             gradP[m][j2] += E->elt_del[lev][m][e].g[p+1][0] * P[m][e];
             gradP[m][j3] += E->elt_del[lev][m][e].g[p+2][0] * P[m][e];
	     }
        }       /* end for el */
     }       /* end for m */

  (E->solver.exchange_id_d)(E, gradP,  lev); /*  correct gradP   */


  strip_bcs_from_residual(E,gradP,lev);

return;
}


double assemble_dAhatp_entry(E,e,level,m)
     struct All_variables *E;
     int e,level,m;

{
  int i,j,p,a,b,node;
    void strip_bcs_from_residual();

    double gradP[81],divU;

    const int ends=enodes[E->mesh.nsd];
    const int dims=E->mesh.nsd;

    //npno=E->lmesh.NPNO[level];

    for(i=0;i<81;i++)
	gradP[i] = 0.0;

    divU=0.0;

    for(a=1;a<=ends;a++) {
      p = (a-1)*dims;
      node = E->IEN[level][m][e].node[a];
      j=E->ID[level][m][node].doff[1];
      gradP[p] += E->BI[level][m][j]*E->elt_del[level][m][e].g[p][0];

      j=E->ID[level][m][node].doff[2];
      gradP[p+1] += E->BI[level][m][j]*E->elt_del[level][m][e].g[p+1][0];

      j=E->ID[level][m][node].doff[3];
      gradP[p+2] += E->BI[level][m][j]*E->elt_del[level][m][e].g[p+2][0];
      }


    /* calculate div U from the same thing .... */

    /* only need to run over nodes with non-zero grad P, i.e. the ones in
       the element accessed above, BUT it is only necessary to update the
       value in the original element, because the diagonal is all we use at
       the end ... */

    for(b=1;b<=ends;b++) {
      p = (b-1)*dims;
      divU +=E->elt_del[level][m][e].g[p][0] * gradP[p];
      divU +=E->elt_del[level][m][e].g[p+1][0] * gradP[p+1];
      divU +=E->elt_del[level][m][e].g[p+2][0] * gradP[p+2];
      }

return(divU);  }


/*==============================================================
  Function to supply the element c matrix for a given element e.
  ==============================================================  */

void get_elt_c(struct All_variables *E, int el,
               higher_precision elt_c[24][1], int lev, int m)
{

    int p, a, i, j, nz;
    double temp, beta, rho_avg, x[4];

    double rho[9];

    const int dims = E->mesh.nsd;
    const int ends = enodes[dims];

    if ((el-1)%E->lmesh.ELZ[lev]==0)
        construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,lev,m,1);

    temp = p_point[1].weight[dims-1] * E->GDA[lev][m][el].ppt[1];

    switch (E->refstate.choice) {
    case 1:
        /* the reference state is computed by rho=exp((1-r)Di/gamma) */
        /* so d(rho)/dr/rho == -Di/gamma */

        beta = - E->control.disptn_number * E->control.inv_gruneisen;

        for(a=1;a<=ends;a++) {
            for (i=1;i<=dims;i++) {
                x[i] = E->N.ppt[GNPINDEX(a,1)]
                    * E->element_Cc.ppt[BPINDEX(3,i,a,1)];
            }
            p=dims*(a-1);
            elt_c[p  ][0] = -x[1] * temp * beta;
            elt_c[p+1][0] = -x[2] * temp * beta;
            elt_c[p+2][0] = -x[3] * temp * beta;
        }
        break;
    default:
        /* compute d(rho)/dr/rho from rho(r) */

        for(a=1;a<=ends;a++) {
            j = E->IEN[lev][m][el].node[a];
            nz = (j - 1) % E->lmesh.noz + 1;
            rho[a] = E->refstate.rho[nz];
        }

        rho_avg = 0;
        for(a=1;a<=ends;a++) {
            rho_avg += rho[a];
        }
        rho_avg /= ends;

        for(a=1;a<=ends;a++) {
            for (i=1;i<=dims;i++) {
                x[i] = rho[a] * E->GNX[lev][m][el].ppt[GNPXINDEX(2,a,1)]
                    * E->N.ppt[GNPINDEX(a,1)]
                    * E->element_Cc.ppt[BPINDEX(3,i,a,1)];
            }
            p=dims*(a-1);
            elt_c[p  ][0] = -x[1] * temp / rho_avg;
            elt_c[p+1][0] = -x[2] * temp / rho_avg;
            elt_c[p+2][0] = -x[3] * temp / rho_avg;
        }

    }

    return;
}


/*==============================================================
  Function to supply the element g matrix for a given element e.
  used for the divergence
  ==============================================================  */

void get_elt_g(E,el,elt_del,lev,m)
     struct All_variables *E;
     int el,m;
     higher_precision elt_del[24][1];
     int lev;

{
   void get_rtf_at_ppts();
   int p,a,i,j,k;
   double ra,ct,si,x[4],rtf[4][9];
   double temp;
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
   double Dtmp[6][6],Duse[6][6],rtf2[4][9],weight;
   const int vpts=VPOINTS3D;
   const int modify_g = TRUE;
   //const int modify_g = FALSE;
   int off;
   double ba[9][9][4][7];
#endif
   const int dims=E->mesh.nsd;
   const int ends=enodes[dims];
 
   /* Special case, 4/8 node bilinear cartesian square/cube element -> 1 pressure point */

   if ((el-1)%E->lmesh.ELZ[lev]==0)
      construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,lev,m,1);

   get_rtf_at_ppts(E, m, lev, el, rtf);

   temp = p_point[1].weight[dims-1] * E->GDA[lev][m][el].ppt[1];

#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
   if(E->viscosity.allow_anisotropic_viscosity && modify_g){
     /* find avg constitutive matrix from all vpts (change this later) */
     get_rtf_at_vpts(E, m, lev, el, rtf2);
     for(i=0;i<6;i++)
       for(j=0;j<6;j++)
	 Duse[i][j]=0.0;
     weight = 1./(2.*vpts);
     for(i=1;i <= vpts;i++){	/* get vag const matrix */
       off = (el-1)*vpts+i;
       get_constitutive(Dtmp,rtf2[1][i],rtf2[2][i],TRUE,
			E->EVIn1[lev][m][off], E->EVIn2[lev][m][off], E->EVIn3[lev][m][off],
			E->EVI2[lev][m][off],E->avmode[lev][m][off],
			E);
       for(j=0;j<6;j++)
	 for(k=0;k<6;k++)
	   Duse[j][k] += Dtmp[j][k]*weight;
     }
     get_ba_p(&(E->N),&(E->GNX[lev][m][el]),&E->element_Cc, &E->element_Ccx,rtf,E->mesh.nsd,ba);

     /* assume single pressure point */
     for(a = 1; a <= ends; a++){
       for(i = 1; i <= dims; i++){
	 x[i] = 0.0;
	 for(k=0;k < 6;k++){
	   x[i] += Duse[0][k] * ba[a][1][i][k+1];
	   x[i] += Duse[1][k] * ba[a][1][i][k+1];
	   x[i] += Duse[2][k] * ba[a][1][i][k+1];
	 }
       }
       p=dims*(a-1);
       elt_del[p  ][0] = -x[1] * temp;
       elt_del[p+1][0] = -x[2] * temp;
       elt_del[p+2][0] = -x[3] * temp;
       
     }
   }else{
#endif
     ra = rtf[3][1];
     si = 1.0/sin(rtf[1][1]);
     ct = cos(rtf[1][1])*si;

     /* old, isotropic part */
     for(a=1;a<=ends;a++)      {
       for (i=1;i<=dims;i++)
	 x[i] = E->GNX[lev][m][el].ppt[GNPXINDEX(2,a,1)] * E->element_Cc.ppt[BPINDEX(3,i,a,1)] + 
	   2.0 * ra * E->N.ppt[GNPINDEX(a,1)]*E->element_Cc.ppt[BPINDEX(3,i,a,1)] + 
	   ra * 
	   (E->GNX[lev][m][el].ppt[GNPXINDEX(0,a,1)]*E->element_Cc.ppt[BPINDEX(1,i,a,1)] +
	    E->N.ppt[GNPINDEX(a,1)]*E->element_Ccx.ppt[BPXINDEX(1,i,1,a,1)] +
	    ct * E->N.ppt[GNPINDEX(a,1)] * E->element_Cc.ppt[BPINDEX(1,i,a,1)] +
	    si * (E->GNX[lev][m][el].ppt[GNPXINDEX(1,a,1)] * E->element_Cc.ppt[BPINDEX(2,i,a,1)] +
		  E->N.ppt[GNPINDEX(a,1)] * E->element_Ccx.ppt[BPXINDEX(2,i,2,a,1)]));
       p=dims*(a-1);
       elt_del[p  ][0] = -x[1] * temp;
       elt_del[p+1][0] = -x[2] * temp;
       elt_del[p+2][0] = -x[3] * temp;
       
      /* fprintf (E->fp,"B= %d %d %g %g %g %g %g\n",el,a,E->GDA[lev][m][el].ppt[1],E->GNX[lev][m][el].ppt[GNPXINDEX(0,a,1)],E->GNX[lev][m][el].ppt[GNPXINDEX(1,a,1)],elt_del[p][0],elt_del[p+1][0]);
       */
     }
#ifdef CITCOM_ALLOW_ANISOTROPIC_VISC
   }
#endif
   return;
 }

/*=================================================================
  Function to create the element force vector (allowing for velocity b.c.'s)
  ================================================================= */

void get_elt_f(E,el,elt_f,bcs,m)
     struct All_variables *E;
     int el,m;
     double elt_f[24];
     int bcs;

{

  int i,p,a,b,j,k,q,nodeb;
  int got_elt_k;
  unsigned int type;
  const unsigned int vbc_flag[] = {0, VBX, VBY, VBZ};

  double force[9],force_at_gs[9],elt_k[24*24];


  const int dims=E->mesh.nsd;
  const int n=loc_mat_size[dims];
  const int ends=enodes[dims];
  const int vpts=vpoints[dims];

  //es = (el-1)/E->lmesh.elz + 1;

  if ((el-1)%E->lmesh.elz==0)
      construct_c3x3matrix_el(E,el,&E->element_Cc,&E->element_Ccx,E->mesh.levmax,m,0);

  for(p=0;p<n;p++) elt_f[p] = 0.0;

  for(p=1;p<=ends;p++)
    force[p] = E->buoyancy[m][E->ien[m][el].node[p]];

  for(j=1;j<=vpts;j++)       {   /*compute force at each int point */
    force_at_gs[j] = 0.0;
    for(k=1;k<=ends;k++)
      force_at_gs[j] += force[k] * E->N.vpt[GNVINDEX(k,j)] ;
    }

  for(i=1;i<=dims;i++)  {
    for(a=1;a<=ends;a++)  {
      //nodea=E->ien[m][el].node[a];
      p= dims*(a-1)+i-1;

      for(j=1;j<=vpts;j++)     /*compute sum(Na(j)*F(j)*det(j)) */
        elt_f[p] += force_at_gs[j] * E->N.vpt[GNVINDEX(a,j)]
           *E->gDA[m][el].vpt[j]*g_point[j].weight[dims-1]
           *E->element_Cc.vpt[BVINDEX(3,i,a,j)];

	  /* imposed velocity terms */

      if(bcs)  {
        got_elt_k = 0;
        for(j=1;j<=dims;j++) {
	  type=vbc_flag[j];
          for(b=1;b<=ends;b++) {
            nodeb=E->ien[m][el].node[b];
            if ((E->node[m][nodeb]&type)&&(E->sphere.cap[m].VB[j][nodeb]!=0.0)){
              if(!got_elt_k) {
                get_elt_k(E,el,elt_k,E->mesh.levmax,m,1);
                got_elt_k = 1;
                }
              q = dims*(b-1)+j-1;
              if(p!=q) {
                elt_f[p] -= elt_k[p*n+q] * E->sphere.cap[m].VB[j][nodeb];
                }
              }
            }  /* end for b */
          }      /* end for j */
        }      /* end if for if bcs */

      }
    } /*  Complete the loops for a,i  	*/



  return;
}


/*=================================================================
  Function to create the element force vector due to stress b.c.
  ================================================================= */

static void get_elt_tr(struct All_variables *E, int bel, int side, double elt_tr[24], int m)
{

	const int dims=E->mesh.nsd;
	const int ends1=enodes[dims-1];
	const int oned = onedvpoints[dims];

	struct CC Cc;
	struct CCX Ccx;

	const unsigned sbc_flag[4] = {0,SBX,SBY,SBZ};

	double traction[4][5],traction_at_gs[4][5], value, tmp;
	int j, b, p, k, a, nodea, d;
	int el = E->boundary.element[m][bel];
	int flagged;
	int found = 0;

	const float rho = E->data.density;
	const float g = E->data.grav_acc;
	const float R = 6371000.0;
	const float eta = E->data.ref_viscosity;
	const float kappa = E->data.therm_diff;
	const float factor = 1.0e+00;
	int nodeas;

	if(E->control.side_sbcs)
		for(a=1;a<=ends1;a++)  {
			nodea = E->ien[m][el].node[ sidenodes[side][a] ];
			for(d=1;d<=dims;d++) {
				value = E->sbc.SB[m][side][d][ E->sbc.node[m][nodea] ];
				flagged = (E->node[m][nodea] & sbc_flag[d]) && (value);
				found |= flagged;
				traction[d][a] = ( flagged ? value : 0.0 );
			}
		}
	else {
		/* if side_sbcs is false, only apply sbc on top and bottom surfaces */
		if(side == SIDE_BOTTOM || side == SIDE_TOP) {
			for(a=1;a<=ends1;a++)  {
				nodea = E->ien[m][el].node[ sidenodes[side][a] ];
				for(d=1;d<=dims;d++) {
					value = E->sphere.cap[m].VB[d][nodea];
					flagged = (E->node[m][nodea] & sbc_flag[d]) && (value);
					found |= flagged;
					traction[d][a] = ( flagged ? value : 0.0 );
				}
			}
		}
	}

	/* skip the following computation if no sbc_flag is set
	   or value of sbcs are zero */
	if(!found) return;

	/* compute traction at each int point */
	construct_side_c3x3matrix_el(E,el,&Cc,&Ccx,
				     E->mesh.levmax,m,0,side);

	for(k=1;k<=oned;k++)
		for(d=1;d<=dims;d++) {
			traction_at_gs[d][k] = 0.0;
			for(j=1;j<=ends1;j++)
				traction_at_gs[d][k] += traction[d][j] * E->M.vpt[GMVINDEX(j,k)] ;
		}

	for(j=1;j<=ends1;j++) {
		a = sidenodes[side][j];
		for(d=1;d<=dims;d++) {
			p = dims*(a-1)+d-1;
			for(k=1;k<=oned;k++) {
				tmp = 0.0;
				for(b=1;b<=dims;b++)
					tmp += traction_at_gs[b][k] * Cc.vpt[BVINDEX(b,d,a,k)];

				elt_tr[p] += tmp * E->M.vpt[GMVINDEX(j,k)]
					* E->boundary.det[m][side][k][bel] * g_1d[k].weight[dims-1];

			}
		}
	}
}

static void get_elt_tr_pseudo_surf(struct All_variables *E, int bel, int side, double elt_tr[24], int m)
{

	const int dims=E->mesh.nsd;
	const int ends1=enodes[dims-1];
	const int oned = onedvpoints[dims];

	struct CC Cc;
	struct CCX Ccx;

	const unsigned sbc_flag[4] = {0,SBX,SBY,SBZ};

	double traction[4][5],traction_at_gs[4][5], value, tmp;
	int j, b, p, k, a, nodea, d;
	int el = E->boundary.element[m][bel];
	int flagged;
	int found = 0;

	const float rho = E->data.density;
	const float g = E->data.grav_acc;
	const float R = 6371000.0;
	const float eta = E->data.ref_viscosity;
	const float kappa = E->data.therm_diff;
	const float factor = 1.0e+00;
	int nodeas;

	if(E->control.side_sbcs)
		for(a=1;a<=ends1;a++)  {
			nodea = E->ien[m][el].node[ sidenodes[side][a] ];
			for(d=1;d<=dims;d++) {
				value = E->sbc.SB[m][side][d][ E->sbc.node[m][nodea] ];
				flagged = (E->node[m][nodea] & sbc_flag[d]) && (value);
				found |= flagged;
				traction[d][a] = ( flagged ? value : 0.0 );
			}
		}
	else {
		if( side == SIDE_TOP && E->parallel.me_loc[3]==E->parallel.nprocz-1 && (el%E->lmesh.elz==0)) {
			for(a=1;a<=ends1;a++)  {
				nodea = E->ien[m][el].node[ sidenodes[side][a] ];
				nodeas = E->ien[m][el].node[ sidenodes[side][a] ]/E->lmesh.noz;
				traction[1][a] = 0.0;
				traction[2][a] = 0.0;
				traction[3][a] = -1.0*factor*rho*g*(R*R*R)/(eta*kappa)
					*(E->slice.freesurf[m][nodeas]+E->sphere.cap[m].V[3][nodea]*E->advection.timestep);
				if(E->parallel.me==11 && nodea==3328)
					fprintf(stderr,"traction=%e vnew=%e timestep=%e coeff=%e\n",traction[3][a],E->sphere.cap[m].V[3][nodea],E->advection.timestep,-1.0*factor*rho*g*(R*R*R)/(eta*kappa));
				found = 1;
#if 0
				if(found && E->parallel.me==1)
					fprintf(stderr,"me=%d bel=%d el=%d side=%d TOP=%d a=%d sidenodes=%d ien=%d noz=%d nodea=%d traction=%e %e %e\n",
						E->parallel.me,bel,el,side,SIDE_TOP,a,sidenodes[side][a],
						E->ien[m][el].node[ sidenodes[side][a] ],E->lmesh.noz,
						nodea,traction[1][a],traction[2][a],traction[3][a]);

#endif
			}
		}
		else {
			for(a=1;a<=ends1;a++)  {
				nodea = E->ien[m][el].node[ sidenodes[side][a] ];
				for(d=1;d<=dims;d++) {
					value = E->sphere.cap[m].VB[d][nodea];
					flagged = (E->node[m][nodea] & sbc_flag[d]) && (value);
					found |= flagged;
					traction[d][a] = ( flagged ? value : 0.0 );
				}
			}
		}
	}

	/* skip the following computation if no sbc_flag is set
	   or value of sbcs are zero */
	if(!found) return;

	/* compute traction at each int point */
	construct_side_c3x3matrix_el(E,el,&Cc,&Ccx,
				     E->mesh.levmax,m,0,side);

	for(k=1;k<=oned;k++)
		for(d=1;d<=dims;d++) {
			traction_at_gs[d][k] = 0.0;
			for(j=1;j<=ends1;j++)
				traction_at_gs[d][k] += traction[d][j] * E->M.vpt[GMVINDEX(j,k)] ;
		}

	for(j=1;j<=ends1;j++) {
		a = sidenodes[side][j];
		for(d=1;d<=dims;d++) {
			p = dims*(a-1)+d-1;
			for(k=1;k<=oned;k++) {
				tmp = 0.0;
				for(b=1;b<=dims;b++)
					tmp += traction_at_gs[b][k] * Cc.vpt[BVINDEX(b,d,a,k)];

				elt_tr[p] += tmp * E->M.vpt[GMVINDEX(j,k)]
					* E->boundary.det[m][side][k][bel] * g_1d[k].weight[dims-1];

			}
		}
	}
}


/* =================================================================
 subroutine to get augmented lagrange part of stiffness matrix
================================================================== */

void get_aug_k(E,el,elt_k,level,m)
     struct All_variables *E;
     int el,m;
     double elt_k[24*24];
     int level;
{
  int i,p[9],a,b;
     double Visc;

     const int n=loc_mat_size[E->mesh.nsd];
     const int ends=enodes[E->mesh.nsd];
     const int vpts=vpoints[E->mesh.nsd];
     const int dims=E->mesh.nsd;

     Visc = 0.0;
     for(a=1;a<=vpts;a++) {
	  p[a] = (a-1)*dims;
	  Visc += E->EVI[level][m][(el-1)*vpts+a];
       }
     Visc = Visc/vpts;

     for(a=1;a<=ends;a++) {
       //nodea=E->IEN[level][m][el].node[a];
        for(b=1;b<=ends;b++) {
	  //nodeb=E->IEN[level][m][el].node[b];      /* for Kab dims*dims  */
	   i = (a-1)*n*dims+(b-1)*dims;
	   elt_k[i  ] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]][0]*
		      E->elt_del[level][m][el].g[p[b]][0];   /*for 11 */
	   elt_k[i+1] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]][0]*
		      E->elt_del[level][m][el].g[p[b]+1][0];  /* for 12 */
	   elt_k[i+n] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+1][0]*
		      E->elt_del[level][m][el].g[p[b]][0];    /* for 21 */
	   elt_k[i+n+1] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+1][0]*
		      E->elt_del[level][m][el].g[p[b]+1][0];  /* for 22 */

           if(3==dims) {
	       elt_k[i+2] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]][0]*
		      E->elt_del[level][m][el].g[p[b]+2][0];  /* for 13 */
	       elt_k[i+n+2] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+1][0]*
		      E->elt_del[level][m][el].g[p[b]+2][0];  /* for 23 */
	       elt_k[i+n+n] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+2][0]*
		      E->elt_del[level][m][el].g[p[b]][0];    /* for 31 */
	       elt_k[i+n+n+1] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+2][0]*
		      E->elt_del[level][m][el].g[p[b]+1][0];  /* for 32 */
	       elt_k[i+n+n+2] += Visc*E->control.augmented*
	              E->elt_del[level][m][el].g[p[a]+2][0]*
		      E->elt_del[level][m][el].g[p[b]+2][0];  /* for 33 */
               }
           }
       }

   return;
   }


/* version */
/* $Id$ */

/* End of file  */
