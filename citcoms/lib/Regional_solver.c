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


#include "global_defs.h"


/* Boundary_conditions.c */
void regional_velocity_boundary_conditions(struct All_variables *);
void regional_temperature_boundary_conditions(struct All_variables *);

/* Geometry_cartesian.c */
void regional_set_2dc_defaults(struct All_variables *);
void regional_set_2pt5dc_defaults(struct All_variables *);
void regional_set_3dc_defaults(struct All_variables *);
void regional_set_3dsphere_defaults(struct All_variables *);

/* Lith_age.c */
void regional_lith_age_read_files(struct All_variables *, int);

/* Parallel_related.c */
void regional_parallel_processor_setup(struct All_variables *);
void regional_parallel_domain_decomp0(struct All_variables *);
void regional_parallel_domain_boundary_nodes(struct All_variables *);
void regional_parallel_communication_routs_v(struct All_variables *);
void regional_parallel_communication_routs_s(struct All_variables *);
void regional_exchange_id_d(struct All_variables *, double **, int);

/* Read_input_from_files.c */
void regional_read_input_files_for_timesteps(struct All_variables *, int, int);

/* Version_dependent.c */
void regional_node_locations(struct All_variables *);
void regional_construct_boundary(struct All_variables *);


void regional_solver_init(struct All_variables *E)
{
    /* Boundary_conditions.c */
    E->solver.velocity_boundary_conditions = regional_velocity_boundary_conditions;
    E->solver.temperature_boundary_conditions = regional_temperature_boundary_conditions;

    /* Geometry_cartesian.c */
    E->solver.set_2dc_defaults = regional_set_2dc_defaults;
    E->solver.set_2pt5dc_defaults = regional_set_2pt5dc_defaults;
    E->solver.set_3dc_defaults = regional_set_3dc_defaults;
    E->solver.set_3dsphere_defaults = regional_set_3dsphere_defaults;

    /* Lith_age.c */
    E->solver.lith_age_read_files = regional_lith_age_read_files;

    /* Parallel_related.c */
    E->solver.parallel_processor_setup = regional_parallel_processor_setup;
    E->solver.parallel_domain_decomp0 = regional_parallel_domain_decomp0;
    E->solver.parallel_domain_boundary_nodes = regional_parallel_domain_boundary_nodes;
    E->solver.parallel_communication_routs_v = regional_parallel_communication_routs_v;
    E->solver.parallel_communication_routs_s = regional_parallel_communication_routs_s;
    E->solver.exchange_id_d = regional_exchange_id_d;

    /* Read_input_from_files.c */
    E->solver.read_input_files_for_timesteps = regional_read_input_files_for_timesteps;

    /* Version_dependent.c */
    E->solver.node_locations = regional_node_locations;
    E->solver.construct_boundary = regional_construct_boundary;
    
    return;
}


/* End of file */
