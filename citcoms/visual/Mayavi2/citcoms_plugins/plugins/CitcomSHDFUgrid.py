#    Script to generate VTKUnstructuredGrid objects from CitcomS hdf files
#    author: Martin Weier
#    Copyright (C) 2006 California Institue of Technology 
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA



from enthought.tvtk.api import tvtk
import tables        #For HDF support
import numpy
from math import *
from datetime import datetime

class CitcomSHDFUgrid:
    
    """This Class converts CitcomS hdf files to tvtk UnstructuredGrid Dataset Objects """
    
    data = None
    _nx = None
    _ny = None
    _nz = None
    _nx_redu = None
    _ny_redu = None
    _nz_redu = None
    _radius_inner = None
    _radius_outer = None
    #timesteps = None
    frequency = None
    
    progress = 0
    
    #Global because a Unstructured Grid can only hold one scalar value at a time
    #but our hdf file reader plugin wants to be able to read both
    __vtkordered_visc = tvtk.FloatArray()
    __vtkordered_temp = tvtk.FloatArray()  
    

    def vtk_iter(self,nx,ny,nz):
        """Iterator for CitcomDataRepresentation(yxz) to VTK(xyz)"""
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    yield k + nz * i + nz * nx * j

    
    def reduce_iter(self,n,nredu):
        """Iterator to reduce the CitcomS grid"""
        i=0
        n_f=float(n)
        nredu_f=float(nredu)
        fl=(n_f-1)/nredu_f
        redu = 0
        for i in xrange(nredu+1):
            yield int(round(redu))
            redu = redu + fl
            
    
    def velocity2cart(self,vel_colat,vel_long,r, x, y, z):
        """Converts vectors in spherical to cartesian coordiantes"""
        x1 = r*sin(x)*cos(y)+vel_colat*cos(x)*cos(y)-vel_long*sin(y)
        y1 = r*sin(x)*sin(y)+vel_colat*cos(x)*sin(y)+vel_long*cos(y)
        z1 = r*cos(x)-vel_colat*sin(x)
        return x1, y1, z1


    #Converts Spherical to Cartesian Coordinates
    def RTF2XYZ(self,thet, phi, r):
        """Converts points from spherical to cartesian coordinates"""
        x = r * sin(thet) * cos(phi)
        y = r * sin(thet) * sin(phi)
        z = r * cos(thet)
        return x, y, z
    
    
    
    def __citcom2vtk(self,f,ft,nproc_surf,nx_redu,ny_redu,nz_redu):
        """Method to convert one timestep from a hdf file to a Vtk file. This Method is used
        by the method initialize. Initialize reads the necessary meta information from the hdf file"""
        
        hexagrid = tvtk.UnstructuredGrid()
        hexagrid.allocate(1,1)
        
        vtkordered_velo = tvtk.FloatArray()
        
        nx = self._nx
        ny = self._ny
        nz = self._nz
        counter = 0
        el_nx_redu = nx_redu + 1
        el_ny_redu = ny_redu + 1
        el_nz_redu = nz_redu + 1
            
        ordered_points = [] #reset Sequences for points   
        ordered_temperature = []
        ordered_velocity = []
        ordered_viscosity = []
    
        for capnr in xrange(nproc_surf):
          
           
            #cap = f.root._f_getChild("cap%02d" % capnr)
    
            temp_coords =  [] # reset Coordinates, Velocity, Temperature Sequence
            temp_vel = []     
            temp_temp = []
            temp_visc = []
    
            #Information from hdf
            #hdf_coords = cap.coord[:]
            #hdf_velocity = cap.velocity[t]
            #hdf_temperature = cap.temperature[t]
            #hdf_viscosity = cap.viscosity[t]

            hdf_coords = f.root.coord[capnr]
            hdf_velocity = ft.root.velocity[capnr]
            hdf_temperature = ft.root.temperature[capnr]
            hdf_viscosity = ft.root.viscosity[capnr]

    

            #Create Iterator to change data representation
            nx_redu_iter = self.reduce_iter(nx,nx_redu)
            ny_redu_iter = self.reduce_iter(ny,ny_redu)
            nz_redu_iter = self.reduce_iter(nz,nz_redu)
      
            #vtk_i = self.vtk_iter(el_nx_redu,el_ny_redu,el_nz_redu)
             
         
            # read citcom data - zxy (z fastest)
            for j in xrange(el_ny_redu):
                j_redu = ny_redu_iter.next()
                nx_redu_iter = self.reduce_iter(nx,nx_redu)
                for i in xrange(el_nx_redu):
                    i_redu = nx_redu_iter.next()
                    nz_redu_iter = self.reduce_iter(nz,nz_redu)
                    for k in xrange(el_nz_redu):
                        k_redu = nz_redu_iter.next()
                
                        colat, lon, r = map(float,hdf_coords[i_redu,j_redu,k_redu])
                        x_coord, y_coord, z_coord = self.RTF2XYZ(colat,lon,r)
                        ordered_points.append((x_coord,y_coord,z_coord))
      
                        ordered_temperature.append(float(hdf_temperature[i_redu,j_redu,k_redu]))
                        ordered_viscosity.append(float(hdf_viscosity[i_redu,j_redu,k_redu]))
                    
                        vel_colat, vel_lon , vel_r = map(float,hdf_velocity[i_redu,j_redu,k_redu])
                        x_velo, y_velo, z_velo = self.velocity2cart(vel_colat,vel_lon,vel_r, colat,lon , r)
                    
                        ordered_velocity.append((x_velo,y_velo,z_velo))                        
        
        
            ##Delete Objects for GC
            del hdf_coords
            del hdf_velocity
            del hdf_temperature
            del hdf_viscosity
        

           
            #Create Connectivity info    
            if counter==0:
                 i=1    #Counts X Direction
                 j=1    #Counts Y Direction
                 k=1    #Counts Z Direction
    
                 for n in xrange((el_nx_redu*el_ny_redu*el_nz_redu)-(el_nz_redu*el_nx_redu)):
                     if (i%el_nz_redu)==0:            #X-Values!!!
                         j+=1                 #Count Y-Values
        
                     if (j%el_nx_redu)==0:
                         k+=1                #Count Z-Values
                  
                     if i%el_nz_redu!=0 and j%el_nx_redu!=0:            #Check if Box can be created
                         #Get Vertnumbers
                         n0 = n+(capnr*(el_nx_redu*el_ny_redu*el_nz_redu))
                         n1 = n0+el_nz_redu
                         n2 = n1+el_nz_redu*el_nx_redu
                         n3 = n0+el_nz_redu*el_nx_redu
                         n4 = n0+1
                         n5 = n4+el_nz_redu
                         n6 = n5+el_nz_redu*el_nx_redu
                         n7 = n4+el_nz_redu*el_nx_redu
                         #Created Polygon Box
                         hexagrid.insert_next_cell(12,[n0,n1,n2,n3,n4,n5,n6,n7])
             
                     i+=1
        
        
        #Store Arrays in Vtk conform Datastructures
        self.__vtkordered_temp.from_array(ordered_temperature)
        self.__vtkordered_visc.from_array(ordered_viscosity)                                    
        vtkordered_velo.from_array(ordered_velocity)
        
        self.__vtkordered_temp.name = 'Temperature'
        self.__vtkordered_visc.name = 'Viscosity'
        hexagrid.point_data.scalars = self.__vtkordered_temp
        vtkordered_velo.name = 'Velocity'
        hexagrid.point_data.vectors = vtkordered_velo
        hexagrid.points = ordered_points
            
        self.progress += 1
        
        return hexagrid
            
            
    def initialize(self,filename,timestep,nx_redu,ny_redu,nz_redu):
        """Call this method to convert a Citcoms Hdf file to a Vtk file"""
        
        from citcoms_plugins.utils import parsemodel
        (step, modelname, metafile, fullpath) = parsemodel(filename)

        if not fullpath:
            # try to load step=0
            import os.path
            step = 0
            pardir, mfilename = os.path.split(metafile)
            fullpath = os.path.join(pardir, "%s.%d.h5" % (modelname, step))

        
        #Read meta-inforamtion
        hdf = tables.openFile(metafile, 'r')
        hdf_t = tables.openFile(fullpath, 'r')

        self._nx = int(hdf.root.input._v_attrs.nodex)
        self._ny = int(hdf.root.input._v_attrs.nodey)
        self._nz = int(hdf.root.input._v_attrs.nodez)
        

        #Clip against boundaries
        if nx_redu>=0 or nx_redu>=self._nx:
            nx_redu = self._nx-1
        if ny_redu==0 or ny_redu>=self._ny:
            ny_redu = self._ny-1
        if nz_redu==0 or nz_redu>=self._nz:
            nz_redu = self._nz-1
        
        #Make reduction factors global
        self._nx_redu = nx_redu
        self._ny_redu = ny_redu
        self._nz_redu = nz_redu
        
        #Number of Timesteps in scene   
        #self.timesteps = int(hdf.root.time.nrows)

        #Number of caps
        nproc_surf = int(hdf.root.input._v_attrs.nproc_surf)
        #Store the Inner Radius. Import if we want to create a core
        self._radius_inner = self._radius_inner = float(hdf.root.input._v_attrs.radius_inner)
        #start computation
        hexgrid = self.__citcom2vtk(hdf, hdf_t, nproc_surf,nx_redu,ny_redu,nz_redu)
        
        hdf.close()
        self.progress = -1
        return hexgrid 
        
    def get_vtk_viscosity(self):
        return self.__vtkordered_visc
    
    def get_vtk_temperature(self):
        return self.__vtkordered_temp
