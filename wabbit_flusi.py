#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import wabbit_tools
import os.path
import matplotlib.pyplot as plt
import insect_tools
import numpy as np
import textwrap
import math

# first, test convergence order in c0
def convergence_c0(vor, ux, dirwabbit, dirflusi):

    if (not vor and not ux):
        vor=True
    ######### plot options
    title_font = {'fontname':'Arial', 'size':'9', 'color':'black', 'weight':'normal'}
    axis_font = {'fontname':'Arial', 'size':'14'}

    ######### initialize
    resolution = np.array([128,256,512,1024,2048])
    c_0 = np.linspace(10,100,10)
    error = np.zeros((len(resolution),len(c_0),100))
    c_0 = np.int_(c_0)
    mean_error = np.zeros((len(resolution),len(c_0)))

    ############ read wabbit and flusi file and save error
    for i in range(len(resolution)):
        level=i+2
        j = 0
        for c in c_0:
            dirwabbit_c = dirwabbit +str(level)+'_level/c_'+str(c)+'/'
            if vor:
                file_wabbit = dirwabbit_c+'vortdense_000040000000.h5'
                f = dirflusi+'vorx_040000.h5'
            if ux:
                file_wabbit = dirwabbit_c+'Uxdense_000040000000.h5'
                f = dirflusi+'uy_040000.h5'                
            mean_error[i,j]=wabbit_tools.wabbit_error_vs_flusi(file_wabbit,f)
            j = j+1

    ######### plot the error
    for i in range(len(resolution)):
        fig = plt.figure()
        title = ('Three vortices: WABBIT acm compared to Flusi'+
                 'incompressible solution at t=40. Resolution: %i. Convergence order: %e'
                 % (resolution[i],(wabbit_tools.convergence_order(c_0, mean_error[i,:])) ))
        plt.title('\n'.join(textwrap.wrap(title,60)), **title_font)
        plt.loglog(c_0,mean_error[i,:], color='black', linewidth=2, label=str(resolution[i]), marker='o')
        plt.grid()
        plt.xlabel('$c_0$', **axis_font)
        if vor:
            plt.ylabel('$||\omega-\omega_{\\mathrm{flusi}}||_2/||\omega_{\\mathrm{flusi}}||_2$', **axis_font)
            plt.savefig('3vortices/conv_c0_res_'+str(resolution[i])+'_vor.eps')
        if ux:
            plt.ylabel('$||u-u_{\\mathrm{flusi}}||_2/||u_{\\mathrm{flusi}}||_2$', **axis_font)
            plt.savefig('3vortices/conv_c0_res_'+str(resolution[i])+'_ux.eps')
        print('CONVERGENCE ORDER: %e' % (wabbit_tools.convergence_order(c_0, mean_error[i,:])))
    return

def spatial_convergence(vor, ux, dirwabbit, dirflusi):
    if (not vor and not ux):
        vor=True
    ######### plot options
    title_font = {'fontname':'Arial', 'size':'9', 'color':'black', 'weight':'normal'}
    axis_font = {'fontname':'Arial', 'size':'14'}
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, 15)]

    ######### initialize
    resolution = np.array([128,256,512,1024,2048])
    c_0 = np.array([10, 20, 30, 40, 80])
    #c_0 = np.array([40])
    c_0 = np.int_(c_0)
    error_acm = np.zeros((len(resolution),len(c_0)))
    error_inc = np.zeros((len(resolution),len(c_0)))
    error_flusi = np.zeros((len(resolution),len(c_0)))

    ############ read wabbit and flusi file and save error
    for i in range(len(resolution)):
        level=i+2
        j = 0
        for c in c_0:
            dirwabbit_acm= dirwabbit + str(level)+'_level/c_'+str(c)+'/'
            dirflusi_acm = dirflusi+'spectral_acm/Re20k_c0_'+str(c)+'_gamma1/upsample/'
            dirflusi_inc = '/work/mutzel/three_vortices/Inicond_Flusi/flusi_reference/2048/'

            if vor:
                file_wabbit = dirwabbit_acm+'vortdense_000040000000.h5'
                file_flusi_acm = dirflusi_acm+'vorx_040000.h5'
                file_flusi_inc = dirflusi_inc+'vorx_040000.h5'
            if ux:
                file_wabbit = dirwabbit_acm+'Uxdense_000040000000.h5'
                file_flusi_acm = dirflusi_acm+'uy_040000.h5'
                file_flusi_inc = dirflusi_inc+'uy_040000.h5'
           
            error_acm[i,j]=wabbit_tools.wabbit_error_vs_flusi(file_wabbit,file_flusi_acm,norm=np.inf)
            error_inc[i,j]=wabbit_tools.wabbit_error_vs_flusi(file_wabbit,file_flusi_inc,norm=np.inf)
            error_flusi[i,j]=flusi_error_vs_flusi(file_flusi_inc,file_flusi_acm,norm=np.inf)

            j = j+1

    ######### plot the error
    j=0
    for i in range(len(c_0)):
        fig = plt.figure(1)
        title = ('Three vortices: WABBIT acm compared to '+
                 'Flusi acm solution and Flusi incompressible solution at t=40.' )
        plt.title('\n'.join(textwrap.wrap(title,60)), **title_font)
        plt.loglog(resolution,error_acm[:,i], linewidth=2, color=colors[j],marker='o', label='Error: Flusi acm-WABBIT acm, c_0=%i' %(c_0[i]))
        j = j+1
        plt.loglog(resolution,error_inc[:,i], linewidth=2, color=colors[j],  linestyle='dashed',marker='o', label='Error: Flusi incomp-WABBIT acm, c_0=%i' %(c_0[i]) )
        j = j+1
        plt.loglog(resolution,error_flusi[:,i], linewidth=2, color=colors[j], linestyle='dotted', marker='o', label='Error: Flusi incomp-FLUSI acm, c_0=%i' %(c_0[i]) )
        j = j+1
    plt.legend(loc='best', fontsize=7)
    plt.grid()
    plt.xlabel('$resolution$', **axis_font)
    if vor:
        plt.ylabel('$||\omega-\omega_{\\mathrm{flusi}}||_2/||\omega_{\\mathrm{flusi}}||_2$', **axis_font)
        plt.savefig('3vortices/spatial_conv_vor.eps')

    if ux:
        plt.ylabel('$||u-u_{\\mathrm{flusi}}||_2/||u_{\\mathrm{flusi}}||_2$', **axis_font)
        plt.savefig('3vortices/spatial_conv_ux.eps')

    print('CONVERGENCE ORDER: %e' % (wabbit_tools.convergence_order(resolution, error_acm[:,i])))
    return

def compare_statistics(plot_c_flusi, dirwabbit, dirflusi):
    ####### set plot options
    title_font = {'fontname':'Arial', 'size':'9', 'color':'black', 'weight':'normal'}
    axis_font = {'fontname':'Arial', 'size':'14'}
    ####### set parameters
    level = np.array([2,3,4,5])
    resolution = np.array([128, 256, 512, 1024,2048])
    #c_0 = np.linspace(10,100,10)
    c_0 = np.array([80])
    c_0 = np.int_(c_0)
    nu = 5*1.e-5
    ####### read flusi data
    file_flusi  = dirflusi+'/spectral_incompressible/energy.t'
    data_f = np.loadtxt(file_flusi, comments='%', skiprows=1, usecols = (0,1,5))
    time_f = data_f[:,0]
    t50 = int(np.where(time_f==50)[0])
    ekin_f = data_f[:,1]
    z_f = data_f[:,2]/nu
    ####### plot flusi data
    # ekin
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time_f[0:t50],ekin_f[0:t50], \
             color='black', linewidth=1, \
             label='Flusi incompressible with res = 1035')
    # enstrophy
    fig = plt.figure(1)
    plt.subplot(2,1,2)
    plt.plot(time_f[0:t50],z_f[0:t50], \
             color='black', linewidth=1, \
             label='Flusi incompressible with res = 1035')
    ######## read and plot wabbit data
    i = 0
    for l in level:
        j = 0
        for c in c_0:
            dirwabbit_c   = dirwabbit+str(l)+'_level/c_'+str(c)+'/'
            file_wabbit = dirwabbit_c+'e_kin.t'
            file_wabbit_z = dirwabbit_c+'enstrophy.t'

            data = np.loadtxt(file_wabbit)
            data_z = np.loadtxt(file_wabbit_z)
            time_w = data[:,0]
            ekin_w = data[:,1]*4*math.pi**2
            time_z = data_z[:,0]
            z_w    = data_z[:,1]*4*math.pi**2
            if plot_c_flusi:
                file_flusi  = dirflusi+'/spectral_acm/Re20k_c0_'+str(c)+'_gamma1/energy.t'
                data_f = np.loadtxt(file_flusi, comments='%', skiprows=1, usecols = (0,1,5))
                time_f = data_f[:,0]
                ekin_f = data_f[:,1]
                z_f = data_f[:,2]/nu
                ####### plot flusi data
                # ekin
                fig = plt.figure(1)
                plt.subplot(2,1,1)
                plt.plot(time_f,ekin_f, linewidth=1, \
                         label='Flusi acm solution with c_0 = %i and res = 1035' % (c))
                plt.subplot(2,1,2)
                plt.plot(time_f,z_f, linewidth=1, \
                         label='Flusi acm solution with c_0 = %i and res = 1035' % (c) )
            j = j+1
            fig = plt.figure(1)
            plt.subplot(2,1,1)
            plt.plot(time_w,ekin_w, linewidth=1, label='Wabbit solution with $c_0$ = %i and res = %i' % (c, resolution[i]))
            plt.subplot(2,1,2)
            plt.plot(time_z,z_w, linewidth=1, label='Wabbit solution with $c_0$ = %i and res = %i' % (c, resolution[i]))
        i = i+1
    plt.subplot(2,1,1)
    plt.legend(loc='upper right', fontsize=7)
    plt.grid()
    plt.ylabel('$e_{\\mathrm{kin}}$', **axis_font)
    plt.subplot(2,1,2)
    plt.legend(loc='upper right', fontsize=7)
    plt.grid()
    plt.xlabel('$t$', **axis_font)
    plt.ylabel('$z$', **axis_font)
    plt.savefig('3vortices/ekin_z_compare_c_'+str(c)+'.eps')

def error_mistral(file_wabbit, file_mistral, file_flusi, file_mistral_2048):
    ####### set plot options
    title_font = {'fontname':'Arial', 'size':'9', 'color':'black', 'weight':'normal'}
    axis_font = {'fontname':'Arial', 'size':'14'}
    error_wabbit=wabbit_tools.wabbit_error_vs_flusi(file_wabbit,file_mistral, norm=np.inf)
    error_flusi = flusi_error_vs_flusi(file_mistral_2048, file_flusi, norm=np.inf)
    print(error_wabbit)
    print(error_flusi)

def flusi_error_vs_flusi(file_flusi_ref, file_flusi,  norm=2):

    # read in flusi's reference solution
    time_ref, box_ref, data_ref = insect_tools.read_flusi_HDF5( file_flusi_ref )
    ny = data_ref.shape[1]
    # squeeze 3D flusi field (where dim0 == 1) to true 2d data
    data_ref = data_ref[:,:,0].copy()
    box_ref = box_ref[1:2].copy()

    # read in flusi's reference solution
    time_flusi, box_flusi, data_flusi = insect_tools.read_flusi_HDF5( file_flusi )
    ny = data_flusi.shape[1]
    # squeeze 3D flusi field (where dim0 == 1) to true 2d data
    data_flusi = data_flusi[:,:,0].copy()
    box_flusi = box_flusi[1:2].copy()

    err = np.ndarray.flatten(data_flusi-data_ref)
    exc = np.ndarray.flatten(data_ref)

    err = np.linalg.norm(err, ord=norm) / np.linalg.norm(exc, ord=norm)

    return err

def do_test(file_wabbit, file_flusi):
    time_ref, box_ref, data_ref = insect_tools.read_flusi_HDF5( file_flusi )
    time, x0, dx, box, data, treecode = wabbit_tools.read_wabbit_hdf5( file_wabbit )
    data_dense, box_dense = wabbit_tools.dense_matrix( x0, dx, data, treecode )
    data_ref = data_ref[:,:,0].copy()
    box_ref = box_ref[1:2].copy()

    plt.figure()
    plt.pcolormesh(data_ref)
    plt.savefig('data_ref.eps')

    plt.figure()
    plt.pcolormesh(data_dense)
    plt.savefig('data_dense.eps')


#convergence_c0(False,True, '/work/mutzel/three_vortices/Inicond_Flusi/equidistant/', '/work/mutzel/three_vortices/Inicond_Flusi/flusi_reference/2048/')

#compare_statistics(True, "/work/mutzel/three_vortices/Inicond_Flusi/equidistant/", "/work/mutzel/three_vortices/Inicond_Flusi/flusi_reference")

spatial_convergence(True,False, '/work/mutzel/three_vortices/Inicond_Flusi/equidistant/', '/work/mutzel/three_vortices/Inicond_Flusi/flusi_reference/')


#error_mistral('/work/mutzel/three_vortices/Inicond_Flusi/new_inicond_equidistant/2_level/c_40/Ux_000040000000.h5', '/home/mutzel/MISTRAL/uy_040000.h5', '/work/mutzel/three_vortices/Inicond_Flusi/flusi_reference/spectral_acm/Re20k_c0_40_gamma1/upsample/uy_040000.h5', '/home/mutzel/MISTRAL/uyup_040000.h5')
#print(wabbit_tools.wabbit_error_vs_flusi('/home/mutzel/WABBIT-convergence-test/mask_000000000000.h5','/home/mutzel/WABBIT-convergence-test/mask_000000.h5', norm=np.inf, dim=3))
