import sys
import os
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
import vtkio
import vtktools
import glob
import numpy as np
from numpy.fft import fft, fftfreq
import pdb
from joblib import Parallel, delayed, wrap_non_picklable_objects

os.environ['LOKY_PICKLER'] = 'pickle'

@wrap_non_picklable_objects
def compute_snapshot(filename, timestep, args):
    reader = vtk.vtkEnSightGoldBinaryReader()
    reader.SetCaseFileName(filename)
    reader.SetTimeValue(timestep)
    reader.Modified()
    reader.Update()
    dataSet = reader.GetOutput()
    merger = vtk.vtkAppendFilter()
    for arg in args:
        data = vtkio.getBlockByName(dataSet, arg)
        merger.AddInputData(data)
    merger.Update()
    ds = dsa.WrapDataObject(merger.GetOutput())
    #vorticity = algs.vorticity(ds.CellData['Velocity'], ds)
    #ds1 = dsa.WrapDataObject(vorticity
    #del ds, merger, reader
    del dataSet, merger, reader
    return np.sqrt((ds.CellData['Velocity']*ds.CellData['Velocity']).sum(axis=1))


folder = '/media/fiusco/CannulaLPTAnimations/baby_pump_urans'
inputFolder = folder + '/vol_data'
outputFolder = folder + '/fft'
cacheFile = outputFolder + '/cache_volume.npz'

vtkio.createFolder(outputFolder)


print('In folder ', os.path.abspath(folder))

if not os.path.isfile(cacheFile):
    files = glob.glob(inputFolder + '/*.case')

    if len(files) == 0:
        sys.exit('No .case files found in {}'.format(inputFolder))

    reader = vtk.vtkEnSightGoldBinaryReader()
    reader.SetCaseFileName(files[0])
    reader.Update()
    merger = vtk.vtkAppendFilter()
    for arg in sys.argv[1:]:
        data = vtkio.getBlockByName(reader.GetOutput(), arg)
        merger.AddInputData(data)
    merger.Update()
    ds = dsa.WrapDataObject(merger.GetOutput())

    times = []
        
    for i in range(reader.GetTimeSets().GetNumberOfItems()):
        array = reader.GetTimeSets().GetItem(i)
        for j in range(array.GetNumberOfTuples()):
            times.append(array.GetComponent(j, 0))
    eigvalues = []
    timecoefficients = []
    del merger, ds, reader
    snaps = Parallel(n_jobs=6, max_nbytes=1e9, verbose=30)(delayed(compute_snapshot)(files[0], time, sys.argv[1:]) for time in times[1:])
    #pdb.set_trace()
    N = len(times)-1
    np.savez('cache_snapshots.npz', snaps=snaps)
    fft_values = np.empty((snaps[0].shape[0], N//2))
    xf = fftfreq(N, 0.001)
    snapshots = np.empty((snaps[0].shape[0], N))
    for i, snap in enumerate(snaps):
        snapshots[:,i] = snap
    ffts = Parallel(n_jobs=6, max_nbytes=1e9, verbose=30, prefer='threads')(delayed(fft)(snapshots[i,:]) for i in range(snapshots.shape[0]))
    for i, snap in enumerate(ffts):
        fft_values[i, :] = 2.0 / N * np.abs(snap[:N//2])
    np.savez(cacheFile, fft_values=fft_values, xf=xf)

else:
    data = np.load(cacheFile)
    fft_values = data['fft_values']
    xf = data['xf']

files = glob.glob(inputFolder + '/*.case')
files.sort()
first_file = files[0]


dataSet = vtkio.readDataSet(first_file)

merger = vtk.vtkAppendFilter()
for arg in sys.argv[1:]:
    merger.AddInputData(vtkio.getBlockByName(dataSet, arg))
merger.Update()

ds = dsa.WrapDataObject(merger.GetOutput())
xf = xf[:(len(xf)-1)//2]
print(xf)
pdb.set_trace()
for freq in [55, 170, 20]:
    index = np.argwhere(np.abs(xf-freq) < 2)[0][0]
    print(index)
    print(fft_values.shape)
    ds.CellData.append(fft_values[:, index], 'Surface_FFT_{}'.format(freq))
vtkio.writeDataSet(ds.VTKObject, 'surface_fft_probes.vtu')







    
        




