import numpy as np
import vtk
import vtk.numpy_interface
import math
from vtk.numpy_interface import dataset_adapter as dsa

class PolyLineIterator:
	'''Iterator for point indices in a vtkPolyLine object::
		polyLine = vtk.vtkPolyLine()
		for pi in PolyLineIterator(pl):
			point = polyLine.GetPoint(pi)
	
	Args:
		:cell: Polyline object to iterate over
	'''

	def __init__(self, cell):
		if not cell.IsA('vtkPolyLine'):
			raise TypeError('Unable to create the cell iterator, only vtkPolyLines supported as cell type')
		self.i = 0
		self.n = cell.GetNumberOfPoints()
		self.pl = cell

	def __iter__(self):
		return self

	def next(self):
		'''Advance the iterator and return the current point index'''
		if self.i < self.n:
			i = self.i
			self.i += 1
			return self.pl.GetPointId(i)
		else:
			raise StopIteration()
	def __getitem__(self, id):
		''' Fetch a certain element by its index'''
		if id < self.n:
			return self.pl.GetPointId(id)
		else:
			raise StopIteration()

class CellIterator:
	'''Iterator for cells in a vtk dataset::
		dataSet = vtk.vtkPolyData()
		for cell in CellIterator(dataSet):
			# Do something with the cell
	
	Args:
		:dataSet: non-composite vtk dataset, e.g. vtkPolyData or vtkUnstructuredGrid
	'''

	def __init__(self, dataSet):
		self.i = 0
		self.n = dataSet.GetNumberOfCells()
		self.ds = dataSet

	def __iter__(self):
		return self
	
	def next(self):
		'''Advance the iterator and return the current cell'''
		if self.i < self.n:
			i = self.i
			self.i += 1
			return self.ds.GetCell(i)
		else:
			raise StopIteration()

def _createMapper(dataSet):
	if dataSet.IsA('vtkPolyData'):
		mapper = vtk.vtkPolyDataMapper()
	elif dataSet.IsA('vtkUnstructuredGrid'):
		mapper = vtk.vtkDataSetMapper()
	else:
		raise RuntimeError('Unsupported dataset type')
	mapper.SetInputData(dataSet)
	return mapper

def renderDataSet(dataSet, **kwargs):
	""" Render a vtk dataset

	Args:
		:dataSet: vtkUnstructuredGrid or vtkPolyData

	Keyword args:
		:colorBy (str): constant or scalar
		:color (tuple): rgb value (if colorBy == constant)
	"""

	colorBy = kwargs.get('colorBy', 'constant')

	mapper = _createMapper(dataSet)

	actor = vtk.vtkActor()
	actor.SetMapper(mapper)

	renderer = vtk.vtkRenderer()
	renderer.SetBackground(1., 1., 1.)
	renderer.AddActor(actor)

	# Coloring
	if colorBy == 'constant':
		color = kwargs.get('color', (0.5, 0.5, 0.5))
		mapper.ScalarVisibilityOff()
		actor.GetProperty().SetColor(color[0], color[1], color[2])
	elif colorBy == 'scalar':
		# Create lookup table
		lut = vtk.vtkLookupTable()
		lut.SetTableRange(0, 1)
		lut.SetHueRange(0, 1)
		lut.SetSaturationRange(1, 1)
		lut.SetValueRange(1, 1)
		lut.Build()

		# Create colorbar
		colorbar = vtk.vtkScalarBarActor()
		colorbar.SetLookupTable(lut)
		colorbar.SetTitle('aaa')
		colorbar.SetNumberOfLabels(4)
		renderer.AddActor2D(colorbar)

		# Add colormap to the actor
		mapper.ScalarVisibilityOn()		
		mapper.SetLookupTable(lut)
	

	# Setup camera
	'''bounds = polyData.GetBounds()
	cx = (bounds[0] + bounds[1]) / 2.0
	cy = (bounds[2] + bounds[3]) / 2.0
	scale = max(math.fabs(bounds[0] - cx), math.fabs(bounds[1] - cx), math.fabs(bounds[2] - cy), math.fabs(bounds[3] - cy))
	camera = renderer.GetActiveCamera()
	camera.ParallelProjectionOn()
	camera.SetParallelScale(scale)
	camera.SetPosition(cx, cy, 1)
	camera.SetFocalPoint(cx, cy, 0)'''

	renderWindow = vtk.vtkRenderWindow()
	renderWindow.SetSize(600, 600)
	renderWindow.AddRenderer(renderer)
	#renderWindow.Render()

	interactor = vtk.vtkRenderWindowInteractor()
	interactor.SetRenderWindow(renderWindow)
	interactor.Initialize()
	#interactor.Render()
	interactor.Start()

def cutPolySurface(dataSet, point, normal):
	''' Cut a surface with a plane, and return an ordered list
	of points around the circumference of the resulting curve. The cut
	must result in a closed loop, and if the cut produces multiple sub-curves
	the closest one is returned.

	Args:
		:dataSet: (vtkPolyData): surface dataset
		:point: origin of the cutplane
		:normal: normal of the cutplane

	Returns:	
		:np.array: List of positions around the cicumference of the cut

	Raises:
		RuntimeError: If the cut results in a non-closed loop being formed
	'''

	# Generate surface cutcurve
	plane = vtk.vtkPlane()
	plane.SetOrigin(point[0], point[1], point[2])
	plane.SetNormal(normal[0], normal[1], normal[2])
	
	cutter = vtk.vtkCutter()
	cutter.SetInputData(dataSet)
	cutter.SetCutFunction(plane)
	cutter.Update()

	# Get cut line edges
	cutData = cutter.GetOutput()
	edges = []
	cutLines = cutData.GetLines()
	cutLines.InitTraversal()
	idList = vtk.vtkIdList()
	while cutLines.GetNextCell(idList) == 1:
		edges.append((idList.GetId(0), idList.GetId(1)))	

	# Gather all points by traversing the edge graph starting
	# from the point closest to the centerline point
	locator = vtk.vtkPointLocator()
	locator.SetDataSet(cutData)
	locator.BuildLocator()
	startPtId = locator.FindClosestPoint(point)

	pointIds = [startPtId]
	while True:
		# Find the edge that starts at the latest point 
		pred = (v[1] for v in edges if v[0] == pointIds[-1])
		currentPtId = pred.next()

		# Check if we've returned to the start point
		if currentPtId == startPtId:
			break

		pointIds.append(currentPtId)
	else:	# if no break occured
		raise RuntimeError('The cut curve does not form a closed loop')
	cutCurve = dsa.WrapDataObject(cutData)
	return cutCurve.Points[pointIds]

def createVtkCylinder(**kwargs):
	''' Create a vtk cylinder
		Keyword arguments:
			:origin (tuple/list/np-array): Origin of the cylinder
			:axis (tuple/list/np-array): Cylinder axis
			:radius (float): Cylinder radius
	'''
	origin = np.array(kwargs.get('origin', [0, 0, 0]))
	axis = np.array(kwargs.get('axis', [0, 0, 1]))
	radius = np.array(kwargs.get('radius', 1))

	cylinder = vtk.vtkCylinder()
	cylinder.SetCenter(0, 0, 0)
	cylinder.SetRadius(radius)

	# The cylinder is (by default) aligned with the y-axis
	# Rotate the cylinder so that it is aligned with the tangent vector
	transform = vtk.vtkTransform()

	yDir = np.array([0, 1, 0])
	# If the tangent is in the y-direction, do nothing
	if np.abs(1. - np.abs(np.dot(yDir, axis))) > 1e-8:
		# Create a vector in the normal direction to the plane spanned by yDir and the tangent
		rotVec = np.cross(yDir, axis)
		rotVec /= np.linalg.norm(rotVec)
		
		# Evaluate rotation angle
		rotAngle = np.arccos(np.dot(yDir, axis))
		transform.RotateWXYZ(-180*rotAngle/np.pi, rotVec)

	transform.Translate(-origin)
	cylinder.SetTransform(transform)
	
	return cylinder

def cutPolyData(dataSet, **kwargs):
	# Read options
	pt = kwargs.get('point')
	normal = kwargs.get('normal')
	delta = kwargs.get('maxDist')

	if pt == None:
		raise RuntimeError('No point provided')
	if normal == None:
		raise RuntimeError('No normal provided')

	# Create plane
	plane = vtk.vtkPlane()
	plane.SetOrigin(pt[0], pt[1], pt[2])
	plane.SetNormal(normal[0], normal[1], normal[2])

	cutter = vtk.vtkCutter()
	cutter.SetCutFunction(plane)
	cutter.SetInputData(dataSet)

	if delta == None:
		cutter.Update()
		return cutter.GetOutput()
	else:
		# Create a box
		box = vtk.vtkBox()
		box.SetBounds(pt[0]-delta, pt[0]+delta, pt[1]-delta, pt[1]+delta, pt[2]-delta, pt[2]+delta)
	
		# Clip data with a box
		clipper = vtk.vtkClipDataSet()
		clipper.SetClipFunction(box)
		clipper.SetInputConnection(cutter.GetOutputPort())
		clipper.InsideOutOn()
		clipper.Update()
		return clipper.GetOutput()
