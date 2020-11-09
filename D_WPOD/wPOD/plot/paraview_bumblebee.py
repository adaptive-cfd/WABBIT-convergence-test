# state file generated using paraview version 5.6.0

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# trace generated using paraview version 5.6.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1184, 900]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.OrientationAxesVisibility = 0
renderView1.CenterOfRotation = [4.000000029802322, 4.000000029802322, 4.000000029802322]
renderView1.StereoType = 0
renderView1.CameraPosition = [-22.592263691896548, 1.02124577688772, 4.729588260977124]
renderView1.CameraFocalPoint = [4.000000029802322, 4.000000029802322, 4.000000029802322]
renderView1.CameraViewUp = [0.02701614735879065, 0.0036580831891271912, 0.9996283040256863]
renderView1.CameraParallelScale = 1.8747837375761771
renderView1.CameraParallelProjection = 1
renderView1.Background = [0.32, 0.34, 0.43]

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.XTitleFontFile = ''
renderView1.AxesGrid.YTitleFontFile = ''
renderView1.AxesGrid.ZTitleFontFile = ''
renderView1.AxesGrid.XLabelFontFile = ''
renderView1.AxesGrid.YLabelFontFile = ''
renderView1.AxesGrid.ZLabelFontFile = ''

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XDMF Reader'
aLL2xmf = XDMFReader(FileNames=['/home/krah/develop/WPOD/D_WPOD/bumblebee/Jmax7/eps1.0e-04/ALL2.xmf'])
aLL2xmf.PointArrayStatus = ['mode1', 'mode2', 'mode3']
aLL2xmf.GridStatus = ['wabbit3D']

# create a new 'Clip'
clip1 = Clip(Input=aLL2xmf)
clip1.ClipType = 'Plane'
clip1.Scalars = ['POINTS', 'mode1']

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [4.000000029802322, 4.000000029802322, 4.000000029802322]
clip1.ClipType.Normal = [0.0, -1.0, 0.0]

# create a new 'Calculator'
calculator1 = Calculator(Input=aLL2xmf)
calculator1.Function = 'sqrt(mode1^2+mode2^2+mode3^2)'

# create a new 'Contour'
contour1 = Contour(Input=calculator1)
contour1.ContourBy = ['POINTS', 'Result']
contour1.Isosurfaces = [20.0]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Contour'
contour3 = Contour(Input=calculator1)
contour3.ContourBy = ['POINTS', 'Result']
contour3.Isosurfaces = [10.0]
contour3.PointMergeMethod = 'Uniform Binning'

# create a new 'XDMF Reader'
aLL2xmf_1 = XDMFReader(FileNames=['/work/krah/bumblebee/bumblebee/CDF44_jmax7/ALL2.xmf'])
aLL2xmf_1.PointArrayStatus = ['mask']
aLL2xmf_1.GridStatus = ['wabbit3D']

# create a new 'Contour'
contour2 = Contour(Input=aLL2xmf_1)
contour2.ContourBy = ['POINTS', 'mask']
contour2.Isosurfaces = [0.5]
contour2.PointMergeMethod = 'Uniform Binning'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from contour1
contour1Display = Show(contour1, renderView1)

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.ColorArrayName = ['POINTS', '']
contour1Display.DiffuseColor = [0.3333333333333333, 1.0, 0.0]
contour1Display.OSPRayScaleArray = 'Normals'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.12789173126220704
contour1Display.SelectScaleArray = 'None'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'None'
contour1Display.GaussianRadius = 0.006394586563110352
contour1Display.SetScaleArray = ['POINTS', 'Normals']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'Normals']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.SelectionCellLabelFontFile = ''
contour1Display.SelectionPointLabelFontFile = ''
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
contour1Display.DataAxesGrid.XTitleFontFile = ''
contour1Display.DataAxesGrid.YTitleFontFile = ''
contour1Display.DataAxesGrid.ZTitleFontFile = ''
contour1Display.DataAxesGrid.XLabelFontFile = ''
contour1Display.DataAxesGrid.YLabelFontFile = ''
contour1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
contour1Display.PolarAxes.PolarAxisTitleFontFile = ''
contour1Display.PolarAxes.PolarAxisLabelFontFile = ''
contour1Display.PolarAxes.LastRadialAxisTextFontFile = ''
contour1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from contour2
contour2Display = Show(contour2, renderView1)

# trace defaults for the display properties.
contour2Display.Representation = 'Surface'
contour2Display.ColorArrayName = ['POINTS', '']
contour2Display.OSPRayScaleArray = 'Normals'
contour2Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour2Display.SelectOrientationVectors = 'None'
contour2Display.ScaleFactor = 0.15795686244964602
contour2Display.SelectScaleArray = 'None'
contour2Display.GlyphType = 'Arrow'
contour2Display.GlyphTableIndexArray = 'None'
contour2Display.GaussianRadius = 0.0078978431224823
contour2Display.SetScaleArray = ['POINTS', 'Normals']
contour2Display.ScaleTransferFunction = 'PiecewiseFunction'
contour2Display.OpacityArray = ['POINTS', 'Normals']
contour2Display.OpacityTransferFunction = 'PiecewiseFunction'
contour2Display.DataAxesGrid = 'GridAxesRepresentation'
contour2Display.SelectionCellLabelFontFile = ''
contour2Display.SelectionPointLabelFontFile = ''
contour2Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
contour2Display.DataAxesGrid.XTitleFontFile = ''
contour2Display.DataAxesGrid.YTitleFontFile = ''
contour2Display.DataAxesGrid.ZTitleFontFile = ''
contour2Display.DataAxesGrid.XLabelFontFile = ''
contour2Display.DataAxesGrid.YLabelFontFile = ''
contour2Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
contour2Display.PolarAxes.PolarAxisTitleFontFile = ''
contour2Display.PolarAxes.PolarAxisLabelFontFile = ''
contour2Display.PolarAxes.LastRadialAxisTextFontFile = ''
contour2Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from contour3
contour3Display = Show(contour3, renderView1)

# trace defaults for the display properties.
contour3Display.Representation = 'Surface'
contour3Display.ColorArrayName = ['POINTS', '']
contour3Display.DiffuseColor = [1.0, 0.6666666666666666, 0.0]
contour3Display.Opacity = 0.2
contour3Display.OSPRayScaleArray = 'Normals'
contour3Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour3Display.SelectOrientationVectors = 'None'
contour3Display.ScaleFactor = 0.2507790565490723
contour3Display.SelectScaleArray = 'None'
contour3Display.GlyphType = 'Arrow'
contour3Display.GlyphTableIndexArray = 'None'
contour3Display.GaussianRadius = 0.012538952827453613
contour3Display.SetScaleArray = ['POINTS', 'Normals']
contour3Display.ScaleTransferFunction = 'PiecewiseFunction'
contour3Display.OpacityArray = ['POINTS', 'Normals']
contour3Display.OpacityTransferFunction = 'PiecewiseFunction'
contour3Display.DataAxesGrid = 'GridAxesRepresentation'
contour3Display.SelectionCellLabelFontFile = ''
contour3Display.SelectionPointLabelFontFile = ''
contour3Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
contour3Display.DataAxesGrid.XTitleFontFile = ''
contour3Display.DataAxesGrid.YTitleFontFile = ''
contour3Display.DataAxesGrid.ZTitleFontFile = ''
contour3Display.DataAxesGrid.XLabelFontFile = ''
contour3Display.DataAxesGrid.YLabelFontFile = ''
contour3Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
contour3Display.PolarAxes.PolarAxisTitleFontFile = ''
contour3Display.PolarAxes.PolarAxisLabelFontFile = ''
contour3Display.PolarAxes.LastRadialAxisTextFontFile = ''
contour3Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# show data from clip1
clip1Display = Show(clip1, renderView1)

# trace defaults for the display properties.
clip1Display.Representation = 'Outline'
clip1Display.AmbientColor = [0.0, 0.0, 0.0]
clip1Display.ColorArrayName = ['POINTS', '']
clip1Display.Opacity = 0.44
clip1Display.OSPRayScaleArray = 'mode1'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 0.8
clip1Display.SelectScaleArray = 'mode1'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'mode1'
clip1Display.GaussianRadius = 0.04
clip1Display.SetScaleArray = ['POINTS', 'mode1']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = ['POINTS', 'mode1']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.SelectionCellLabelFontFile = ''
clip1Display.SelectionPointLabelFontFile = ''
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityUnitDistance = 0.038909116022299545

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
clip1Display.DataAxesGrid.XTitleFontFile = ''
clip1Display.DataAxesGrid.YTitleFontFile = ''
clip1Display.DataAxesGrid.ZTitleFontFile = ''
clip1Display.DataAxesGrid.XLabelFontFile = ''
clip1Display.DataAxesGrid.YLabelFontFile = ''
clip1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
clip1Display.PolarAxes.PolarAxisTitleFontFile = ''
clip1Display.PolarAxes.PolarAxisLabelFontFile = ''
clip1Display.PolarAxes.LastRadialAxisTextFontFile = ''
clip1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# ----------------------------------------------------------------
# finally, restore active source
SetActiveSource(aLL2xmf)
# ----------------------------------------------------------------