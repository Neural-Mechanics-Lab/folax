import pytest
import os
import numpy as np
import pyvista as pv
from fol.tools.plotter import Plotter2D


@pytest.fixture
def tmp_2d_vtk(tmp_path):
    """Create a flat 2D mesh (z=0) with FOL and FE displacement fields."""
    # Create a simple 2D grid (flat in z-plane)
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    mesh = pv.PolyData(points)
    mesh = mesh.delaunay_2d()
    
    pts = mesh.points.shape[0]
    
    # Sample 0: larger error (use uppercase U_FOL_ and U_FE_ to match defaults)
    mesh.point_data['U_FOL_0'] = np.column_stack([
        0.1 * np.random.rand(pts),
        0.1 * np.random.rand(pts),
        np.zeros(pts)
    ])
    mesh.point_data['U_FE_0'] = np.column_stack([
        0.1 * np.random.rand(pts) + 0.5,
        0.1 * np.random.rand(pts) + 0.5,
        np.zeros(pts)
    ])
    mesh.point_data['abs_error_0'] = mesh.point_data['U_FE_0'] - mesh.point_data['U_FOL_0']
    
    # Sample 1: smaller error (should be selected as best)
    mesh.point_data['U_FOL_1'] = np.column_stack([
        0.2 * np.ones(pts),
        0.3 * np.ones(pts),
        np.zeros(pts)
    ])
    mesh.point_data['U_FE_1'] = np.column_stack([
        0.21 * np.ones(pts),
        0.31 * np.ones(pts),
        np.zeros(pts)
    ])
    mesh.point_data['abs_error_1'] = mesh.point_data['U_FE_1'] - mesh.point_data['U_FOL_1']
    
    vtk_file = tmp_path / "mesh_2d.vtk"
    mesh.save(str(vtk_file))
    return str(vtk_file)


@pytest.fixture
def tmp_3d_vtk(tmp_path):
    """Create a 3D mesh (not flat) to test error handling."""
    mesh = pv.Cube().triangulate()
    pts = mesh.points.shape[0]
    mesh.point_data['U_FOL_0'] = np.zeros((pts, 3))
    mesh.point_data['U_FE_0'] = np.ones((pts, 3))
    mesh.point_data['abs_error_0'] = mesh.point_data['U_FE_0'] - mesh.point_data['U_FOL_0']
    
    vtk_file = tmp_path / "mesh_3d.vtk"
    mesh.save(str(vtk_file))
    return str(vtk_file)


def test_plotter2d_rejects_3d_mesh(tmp_3d_vtk):
    """Test that Plotter2D raises ValueError for non-flat meshes."""
    config = {}
    with pytest.raises(ValueError, match="flat 2-D mesh"):
        Plotter2D(vtk_path=tmp_3d_vtk, config=config)


def test_plotter2d_initialization(tmp_2d_vtk):
    """Test that Plotter2D initializes correctly with a flat mesh."""
    config = {
        "warp_factor_2d": 0.0
    }
    plotter = Plotter2D(vtk_path=tmp_2d_vtk, config=config)
    
    # Check initialization
    assert plotter.do_clip is False
    assert plotter.warp_factor_2d is None or plotter.warp_factor_2d == 0.0
    assert plotter.u_fol_pre == "U_FOL_"  # Default uppercase prefix
    assert plotter.u_ref_pre == "U_FE_"    # Default uppercase prefix


def test_find_best_sample_by_abs_error(tmp_2d_vtk):
    """Test that _find_best_sample_by_abs_error selects the sample with minimum error."""
    config = {}
    plotter = Plotter2D(vtk_path=tmp_2d_vtk, config=config)
    
    # Should select sample 1 (smaller error)
    assert plotter.best_id == 1
    assert plotter.fields["U_FOL"] == "U_FOL_1"  # Uses default uppercase prefix
    assert plotter.fields["U_REF"] == "U_FE_1"    # Uses default uppercase prefix
    assert plotter.fields["ERR"] == "abs_error_1_mag"
    
    # Check that magnitude fields were created
    assert "abs_error_0_mag" in plotter.mesh.point_data
    assert "abs_error_1_mag" in plotter.mesh.point_data


def test_ensure_mag_creates_magnitude_field(tmp_2d_vtk):
    """Test that _ensure_mag correctly creates magnitude fields."""
    config = {}
    plotter = Plotter2D(vtk_path=tmp_2d_vtk, config=config)
    
    # Test with vector field
    mag_name = plotter._ensure_mag("U_FOL_1")
    assert mag_name == "U_FOL_1_mag"
    assert "U_FOL_1_mag" in plotter.mesh.point_data
    
    # Verify magnitude values
    expected_mag = np.linalg.norm(plotter.mesh["U_FOL_1"], axis=1)
    np.testing.assert_allclose(plotter.mesh["U_FOL_1_mag"], expected_mag)
    
    # Test with already magnitude field
    mag_name_2 = plotter._ensure_mag("U_FOL_1_mag")
    assert mag_name_2 == "U_FOL_1_mag"


def test_warped_mesh_creates_copy(tmp_2d_vtk):
    """Test that _warped_mesh creates a warped copy without modifying original."""
    config = {
        "warp_factor_2d": 2.0
    }
    plotter = Plotter2D(vtk_path=tmp_2d_vtk, config=config)
    
    original_points = plotter.mesh.points.copy()
    
    # Create warped mesh
    plotter._ensure_mag("U_FOL_1")
    warped = plotter._warped_mesh("U_FOL_1", "U_FOL_1_mag")
    
    # Original mesh should be unchanged
    np.testing.assert_allclose(plotter.mesh.points, original_points)
    
    # Warped mesh should be different
    assert not np.allclose(warped.points, original_points)
    
    # Magnitude field should exist in warped mesh
    assert "U_FOL_1_mag" in warped.point_data


def test_render_panel_invokes_screenshot(tmp_2d_vtk, monkeypatch):
    """Test that render_panel creates a screenshot file."""
    config = {
        "window_size": (800, 600),
        "cmap": "viridis",
        "zoom": 1.0,
        "title_font_size": 12
    }
    plotter = Plotter2D(vtk_path=tmp_2d_vtk, config=config)
    
    calls = []
    def fake_screenshot(*args, **kwargs):
        # Capture the filename argument
        calls.append(args[-1] if args else kwargs.get('filename'))
    
    monkeypatch.setattr(pv.Plotter, 'screenshot', fake_screenshot)
    
    # Ensure magnitude field exists
    plotter._ensure_mag("U_FOL_1")
    
    fname = "test_panel.png"
    plotter.render_panel(
        plotter.mesh,
        field="U_FOL_1_mag",
        clim=[0, 1],
        title="Test Panel",
        fname=fname
    )
    
    expected = os.path.join(plotter.output_dir, fname)
    assert calls == [expected]


def test_render_all_panels_creates_files(tmp_2d_vtk, monkeypatch):
    """Test that render_all_panels creates all expected output files."""
    config = {
        "window_size": (800, 600),
        "cmap": "viridis",
        "zoom": 1.0,
        "title_font_size": 12,
        "output_image": "test_overview2d.png"
    }
    plotter = Plotter2D(vtk_path=tmp_2d_vtk, config=config)
    
    screenshot_calls = []
    def fake_screenshot(*args, **kwargs):
        screenshot_calls.append(args[-1] if args else kwargs.get('filename'))
    
    monkeypatch.setattr(pv.Plotter, 'screenshot', fake_screenshot)
    
    # Mock matplotlib image reading to avoid needing actual files
    import matplotlib.image as mpimg
    def fake_imread(fname):
        # Return a dummy numpy array (fake image)
        return np.zeros((100, 100, 3))
    monkeypatch.setattr(mpimg, 'imread', fake_imread)
    
    # Mock matplotlib to avoid actual image processing
    import matplotlib.pyplot as plt
    savefig_calls = []
    def fake_savefig(*args, **kwargs):
        savefig_calls.append(args[0] if args else kwargs.get('fname'))
    monkeypatch.setattr(plt, 'savefig', fake_savefig)
    
    plotter.render_all_panels()
    
    # Check that three panel screenshots were created
    assert len(screenshot_calls) == 3
    expected_panels = ["panel_fol.png", "panel_ref.png", "panel_err.png"]
    for panel in expected_panels:
        expected_path = os.path.join(plotter.output_dir, panel)
        assert expected_path in screenshot_calls
    
    # Check that overview image was saved
    assert len(savefig_calls) == 1
    expected_overview = os.path.join(plotter.output_dir, "test_overview2d.png")
    assert savefig_calls[0] == expected_overview


def test_plotter2d_with_warp_factor(tmp_2d_vtk):
    """Test that Plotter2D correctly handles warp_factor_2d configuration."""
    config_no_warp = {
        "warp_factor_2d": 0.0
    }
    plotter_no_warp = Plotter2D(vtk_path=tmp_2d_vtk, config=config_no_warp)
    assert plotter_no_warp.warp_factor_2d is None or plotter_no_warp.warp_factor_2d == 0.0
    
    config_with_warp = {
        "warp_factor_2d": 5.0
    }
    plotter_with_warp = Plotter2D(vtk_path=tmp_2d_vtk, config=config_with_warp)
    assert plotter_with_warp.warp_factor_2d == 5.0


def test_plotter2d_no_abs_error_raises(tmp_path):
    """Test that Plotter2D raises error when no abs_error fields exist."""
    # Create mesh without abs_error fields
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    zz = np.zeros_like(xx)
    
    points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    mesh = pv.PolyData(points)
    mesh = mesh.delaunay_2d()
    
    pts = mesh.points.shape[0]
    mesh.point_data['U_FOL_0'] = np.zeros((pts, 3))
    mesh.point_data['U_FE_0'] = np.ones((pts, 3))
    # No abs_error_0 field!
    
    vtk_file = tmp_path / "mesh_no_error.vtk"
    mesh.save(str(vtk_file))
    
    config = {}
    
    with pytest.raises(ValueError, match="No abs_error"):
        Plotter2D(vtk_path=str(vtk_file), config=config)
