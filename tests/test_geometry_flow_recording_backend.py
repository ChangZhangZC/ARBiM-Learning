import os
import subprocess
import sys
from types import SimpleNamespace
import pytest
from experiments.geometry_flow.eval import configure_recording_gl

@pytest.mark.parametrize("enabled,platform,expected",[(True,"linux","egl"),(False,"linux","glfw"),(True,"darwin","glfw")])
def test_backend_selection(monkeypatch,enabled,platform,expected):
    monkeypatch.setattr(sys,"platform",platform)
    monkeypatch.delitem(sys.modules,"mujoco",raising=False)
    monkeypatch.delitem(sys.modules,"OpenGL.platform",raising=False)
    monkeypatch.setenv("MUJOCO_GL","glfw")
    monkeypatch.setenv("PYOPENGL_PLATFORM","glx")
    configure_recording_gl(enabled)
    assert os.environ["MUJOCO_GL"]==expected
    assert os.environ["PYOPENGL_PLATFORM"]==("egl" if expected=="egl" else "glx")

def test_reject_late_backend_change(monkeypatch):
    monkeypatch.setattr(sys,"platform","linux")
    monkeypatch.setitem(sys.modules,"mujoco",SimpleNamespace(GLContext=type("GLContext",(),{"__module__":"mujoco.glfw"})))
    with pytest.raises(RuntimeError,match="imported before EGL"):
        configure_recording_gl(True)

def test_entry_import_does_not_initialize_gl():
    code="import sys; import experiments.geometry_flow.eval; assert 'mujoco' not in sys.modules; assert 'OpenGL.platform' not in sys.modules; assert 'rcs' not in sys.modules"
    subprocess.run([sys.executable,"-c",code],check=True)
