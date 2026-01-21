
import numpy as np
import chgnet.graph.converter
from chgnet.graph import cygraph
from chgnet.model import CHGNet
from pymatgen.core import Structure, Lattice

# --- Monkeypatch Start ---
_original_make_graph = cygraph.make_graph

def make_graph_patched(center_index, num_edges, neighbor_index, image, distance, n_atoms):
    print(
        "make_graph args: "
        f"center_index={type(center_index).__name__}/{getattr(center_index, 'dtype', None)}, "
        f"num_edges={type(num_edges).__name__}, "
        f"neighbor_index={type(neighbor_index).__name__}/{getattr(neighbor_index, 'dtype', None)}, "
        f"image={type(image).__name__}/{getattr(image, 'dtype', None)}, "
        f"distance={type(distance).__name__}/{getattr(distance, 'dtype', None)}, "
        f"n_atoms={type(n_atoms).__name__}"
    )

    def ensure_int64(arr):
        return np.ascontiguousarray(arr, dtype=np.int64)

    def ensure_float64(arr):
        return np.ascontiguousarray(arr, dtype=np.float64)

    center_index = ensure_int64(center_index)
    neighbor_index = ensure_int64(neighbor_index)
    # image buffer was reported as double, but Cython expects int64
    image = ensure_int64(image)
    distance = ensure_float64(distance)
    num_edges = int(num_edges)
    n_atoms = int(n_atoms)

    return _original_make_graph(
        center_index, num_edges, neighbor_index, image, distance, n_atoms
    )

# Apply patch
chgnet.graph.converter.make_graph = make_graph_patched
print("Patched make_graph for Windows int64 compatibility")
# --- Monkeypatch End ---

s = Structure(Lattice.cubic(4.0), ['Li', 'O'], [[0,0,0], [0.5,0.5,0.5]])
m = CHGNet.load()
res = m.predict_structure(s, batch_size=1)
print("Prediction successful:", res)
