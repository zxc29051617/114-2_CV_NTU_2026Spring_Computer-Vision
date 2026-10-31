"""Turn a depth map into a coloured triangle mesh (GLB) for the 3D viewer."""
import numpy as np


def depth_to_glb(depth, image_bgr, f, cx, cy, path, step=3, max_depth=None, max_jump=0.04):
    import trimesh
    d = depth[::step, ::step]
    col = image_bgr[::step, ::step, ::-1]
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    u = u * step
    v = v * step
    ok = np.isfinite(d) & (d > 0)
    if max_depth:
        ok &= d < max_depth
    z = np.where(ok, d, 0)
    X = (u - cx) * z / f
    Y = -(v - cy) * z / f
    Z = -z
    verts = np.stack([X, Y, Z], -1).reshape(-1, 3)
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    za, zb, zc, ze = z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]
    oka, okb, okc, oke = ok[:-1, :-1], ok[:-1, 1:], ok[1:, :-1], ok[1:, 1:]

    def smooth(*zs):
        zz = np.stack(zs)
        return (zz.max(0) - zz.min(0)) < max_jump * zz.min(0).clip(1e-6) * 3
    t1 = oka & okb & okc & smooth(za, zb, zc)
    t2 = okb & oke & okc & smooth(zb, ze, zc)
    faces = np.concatenate([np.stack([a[t1], c[t1], b[t1]], 1), np.stack([b[t2], c[t2], e[t2]], 1)])
    colors = np.concatenate([col.reshape(-1, 3), np.full((h * w, 1), 255, np.uint8)], 1)
    mesh = trimesh.Trimesh(verts, faces, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    mesh.export(path)
    return path
