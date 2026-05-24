import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] != N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    x = u[:, 0]
    y = u[:, 1]
    xp = v[:, 0]
    yp = v[:, 1]
    zeros = np.zeros(N)
    ones = np.ones(N)

    A = np.empty((2 * N, 9), dtype=np.result_type(u, v, float))
    A[0::2] = np.column_stack((-x, -y, -ones, zeros, zeros, zeros, x * xp, y * xp, xp))
    A[1::2] = np.column_stack((zeros, zeros, zeros, -x, -y, -ones, x * yp, y * yp, yp))

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    if H[2, 2] != 0:
        H = H / H[2, 2]

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape

    x, y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))

    dst_pixels = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size, dtype=x.dtype)))

    if direction == 'b':

        H_inv = np.linalg.inv(H)
        
        src_pixels = dst_pixels @ H_inv.T
        src_pixels /= src_pixels[:, 2:3]
        u = src_pixels[:, 0].reshape(ymax - ymin, xmax - xmin)
        v = src_pixels[:, 1].reshape(ymax - ymin, xmax - xmin)

        mask = (u >= 0) & (u < w_src) & (v >= 0) & (v < h_src)

        sampled_src = src[v[mask].astype(np.intp), u[mask].astype(np.intp)]

        dst[y[mask], x[mask]] = sampled_src

    elif direction == 'f':
        denom = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denom
        v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denom

        mask = (u >= 0) & (u < w_dst) & (v >= 0) & (v < h_dst)

        u_valid = u[mask].astype(np.intp)
        v_valid = v[mask].astype(np.intp)
        x_valid = x[mask]
        y_valid = y[mask]

        dst[v_valid, u_valid] = src[y_valid, x_valid]

    return dst 
