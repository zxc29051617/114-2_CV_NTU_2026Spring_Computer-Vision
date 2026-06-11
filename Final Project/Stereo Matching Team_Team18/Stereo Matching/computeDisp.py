import numpy as np
import cv2
import cv2.ximgproc as xip


# popcount lookup: number of set bits in each byte value 0..255
_POPCOUNT = np.array([bin(i).count('1') for i in range(256)], dtype=np.float32)
def popcount_u32(x):
    x = x - ((x >> 1) & np.uint32(0x55555555))
    x = (x & np.uint32(0x33333333)) + ((x >> 2) & np.uint32(0x33333333))
    return (((x + (x >> 4)) & np.uint32(0x0F0F0F0F)) * np.uint32(0x01010101)) >> np.uint32(24)

def census_transform_u32(img_gray, win_r=2):
    h, w = img_gray.shape
    padded = cv2.copyMakeBorder(
        img_gray, win_r, win_r, win_r, win_r,
        cv2.BORDER_REFLECT
    )

    census = np.zeros((h, w), dtype=np.uint32)
    bit = 0

    for dy in range(-win_r, win_r + 1):
        for dx in range(-win_r, win_r + 1):
            if dy == 0 and dx == 0:
                continue

            shifted = padded[
                win_r + dy: win_r + dy + h,
                win_r + dx: win_r + dx + w
            ]

            census |= ((shifted < img_gray).astype(np.uint32) << bit)
            bit += 1

    return census


def compute_cost_volume_u32(census_ref, census_tgt, max_disp, direction='left'):
    h, w = census_ref.shape
    cost = np.empty((h, w, max_disp + 1), dtype=np.uint8)

    for d in range(max_disp + 1):
        if d == 0:
            cost[:, :, d] = popcount_u32(np.bitwise_xor(census_ref, census_tgt)).astype(np.uint8)
            continue

        if direction == 'left':
            cost[:, d:, d] = popcount_u32(
                np.bitwise_xor(census_ref[:, d:], census_tgt[:, :w-d])
            ).astype(np.uint8)

            cost[:, :d, d] = popcount_u32(
                np.bitwise_xor(census_ref[:, :d], census_tgt[:, 0:1])
            ).astype(np.uint8)

        else:
            cost[:, :w-d, d] = popcount_u32(
                np.bitwise_xor(census_ref[:, :w-d], census_tgt[:, d:])
            ).astype(np.uint8)

            cost[:, w-d:, d] = popcount_u32(
                np.bitwise_xor(census_ref[:, w-d:], census_tgt[:, w-1:w])
            ).astype(np.uint8)

    return cost


def derive_right_from_left(cost_L, max_disp):
    """
    Derive the RIGHT cost volume from the LEFT one without a second
    XOR + popcount pass. Because the Census matching cost (Hamming distance) is
    symmetric, the raw right cost is just a per-disparity shift of the raw left
    cost:

        cost_R[y, x, d] = cost_L[y, x + d, d]

    For the rightmost d columns (x + d >= w) there is no left-cost value to
    borrow, so we clamp to the last valid column -- consistent with the
    closest-valid out-of-bound rule. This is exact on every interior pixel; only
    the thin right-boundary strip differs, and that strip is discarded anyway by
    the left-right consistency check / hole filling later.

    MUST be called on the RAW (pre-aggregation) left cost volume.
    """
    _, w, _ = cost_L.shape
    cost_R = np.empty_like(cost_L)
    for d in range(max_disp + 1):
        if d > 0:
            cost_R[:, :w - d, d] = cost_L[:, d:, d]          # interior (exact)
            cost_R[:, w - d:, d] = cost_L[:, w - 1:w, d]     # clamp right edge
        else:
            cost_R[:, :, d] = cost_L[:, :, d]
    return cost_R


def build_cost_volumes(Il, Ir, max_disp, win_r=2):
    gray_l = cv2.cvtColor(Il.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(Ir.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    census_l = census_transform_u32(gray_l, win_r=win_r)
    census_r = census_transform_u32(gray_r, win_r=win_r)

    cost_L = compute_cost_volume_u32(census_l, census_r, max_disp, direction='left')
    cost_R = derive_right_from_left(cost_L, max_disp)

    return cost_L.astype(np.float32), cost_R.astype(np.float32)


def aggregate_cost_volumes(cost_L, cost_R, Il, Ir, radius=9, eps=0.0001, scale=2):
    """
    Cost Aggregation with sub-sampling trick: filter at half resolution,
    then upsample. Matches the spirit of fastGuidedFilter manually.
    """
    h, w = Il.shape[:2]
    h_s, w_s = h // scale, w // scale

    # ย่อ guide และ cost
    guide_l_s = cv2.resize(Il.astype(np.uint8), (w_s, h_s), interpolation=cv2.INTER_LINEAR)
    guide_r_s = cv2.resize(Ir.astype(np.uint8), (w_s, h_s), interpolation=cv2.INTER_LINEAR)
    cost_L_s = cv2.resize(cost_L, (w_s, h_s), interpolation=cv2.INTER_LINEAR)
    cost_R_s = cv2.resize(cost_R, (w_s, h_s), interpolation=cv2.INTER_LINEAR)

    # Filter ที่ resolution ครึ่ง (เร็วขึ้น 4 เท่า)
    agg_L_s = xip.guidedFilter(guide=guide_l_s, src=cost_L_s,
                               radius=radius // scale, eps=eps)
    agg_R_s = xip.guidedFilter(guide=guide_r_s, src=cost_R_s,
                               radius=radius // scale, eps=eps)

    # ขยายกลับเป็นขนาดเดิม
    agg_L = cv2.resize(agg_L_s, (w, h), interpolation=cv2.INTER_LINEAR)
    agg_R = cv2.resize(agg_R_s, (w, h), interpolation=cv2.INTER_LINEAR)

    return agg_L, agg_R


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # Census cost = Local binary pattern -> Hamming distance, both directions.
    # Out-of-bound pixels use the cost of the closest valid pixel.
    cost_L, cost_R = build_cost_volumes(Il, Ir, max_disp, win_r=2)

    # >>> Cost Aggregation
    # Guided filter on each disparity slice, guided by the reference image.
    agg_L, agg_R = aggregate_cost_volumes(cost_L, cost_R, Il, Ir)

    # >>> Disparity Optimization
    # Winner-take-all: pick the disparity with the minimum aggregated cost.
    disp_L = np.argmin(agg_L, axis=2).astype(np.int32)   # (H, W)
    disp_R = np.argmin(agg_R, axis=2).astype(np.int32)   # (H, W)

    # >>> Disparity Refinement
    # (a) Left-Right consistency check (strict: keep only if disp_L == disp_R at match)
    yy, xx = np.indices((h, w))
    xr = xx - disp_L                       # corresponding column in the right map
    valid = xr >= 0                        # must land inside the image to be valid
    xr_clamped = np.clip(xr, 0, w - 1)
    disp_R_at = disp_R[yy, xr_clamped]     # gather right disparity at the match
    consistent = valid & (disp_L == disp_R_at)

    labels = disp_L.astype(np.float32)
    labels[~consistent] = -1               # inconsistent pixels become holes

    # (b) Hole filling: fill from nearest valid disparity on the left (FL) and
    #     right (FR), then take the pixel-wise minimum. Boundary holes padded with
    #     max_disp so a leading run of holes resolves to the far (background) value.
    #     Vectorized as a running last-valid-column index per row.
    col = np.arange(w)[None, :]
    is_valid = labels != -1
    idx_l = np.maximum.accumulate(np.where(is_valid, col, -1), axis=1)
    FL = np.where(idx_l >= 0,
                  np.take_along_axis(labels, np.clip(idx_l, 0, w - 1), axis=1),
                  max_disp)
    idx_r = np.minimum.accumulate(np.where(is_valid, col, w)[:, ::-1], axis=1)[:, ::-1]
    FR = np.where(idx_r < w,
                  np.take_along_axis(labels, np.clip(idx_r, 0, w - 1), axis=1),
                  max_disp)
    labels = np.minimum(FL, FR)

    # (c) Weighted median filter, guided by the left color image. r=7 (rather
    #     than 9) over-smooths thin structures less while staying noise-free in
    #     hole-filled regions.
    labels = xip.weightedMedianFilter(joint=Il.astype(np.uint8),
                                      src=labels.astype(np.uint8), r=7)

    return labels.astype(np.uint8)