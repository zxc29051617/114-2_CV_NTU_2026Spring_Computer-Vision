"""How to combine several slide-length estimates, tuned on some scenes and tested on others.

    python eval/eval_scale_fusion.py

Inputs (all cached by earlier scripts):
  * Middlebury (26 scenes): outputs/cache/scale/<model>.json   (eval/eval_scale.py)
  * ARKitScenes slide pairs: results/arkit/arkit_pairs.json + outputs/cache/arkit/*.npz (eval/eval_arkit.py)
Every estimate is a log-ratio e = log(B_hat / B).  Strategies:
  geo3      geometric mean of Depth Pro, UniDepthV2, DA-V2-L (the previous default)
  geoall    geometric mean of every model;  medall  median of every model
  ivw       weighted geometric mean, weights 1 / variance of each model's log error on the tuning set
  ivw+bias  the same after removing each model's mean log bias measured on the tuning set
  +size     ivw+bias blended with a size prior from recognised objects (bottle, chair, tv, ...)
Splits: tune on Middlebury -> test on ARKit, and 2-fold by video inside ARKit (both directions averaged).
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stereoguard.objects import SIZE_PRIORS  # noqa: E402

MODELS = ["depthpro", "unidepth", "dav2l", "dav2s", "moge2", "moge2l", "metric3d"]
MB_NAMES = {"moge2": "moge2s"}


def middlebury():
    rows = {}
    for m in MODELS:
        f = os.path.join(ROOT, "outputs", "cache", "scale", f"{MB_NAMES.get(m, m)}.json")
        if not os.path.exists(f):
            continue
        for r in json.load(open(f))["rows"]:
            rows.setdefault(("mb", r["scene"]), {"B": r["B"], "group": "mb"})[m] = r["B_depth"]
    return rows


def size_prior(npz):
    """Slide length implied by typical object sizes: unit-slide geometry, prior / measured."""
    z = np.load(npz, allow_pickle=True)
    f, cx, cy = float(z["f"]), float(z["cx"]), float(z["cy"])
    disp, conf = z["disp"], z["confident"]
    depth1 = np.where(conf & (disp > 0.5), f / np.maximum(disp, 1e-6), np.nan)     # slide = 1 m
    out = []
    for name, mask in zip(z["names"], z["masks"]):
        prior = SIZE_PRIORS.get(str(name))
        if prior is None:
            continue
        ys, xs = np.nonzero(mask)
        if xs.min() < 3 or ys.min() < 3 or xs.max() >= mask.shape[1] - 3 or ys.max() >= mask.shape[0] - 3:
            continue
        m = mask & np.isfinite(depth1)
        if m.sum() < 50:
            continue
        v, u = np.nonzero(m)
        Z = depth1[m]
        X = (u - cx) * Z / f
        Y = (v - cy) * Z / f
        dim, lo, hi = prior
        size = (np.percentile(Y, 95) - np.percentile(Y, 5)) if dim == "height" else \
            (np.percentile(X, 95) - np.percentile(X, 5))
        if size > 0:
            out.append(np.sqrt(lo * hi) / size)
    return float(np.exp(np.median(np.log(out)))) if out else None


def arkit():
    rows = {}
    pairs = json.load(open(os.path.join(ROOT, "results", "arkit", "arkit_pairs.json")))
    for r in pairs:
        d = {"B": r["B"], "group": "ark", "video": r["video"]}
        d.update(r["B_votes"])
        npz = os.path.join(ROOT, "outputs", "cache", "arkit", f"{r['pair']}.npz")
        if os.path.exists(npz):
            d["size"] = size_prior(npz)
        rows[("ark", r["pair"])] = d
    return rows


def logerr(rows, m):
    return np.array([np.log(r[m] / r["B"]) for r in rows if r.get(m)])


def fuse(r, strategy, w=None, bias=None, models=MODELS):
    have = [m for m in models if r.get(m)]
    logs = np.array([np.log(r[m]) for m in have])
    if strategy == "geo3":
        logs = np.array([np.log(r[m]) for m in ("depthpro", "unidepth", "dav2l")])
        return float(np.exp(logs.mean()))
    if strategy == "geoall":
        return float(np.exp(logs.mean()))
    if strategy == "medall":
        return float(np.exp(np.median(logs)))
    ww = np.array([w[m] for m in have])
    if strategy.startswith("ivw+bias"):
        logs = logs - np.array([bias[m] for m in have])
    est = float((ww * logs).sum() / ww.sum())
    if strategy.endswith("+size") and r.get("size"):
        # the size prior is weighted like a model with ~35 % log error
        wp = 1 / 0.35 ** 2
        est = (ww.sum() * est + wp * np.log(r["size"])) / (ww.sum() + wp)
    return float(np.exp(est))


def evaluate(train, test, strategies):
    w = {m: 1 / max(np.var(logerr(train, m)) + np.mean(logerr(train, m)) ** 2, 1e-3) for m in MODELS
         if len(logerr(train, m))}
    wb = {m: 1 / max(np.var(logerr(train, m)), 1e-3) for m in w}
    bias = {m: float(np.mean(logerr(train, m))) for m in w}
    models = [m for m in MODELS if m in w]
    res = {}
    for s in strategies:
        ws = wb if "bias" in s else w
        e = [abs(fuse(r, s, ws, bias, models) / r["B"] - 1) for r in test]
        res[s] = (float(np.mean(e)), float(np.median(e)), float(np.mean(np.array(e) < 0.1)))
    for m in models:
        e = [abs(r[m] / r["B"] - 1) for r in test if r.get(m)]
        res[m] = (float(np.mean(e)), float(np.median(e)), float(np.mean(np.array(e) < 0.1)))
    return res, w, bias


def arkit_distances(ark, half, strategies):
    """Pixel / object errors of the stereo depth on the ARKit pairs with a held-out slide length."""
    pairs = {r["pair"]: r for r in json.load(open(os.path.join(ROOT, "results", "arkit", "arkit_pairs.json")))}
    out = {}
    for s in strategies:
        px, ob = [], []
        for fold in (True, False):
            train = [r for r in ark if (r["video"] in half) == fold]
            test = [r for r in ark if (r["video"] in half) != fold]
            w = {m: 1 / max(np.var(logerr(train, m)), 1e-3) for m in MODELS}
            bias = {m: float(np.mean(logerr(train, m))) for m in MODELS}
            for r in test:
                B_hat = r["B"] if s == "known" else fuse(r, s, w, bias)
                name = [k for k, v in pairs.items() if v["video"] == r["video"] and abs(v["B"] - r["B"]) < 1e-9][0]
                z = np.load(os.path.join(ROOT, "outputs", "cache", "arkit", f"{name}.npz"), allow_pickle=True)
                d, conf, gt, f = z["disp"], z["confident"], z["gt"], float(z["f"])
                depth = np.where(conf & (d > 0.5), f * B_hat / np.maximum(d, 1e-6), np.nan)
                ok = np.isfinite(depth) & np.isfinite(gt)
                rr = np.maximum(depth[ok] / gt[ok], gt[ok] / depth[ok])
                px.append(float(np.mean(rr < 1.05)))
                for m in z["masks"]:
                    core = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    g = gt[core & np.isfinite(gt)]
                    p_ = depth[core & np.isfinite(depth)]
                    if g.size >= 200 and p_.size >= 30:
                        gv, pv = np.percentile(g, 20), np.percentile(p_, 20)
                        ob.append(abs(pv - gv) / gv)
        ob = np.array(ob)
        out[s] = {"pixels_within5": float(np.mean(px)), "objects": int(ob.size),
                  "object_mean_rel": float(ob.mean()) if ob.size else None,
                  "object_median_rel": float(np.median(ob)) if ob.size else None}
    return out


def main():
    mb = list(middlebury().values())
    ark = list(arkit().values())
    strategies = ["geo3", "geoall", "medall", "ivw", "ivw+bias", "ivw+bias+size"]
    out = {"n_middlebury": len(mb), "n_arkit": len(ark),
           "n_arkit_with_size_prior": int(sum(1 for r in ark if r.get("size")))}
    r1, w1, b1 = evaluate(mb, ark, strategies)
    out["tune_middlebury_test_arkit"] = r1
    out["weights_from_middlebury"] = w1
    out["bias_from_middlebury"] = b1
    vids = sorted({r["video"] for r in ark})
    half = set(vids[::2])
    a, b = [r for r in ark if r["video"] in half], [r for r in ark if r["video"] not in half]
    ra, _, _ = evaluate(a, b, strategies)
    rb, _, _ = evaluate(b, a, strategies)
    out["arkit_2fold_by_video"] = {k: tuple((np.array(ra[k]) * len(b) + np.array(rb[k]) * len(a)) / len(ark))
                                   for k in ra}
    rm, _, _ = evaluate(ark, mb, strategies)
    out["tune_arkit_test_middlebury"] = rm
    # distances on the ARKit pairs with a held-out slide length (weights from the other half of the videos)
    out["arkit_distances"] = arkit_distances(ark, half, strategies=("known", "geo3", "ivw+bias"))
    # the deployed fusion: inverse-variance weights and biases fitted on every scene
    allr = mb + ark
    fit = {m: {"weight": float(1 / max(np.var(logerr(allr, m)), 1e-3)), "bias": float(np.mean(logerr(allr, m)))}
           for m in MODELS if len(logerr(allr, m))}
    json.dump({"models": fit, "fitted_on": f"{len(mb)} Middlebury scenes + {len(ark)} ARKitScenes pairs"},
              open(os.path.join(ROOT, "stereoguard", "scale_fusion.json"), "w"), indent=1)
    json.dump(out, open(os.path.join(ROOT, "results", "scale", "fusion.json"), "w"), indent=1, default=float)
    for split in ["tune_middlebury_test_arkit", "arkit_2fold_by_video", "tune_arkit_test_middlebury"]:
        print(split)
        for k, (mean, med, w10) in out[split].items():
            print(f"   {k:16s} mean {100 * mean:5.1f}%  median {100 * med:5.1f}%  within10 {100 * w10:4.0f}%")


if __name__ == "__main__":
    main()
