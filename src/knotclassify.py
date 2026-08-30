"""
knotclassify.py - Classical knot classification for simulated polymer chains.

Two independent pieces, chained together:
    1. KMT (Koniaris-Muthukumar-Taylor) geometric simplification.
    2. Alexander polynomial via the matrix method (Alexander, 1928),
       computed from a 2D projection's crossing diagram.

Validated against literature-known values for: Unknot, Trefoil (two
independent parametrizations), Figure-8, Cinquefoil (5_1).

Bug fix: during broader validation (04_broader_validation.ipynb):
the Alexander matrix table originally had 'i' and 'j' roles swapped. This
gave the correct answer for the trefoil purely by luck (its symmetric
structure doesn't discriminate between table permutations - verified: all
6 possible role-to-coefficient permutations "match" trefoil alone). Found
via testing against cinquefoil and figure-8, root-caused via systematic
brute-force testing against literature values. Correct table (see
alexander_polynomial() below): the over-arc coefficient (1-t) is
handedness-independent; only the two under-arc coefficients (t and -1)
swap between i and k depending on crossing sign.
"""

import numpy as np
import sympy as sp



# KMT geometric simplification


def _segment_triangle_intersect(p0, p1, v0, v1, v2, eps=1e-9):
    d = p1 - p0
    seg_len = np.linalg.norm(d)
    if seg_len < eps:
        return False
    d = d / seg_len
    e1 = v1 - v0
    e2 = v2 - v0
    h = np.cross(d, e2)
    a = np.dot(e1, h)
    if abs(a) < eps:
        return False
    f = 1.0 / a
    s = p0 - v0
    u = f * np.dot(s, h)
    if u < -eps or u > 1.0 + eps:
        return False
    q = np.cross(s, e1)
    v = f * np.dot(d, q)
    if v < -eps or u + v > 1.0 + eps:
        return False
    t = f * np.dot(e2, q)
    return -eps < t < seg_len + eps


def kmt_simplify(positions, max_passes=50):
    """KMT polygon simplification - topology-preserving by construction.
    Excludes the two segments forming the candidate triangle, and the two
    segments that merely touch it at a single shared vertex (both
    non-blocking geometrically, but can trigger spurious floating-point
    'intersections' at the shared corner if not explicitly excluded)."""
    pts_arr = positions.copy()
    for _pass in range(max_passes):
        removed_any = False
        n = len(pts_arr)
        if n <= 4:
            break
        idx = 0
        while idx < n and n > 4:
            i = idx
            prev = (i - 1) % n
            nxt = (i + 1) % n
            v0, v1, v2 = pts_arr[prev], pts_arr[i], pts_arr[nxt]
            blocked = False
            for j in range(n):
                if j in (prev, i, nxt):
                    continue
                jn = (j + 1) % n
                if jn == prev:
                    continue
                if _segment_triangle_intersect(pts_arr[j], pts_arr[jn], v0, v1, v2):
                    blocked = True
                    break
            if not blocked:
                pts_arr = np.delete(pts_arr, i, axis=0)
                n -= 1
                removed_any = True
            else:
                idx += 1
        if not removed_any:
            break
    return pts_arr





# 2D projection + crossing extraction


def _segment_2d_intersect(p1, p2, p3, p4, eps=1e-9):
    d1 = p2 - p1
    d2 = p4 - p3
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < eps:
        return False, None, None
    diff = p3 - p1
    t = (diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = (diff[0] * d1[1] - diff[1] * d1[0]) / denom
    if eps < t < 1 - eps and eps < u < 1 - eps:
        return True, t, u
    return False, None, None


def extract_crossings(positions_3d, eps=1e-9):
    N = len(positions_3d)
    xy = positions_3d[:, :2]
    z = positions_3d[:, 2]
    crossings = []
    for i in range(N):
        i2 = (i + 1) % N
        for j in range(i + 1, N):
            j2 = (j + 1) % N
            if j == i or j2 == i or j == i2 or j2 == i2:
                continue
            hit, t, u = _segment_2d_intersect(xy[i], xy[i2], xy[j], xy[j2], eps)
            if not hit:
                continue
            z_i = z[i] + t * (z[i2] - z[i])
            z_j = z[j] + u * (z[j2] - z[j])
            dir_i = xy[i2] - xy[i]
            dir_j = xy[j2] - xy[j]
            if z_i > z_j:
                over_dir, under_dir = dir_i, dir_j
                pos_over, pos_under = i + t, j + u
            else:
                over_dir, under_dir = dir_j, dir_i
                pos_over, pos_under = j + u, i + t
            sign = 1 if (over_dir[0] * under_dir[1] - over_dir[1] * under_dir[0]) > 0 else -1
            crossings.append(dict(pos_over=pos_over, pos_under=pos_under, sign=sign))
    return crossings


def _random_rotation(rng):
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))
    return Q


def best_projection_crossings(positions_3d, n_tries=200, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_tries):
        R = _random_rotation(rng)
        rotated = positions_3d @ R.T
        c = extract_crossings(rotated)
        if best is None or len(c) < len(best):
            best = c
    return best






# Gauss code

def gauss_code(crossings):
    events = []
    for idx, c in enumerate(crossings):
        events.append((c['pos_over'], idx, 1))
        events.append((c['pos_under'], idx, -1))
    events.sort(key=lambda e: e[0])
    label_map = {}
    next_label = 1
    code = []
    for pos, idx, sign in events:
        if idx not in label_map:
            label_map[idx] = next_label
            next_label += 1
        code.append(sign * label_map[idx])
    return code






# Alexander polynomial via the matrix method (fixed table)


def alexander_polynomial(crossings):
    """Alexander polynomial via the matrix method. See module docstring
    for the bug-fix details. Correct table:
        Left-handed:  i:t,  j:(1-t), k:-1
        Right-handed: i:-1, j:(1-t), k:t
    (over-arc coefficient (1-t) is handedness-independent; only the
    under-arc coefficients t/-1 swap between i and k with crossing sign)
    """
    n = len(crossings)
    if n == 0:
        return {0: 1}

    events = []
    for idx, c in enumerate(crossings):
        events.append((c['pos_over'], idx, '+'))
        events.append((c['pos_under'], idx, '-'))
    events.sort(key=lambda e: e[0])

    arc_id = 0
    arc_incoming = {}
    arc_outgoing = {}
    arc_over = {}
    for pos, idx, kind in events:
        if kind == '+':
            arc_over[idx] = arc_id
        else:
            arc_incoming[idx] = arc_id
            arc_id += 1
            arc_outgoing[idx] = arc_id
    n_arcs = arc_id
    for d in (arc_outgoing, arc_over):
        for k in d:
            if d[k] == n_arcs:
                d[k] = 0

    t = sp.symbols('t')
    M = sp.zeros(n, n)
    for row, c in enumerate(crossings):
        i_arc, k_arc, j_arc = arc_incoming[row], arc_outgoing[row], arc_over[row]
        if c['sign'] > 0:   # right-handed
            M[row, i_arc] += -1
            M[row, j_arc] += (1 - t)
            M[row, k_arc] += t
        else:               # left-handed
            M[row, i_arc] += t
            M[row, j_arc] += (1 - t)
            M[row, k_arc] += -1

    reduced = M[:-1, :-1]
    det = sp.factor(sp.expand(reduced.det()))
    poly = sp.Poly(sp.expand(det), t)
    coeffs = poly.all_coeffs()[::-1]
    lowest_nonzero = next(i for i, c in enumerate(coeffs) if c != 0)
    normalized = {i - lowest_nonzero: coeffs[i] for i in range(len(coeffs)) if coeffs[i] != 0}

    shift = max(normalized.keys()) // 2
    centered = {k - shift: v for k, v in normalized.items()}

    at_1 = sum(centered.values())
    if at_1 < 0:
        centered = {k: -v for k, v in centered.items()}
    return centered


def format_polynomial(coeffs):
    terms = []
    for power in sorted(coeffs.keys(), reverse=True):
        c = coeffs[power]
        if c == 0:
            continue
        sign = '+' if c > 0 else '-'
        mag = abs(c)
        if power == 0:
            term = f"{mag}"
        elif power == 1:
            term = f"{mag}t" if mag != 1 else "t"
        elif power == -1:
            term = f"{mag}/t" if mag != 1 else "1/t"
        else:
            term = f"{mag}t^{power}" if mag != 1 else f"t^{power}"
        terms.append((sign, term))
    if not terms:
        return "0"
    s = terms[0][1] if terms[0][0] == '+' else f"-{terms[0][1]}"
    for sign, term in terms[1:]:
        s += f" {sign} {term}"
    return s


def classify_knot(positions_3d, n_projection_tries=200, seed=0, simplify=True):
    chain = kmt_simplify(positions_3d) if simplify else positions_3d
    crossings = best_projection_crossings(chain, n_tries=n_projection_tries, seed=seed)
    poly = alexander_polynomial(crossings)
    return poly, crossings, chain
