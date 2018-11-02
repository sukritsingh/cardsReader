import numpy as np

def convert_u_to_p(u, kT=2.479):
    p = np.exp(-(u-u.mean())/kT)
    p /= p.sum()
    return p

def calc_entropy(p):
    inds = np.where(p>0)[0]
    s = -p[inds].dot(np.log(p[inds]))
    return s

def calc_rel_ent(p, q):
    p_inds = np.where(p>0)[0]
    q_inds = np.where(q>0)[0]
    inds = np.intersect1d(p_inds, q_inds)
    rel_ent = p[inds].dot(np.log(p[inds]/q[inds]))
    return rel_ent
    
def calc_js_divergence(p, q):
    m = 0.5*(p+q)
    js = 0.5*calc_rel_ent(p,m) + 0.5*calc_rel_ent(q,m)
    return js
    