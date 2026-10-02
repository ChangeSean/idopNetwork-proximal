"""Joint readout representation followed by conditional bridge information.

Estimate the W signal space using (A,Z) before removing A. Test identifying
information after projecting observed W into that joint space and conditioning
on A. This uses A to learn readout coordinates without adjusting away A from
the causal outcome equation. Outcome data never enter this representation.
"""
import numpy as np
from proximal import bridge_information_rank,proxy_roles,proxy_strength,screen
from multiscale_bridge import design_grid


def joint_projection(X,a,Z,W,r,C=None):
    n=len(X)
    base=np.column_stack([np.ones(n)]+([C] if C is not None and C.shape[1] else []))
    instruments=np.column_stack([base,X[:,a],X[:,Z]])
    fitted=instruments@np.linalg.lstsq(instruments,X[:,W],rcond=None)[0]
    residual=fitted-base@np.linalg.lstsq(base,fitted,rcond=None)[0]
    _,sv,vt=np.linalg.svd(residual,full_matrices=False)
    if r<1 or r>len(sv) or sv[r-1]<=max(sv[0]*1e-8,1e-12):raise ValueError('Joint readout space unavailable')
    return vt[:r].T


def candidate(res,a,r,treatment_eligible=None):
    X,C=res['X'],res['Cov'];level=min(.05,len(X)**(-.5))
    eligible=np.ones(len(res['names']),bool) if treatment_eligible is None else np.asarray(treatment_eligible,bool)
    Z,W=proxy_roles(res['Und'],a,eligible,np.linalg.norm(res['latent']['Lam'],axis=1),1,limit_to_treatment=False)
    row=dict(exposure=res['names'][a],nZ=len(Z),nW=len(W),Z=';'.join(res['names'][j] for j in Z),
             W=';'.join(res['names'][j] for j in W),r_bridge=r,r_signal=0,r_joint=0,
             nu=np.nan,nu_raw=np.nan,window_fraction=res.get('window_fraction',.2),alpha=res['alpha'])
    if not screen(Z,W,max(r,1),overcomplete=True):row['status']='proxy_dimensions';return row,Z,W
    joint=bridge_information_rank(X[:,W],X[:,[a]+Z],None,C,alpha=level)
    row['r_joint']=joint['r']
    if not r or joint['r']!=r:row['status']='proxy_rank_coverage';return row,Z,W
    P=joint_projection(X,a,Z,W,r,C)
    conditional=bridge_information_rank(X[:,W]@P,X[:,Z],X[:,a],C,alpha=level)
    row['r_signal']=conditional['r']
    row['nu_raw']=proxy_strength(X[:,W],X[:,Z],X[:,a],r,C)
    row['nu']=proxy_strength(X[:,W]@P,X[:,Z],X[:,a],r,C)
    if conditional['r']!=r:row['status']='proxy_rank_coverage';return row,Z,W
    row['status']='reported' if row['nu']>=.05 else 'weak_bridge_information'
    return row,Z,W


def select(grid,a,treatment_eligible=None):
    from multiscale_bridge import candidate as marginal_candidate
    marginal=[marginal_candidate(res,a,treatment_eligible) for res in grid]
    r=max(row['r_joint'] for row,_,_ in marginal)
    candidates=[(*candidate(res,a,r,treatment_eligible),res) for res in grid]
    chosen=next((c for c in candidates if c[0]['status']=='reported'),
                next((c for c in candidates if c[0]['r_joint']==r),candidates[0]))
    row,Z,W,res=chosen;row['r_grid']=r
    return row,Z,W,res


def estimate(Y,A,W,Z,r,C=None):
    n=len(Y);base=np.column_stack([np.ones(n)]+([C] if C is not None and C.shape[1] else []))
    M=np.column_stack([base,A,Z]);fitted=M@np.linalg.lstsq(M,W,rcond=None)[0]
    joint=fitted-base@np.linalg.lstsq(base,fitted,rcond=None)[0]
    _,sv,vt=np.linalg.svd(joint,full_matrices=False)
    if r>len(sv) or sv[r-1]<=max(sv[0]*1e-8,1e-12):raise ValueError('Joint readout projection unavailable')
    P=vt[:r].T
    fitted_reduced=fitted@P
    H=np.column_stack([base,A,fitted_reduced])
    if np.linalg.matrix_rank(H)<H.shape[1]:raise ValueError('Bridge directions unavailable beyond exposure')
    return float(np.linalg.lstsq(H,Y,rcond=None)[0][base.shape[1]])


def select_joint_space(grid,a,treatment_eligible=None):
    """Select joint information and quantify conditional information separately.

Identification assumes conditional completeness of the selected proxy design;
finite-sample rank or strength tests do not establish this condition. Among
designs spanning the common joint rank with enough treatment proxies, choose
the largest last retained joint canonical root, in fixed grid order for ties.
Conditional roots and raw/reduced strength accompany every selected design.
"""
    from multiscale_bridge import candidate as marginal_candidate
    marginals=[(*marginal_candidate(res,a,treatment_eligible),res) for res in grid]
    r=max(row['r_joint'] for row,_,_,_ in marginals)
    options=[]
    for row,Z,W,res in marginals:
        if not r or row['r_joint']!=r or min(len(Z),len(W))<r:continue
        X,C=res['X'],res['Cov'];level=min(.05,len(X)**(-.5))
        info=bridge_information_rank(X[:,W],X[:,[a]+Z],None,C,alpha=level)
        P=joint_projection(X,a,Z,W,r,C)
        conditional=bridge_information_rank(X[:,W]@P,X[:,Z],X[:,a],C,alpha=level)
        reduced=X[:,W]@P
        row=dict(row,status='reported',r_grid=r,r_bridge=r,r_signal=conditional['r'],
                 joint_root=float(info['roots'][r-1]),
                 conditional_roots=';'.join(map(str,conditional['roots'])),
                 nu_raw=proxy_strength(X[:,W],X[:,Z],X[:,a],r,C),
                 nu=proxy_strength(reduced,X[:,Z],X[:,a],r,C),
                 conditional_rank_agreement=conditional['r']==r)
        options.append((row,Z,W,res))
    if options:
        return max(options,key=lambda option:option[0]['joint_root'])
    row,Z,W,res=next(c for c in marginals if c[0]['r_joint']==r)
    row=dict(row,status='proxy_dimensions' if r else 'no_shared_direction',r_grid=r)
    return row,Z,W,res
