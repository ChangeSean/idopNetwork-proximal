"""Development multiscale information design and Fuller reduced proximal bridge.

No clinical outcome enters the graph, roles, or rank. Positive information is
retained without the old .05 reporting cutoff. Fuller(1) is applied only after
the outcome-blind reduced W space has been determined. Its inference is checked
through refitting the entire estimator, not a naive second-stage standard error.
"""
import numpy as np
from scipy.linalg import eigh
from proximal import proxy_roles,screen,proxy_strength,bridge_information_rank
from run_causal_application import analyse_design_grid
from probe_overlapping_windows import overlapping_grid


def design_grid(cohort):
    original=analyse_design_grid(cohort)
    return original+overlapping_grid(original)


def candidate(res,a,treatment_eligible=None):
    eligible=np.ones(len(res['names']),bool) if treatment_eligible is None else np.asarray(treatment_eligible,bool)
    if eligible.shape!=(len(res['names']),):raise ValueError('Eligibility must follow protein ordering')
    Z,W=proxy_roles(res['Und'],a,eligible,np.linalg.norm(res['latent']['Lam'],axis=1),1,limit_to_treatment=False)
    X,C=res['X'],res['Cov'];n=len(X);alpha_rank=min(.05,n**(-.5))
    row=dict(exposure=res['names'][a],nZ=len(Z),nW=len(W),
             Z=';'.join(res['names'][j] for j in Z),W=';'.join(res['names'][j] for j in W),
             r_pca=res['latent']['r'],r_bridge=0,r_joint=0,r_signal=0,nu=np.nan,
             rank_test_level=alpha_rank,window_fraction=res.get('window_fraction',.2),alpha=res['alpha'])
    if not screen(Z,W,1,overcomplete=True):row['status']='proxy_dimensions';return row,Z,W
    joint=bridge_information_rank(X[:,W],X[:,[a]+Z],None,C,alpha=alpha_rank)
    conditional=bridge_information_rank(X[:,W],X[:,Z],X[:,a],C,alpha=alpha_rank)
    row.update(r_joint=joint['r'],r_signal=conditional['r'],r_bridge=joint['r'],
               joint_pvalues=';'.join(map(str,joint['pvalues'])),
               conditional_pvalues=';'.join(map(str,conditional['pvalues'])))
    if not joint['r'] or not conditional['r']:row['status']='no_shared_direction';return row,Z,W
    if joint['r']!=conditional['r'] or not screen(Z,W,joint['r'],overcomplete=True):
        row['status']='proxy_rank_coverage';return row,Z,W
    row['nu']=proxy_strength(X[:,W],X[:,Z],X[:,a],joint['r'],C)
    row['status']='reported'
    return row,Z,W


def select(grid,a,treatment_eligible=None):
    candidates=[(*candidate(res,a,treatment_eligible),res) for res in grid]
    dimension=max(c[0]['r_joint'] for c in candidates)
    selected=next((c for c in candidates if c[0]['status']=='reported' and c[0]['r_joint']==dimension),
                  next(c for c in candidates if c[0]['r_joint']==dimension))
    row,Z,W,res=selected
    row.update(r_grid=dimension,grid_joint_ranks=';'.join(str(c[0]['r_joint']) for c in candidates),
               grid_conditional_ranks=';'.join(str(c[0]['r_signal']) for c in candidates))
    return row,Z,W,res


def reduced_bridge(Y,A,W,Z,r,C=None,solver='fuller',fuller_alpha=1.):
    """Fuller(1) k-class IV fit on an outcome-blind first-stage W projection.

Exogenous D=(1,A,C), instruments M=(D,Z), endogenous readouts W P.
The LIML root solves R' M_D R versus R' M_M R for R=(Y,W P).
Fuller kappa is LIML kappa minus alpha/(n-rank(M)).
"""
    Y=np.asarray(Y,float);A=np.asarray(A,float);n=len(Y)
    W=np.asarray(W,float).reshape(n,-1);Z=np.asarray(Z,float).reshape(n,-1)
    if solver not in ('fuller','2sls'):raise ValueError('Unknown bridge solver')
    D=np.column_stack([np.ones(n),A]+([C] if C is not None and np.asarray(C).shape[1] else []))
    M=np.column_stack([D,Z]);rankM=np.linalg.matrix_rank(M)
    if n<=rankM:raise ValueError('No residual degrees of freedom')
    What=M@np.linalg.lstsq(M,W,rcond=None)[0]
    residual=What-D@np.linalg.lstsq(D,What,rcond=None)[0]
    _,sv,vt=np.linalg.svd(residual,full_matrices=False)
    if r<1 or r>len(sv) or sv[r-1]<=max(sv[0]*1e-8,1e-12):
        raise ValueError('Requested bridge direction is numerically unavailable')
    P=vt[:r].T;readouts=W@P
    H=np.column_stack([D,readouts]);projected=np.column_stack([D,What@P])
    if solver=='2sls':
        coefficient=np.linalg.lstsq(projected,Y,rcond=None)[0];kappa=1.
    else:
        R=np.column_stack([Y,readouts])
        RD=R-D@np.linalg.lstsq(D,R,rcond=None)[0]
        RM=R-M@np.linalg.lstsq(M,R,rcond=None)[0]
        AA=RD.T@RD;BB=RM.T@RM
        kappa_liml=float(eigh((AA+AA.T)/2,(BB+BB.T)/2,subset_by_index=[0,0],eigvals_only=True)[0])
        kappa=kappa_liml-fuller_alpha/(n-rankM)
        residual_H=H-M@np.linalg.lstsq(M,H,rcond=None)[0]
        residual_Y=Y-M@np.linalg.lstsq(M,Y,rcond=None)[0]
        G=H.T@H-kappa*(residual_H.T@residual_H)
        rhs=H.T@Y-kappa*(residual_H.T@residual_Y)
        coefficient=np.linalg.solve((G+G.T)/2,rhs)
    return dict(estimate=float(coefficient[1]),kappa=kappa,
                nu=proxy_strength(W,Z,A,r,C),rank=r,solver=solver)
