from scipy.stats import rankdata

fname = 'newlogistic'

kshaps_indep = np.load(fname+'_kshap_indep.npy')
sss_indep = np.load(fname+'_ss_indep.npy')
kshaps_dep = np.load(fname+'_kshap_dep.npy')
sss_dep = np.load(fname+'_ss_dep.npy')



#%%

kshaps_dep = np.array(kshaps_dep)
kshaps_indep = np.array(kshaps_indep)

sss_dep = np.array(sss_dep)
sss_indep = np.array(sss_indep)

means_kshap_dep = np.mean(kshaps_dep,axis=1)
means_kshap_indep = np.mean(kshaps_indep,axis=1)
vars_kshap_dep = np.var(kshaps_dep,axis=1)
vars_kshap_indep = np.var(kshaps_indep,axis=1)

means_ss_dep = np.mean(sss_dep,axis=1)
means_ss_indep = np.mean(sss_indep,axis=1)
vars_ss_dep = np.var(sss_dep,axis=1)
vars_ss_indep = np.var(sss_indep,axis=1)

var_reducs_indep = 1-np.array([[vars_ss_indep[i][0]/vars_ss_indep[i][1] for i in range(n_pts)],
                    #[vars_kshap_indep[i][2]/vars_kshap_indep[i][0] for i in range(n_pts)],
                    [vars_kshap_indep[i][6]/vars_kshap_indep[i][0] for i in range(n_pts)]])

reducs_indep_25 = np.reshape(np.quantile(var_reducs_indep,0.25,axis=1).T,[2*d])
reducs_indep_50 = np.reshape(np.quantile(var_reducs_indep,0.50,axis=1).T,[2*d])
reducs_indep_75 = np.reshape(np.quantile(var_reducs_indep,0.75,axis=1).T,[2*d])

xpts = np.repeat( np.arange(0,d), 2) + np.tile(np.array([-0.1,0.1]),d)
plt.errorbar(xpts,reducs_indep_50,yerr=np.array([reducs_indep_50-reducs_indep_25,reducs_indep_75-reducs_indep_50]),fmt='o')
plt.ylim([-1,1])


var_reducs_dep = 1-np.array([[vars_ss_dep[i][0]/vars_ss_dep[i][1] for i in range(n_pts)],
                    #[vars_kshap_dep[i][2]/vars_kshap_dep[i][0] for i in range(n_pts)],
                    [vars_kshap_dep[i][6]/vars_kshap_dep[i][0] for i in range(n_pts)]])


reducs_dep_25 = np.reshape(np.quantile(var_reducs_dep,0.25,axis=1).T,[2*d])
reducs_dep_50 = np.reshape(np.quantile(var_reducs_dep,0.50,axis=1).T,[2*d])
reducs_dep_75 = np.reshape(np.quantile(var_reducs_dep,0.75,axis=1).T,[2*d])

plt.errorbar(xpts,reducs_dep_50,yerr=np.array([reducs_dep_50-reducs_dep_25,reducs_dep_75-reducs_dep_50]),fmt='o')
plt.ylim([-0.25,1])
plt.axhline(y=0,color='black')


#%%


ss_rank_cors_indep = []
kshap_rank_cors_indep = []

cv_rank_cors_indep = []
boot_rank_cors_indep = []
group_rank_cors_indep = []
wls_rank_cors_indep = []


ss_rank_cors_dep = []
kshap_rank_cors_dep = []

cv_rank_cors_dep = []
boot_rank_cors_dep = []
group_rank_cors_dep = []
wls_rank_cors_dep = []


for i in range(n_pts):
    rankmat = np.array([rankdata(sss_indep[i][j][1]) for j in range(nsim_per_point)])
    ss_rank_cors_indep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(sss_dep[i][j][1]) for j in range(nsim_per_point)])
    ss_rank_cors_dep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(sss_indep[i][j][0]) for j in range(nsim_per_point)])
    cv_rank_cors_indep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(sss_dep[i][j][0]) for j in range(nsim_per_point)])
    cv_rank_cors_dep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)  


    rankmat = np.array([rankdata(kshaps_indep[i][j][0]) for j in range(nsim_per_point)])
    kshap_rank_cors_indep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_dep[i][j][0]) for j in range(nsim_per_point)])
    kshap_rank_cors_dep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_indep[i][j][2]) for j in range(nsim_per_point)])
    boot_rank_cors_indep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_dep[i][j][2]) for j in range(nsim_per_point)])
    boot_rank_cors_dep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_indep[i][j][4]) for j in range(nsim_per_point)])
    group_rank_cors_indep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_dep[i][j][4]) for j in range(nsim_per_point)])
    group_rank_cors_dep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_indep[i][j][6]) for j in range(nsim_per_point)])
    wls_rank_cors_indep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)

    rankmat = np.array([rankdata(kshaps_dep[i][j][6]) for j in range(nsim_per_point)])
    wls_rank_cors_dep.append(np.sum(np.abs(rankmat[:,None,:]-rankmat[None,:,:]))/n_pts**2)


plt.hist([ss_rank_cors_dep,cv_rank_cors_dep])
plt.hist([ss_rank_cors_indep,cv_rank_cors_indep])


plt.hist([kshap_rank_cors_dep,wls_rank_cors_dep])
plt.hist([kshap_rank_cors_indep,wls_rank_cors_indep])
