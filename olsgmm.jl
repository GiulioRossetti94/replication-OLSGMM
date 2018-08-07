using Distributions
function olsgmm(lhv,rhv,lags,weight)
    global Exxprim
    global inner
    
    
    if size(rhv,1) != size(lhv,1)
        sLHV = size(lhv)
        sRHV = size(rhv)
        println("olsgmm: left and right sides must have same number of rows. Current rows are: $sLHV and $sRHV")
    end
    
    T = size(lhv,1)
    N = size(lhv,2)
    K = size(rhv,2)
    sebv = zeros(K,N)
    Exxprim = inv((rhv'*rhv)/T)
    bv = rhv\lhv
    F = zeros(N,3)
    
    if weight == -1
        sebv = NaN
        R2v = NaN
        R2vadj = NaN
        v = NaN
        F = NaN
    else
        errv = lhv -rhv*bv
        s2 = mean(errv.^2)
        vary = lhv - ones(T,1)*mean(lhv)
        vary = mean(vary.^2)
        
        R2v = (1-s2 ./ vary)'
        R2vadj = (1-(s2./vary) *(T-1)/(T-K))'
   
        if (weight == 0) || (weight ==1)
            #compute GMM standard errors
            for indx = 1:N
                err = errv[:,indx]
                inner = (rhv.*(err*ones(1,K)))'*(rhv.*(err*ones(1,K)))/T
         
                for jindx = 1:lags
                    inneradd = (rhv[1:T-jindx,:].*(err[1:T-jindx]*ones(1,K)))'*(rhv[1+jindx:T,:].*(err[1+jindx:T]*ones(1,K)))/T
                    inner = inner + (1-weight*jindx/(lags+1))*(inneradd+inneradd')
                end
        
                 varb = 1/T * Exxprim*inner*Exxprim
                #F test for all coeffs (except constant) zero chi2 test
                if rhv[:,1] == ones(size(rhv,1),1)
                    chi2val = (bv[2:end,indx])'*inv(varb[2:end,2:end])*bv[2:end,indx]
                    dof = size(bv[2:end,1],1)
                    pval = 1 - cdf(Chisq(dof),chi2val)
                    F[indx,1:3] = [chi2val dof pval]
                else
                    chi2val = bv[:,indx]'*inv(varb)*bv[:,indx]
                    dof = size(bv[:,1],1)
                    pval = 1 - cdf(Chisq(dof),chi2val)
                    F[indx,1:3] = [chi2val dof pval]
                end
                
                if indx ==1
                    v = varb
                else
                    v = [v; varb]
                end
                
                seb = diag(varb)
                seb = sign(seb).*(abs(seb).^0.5)
                sebv[:,indx] = seb
            end 
    
        elseif weight == 2
                                        
                for indx = 1:N
                    err=errv[:,indx]
                    inner = (rhv)'*(rhv)/T
        
                    for jindx = (1:lags)
                        inneradd = (rhv[1:T-jindx,:])'*(rhv[1+jindx:T,:])/T
                        inner = inner + (1-jindx/(lags))*(inneradd + inneradd')   #lags = horizon of regression, t+h on t
                    end
                sigv = std(err)^2       
                varb = 1/T*Exxprim*sigv*inner*Exxprim
                if rhv[:,1] == ones(size(rhv,1),1)
                    chi2val = (bv[2:end,indx])'*inv(varb[2:end,2:end])*bv[2:end,indx]
                    dof = size(bv[2:end,1],1)
                    pval = 1 - cdf(Chisq(dof),chi2val)
                    F[indx,1:3] = [chi2val dof pval]
                else
                    chi2val = bv[:,indx]'*inv(varb)*bv[:,indx]
                    dof = size(bv[:,1],1)
                    pval = 1 - cdf(Chisq(dof),chi2val)
                    F[indx,1:3] = [chi2val dof pval]
                end
                   if indx ==1
                    v = varb
                else
                    v = [v; varb]
                end
                
                seb = diag(varb)
                seb = sign(seb).*(abs(seb).^0.5)
                sebv[:,indx] = seb
            end
        elseif (weight > 0) && (weight < 1)  # rho^j weights
            for indx = 1:N
                    err=errv[:,indx]
                    inner = (rhv)'*(rhv)/T
        
                    for jindx = (1:lags)
                        inneradd = (rhv[1:T-jindx,:])'*(rhv[1+jindx:T,:])/T
                        inner = inner + (weight^jindx)*(1-weight^(2*(lags-jindx)))/(1-weight^(2*lags))*(inneradd')  
                        
                    end
                sigv = std(err)^2       
                varb = 1/T*Exxprim*sigv*inner*Exxprim    
                
                if rhv[:,1] == ones(size(rhv,1),1)
                    chi2val = (bv[2:end,indx])'*inv(varb[2:end,2:end])*bv[2:end,indx]
                    dof = size(bv[2:end,1],1)
                    pval = 1 - cdf(Chisq(dof),chi2val)
                    F[indx,1:3] = [chi2val dof pval]
                else
                    chi2val = bv[:,indx]'*inv(varb)*bv[:,indx]
                    dof = size(bv[:,1],1)
                    pval = 1 - cdf(Chisq(dof),chi2val)
                    F[indx,1:3] = [chi2val dof pval]
                end
                
               if indx ==1
                    v = varb
                else
                    v = [v; varb]
                end
                
                seb = diag(varb)
                seb = sign(seb).*(abs(seb).^0.5)
                sebv[:,indx] = seb
         end
        end
    end
    return bv,sebv,R2v,R2vadj,v,F
end
