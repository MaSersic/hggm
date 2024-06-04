function [coefs] = lasso_reg(PHI, row, lag, lambda, normal, inverse_normal, poisson, gamma, binomial)

    row = row';
    addpath('./penalized');
    addpath('./penalized/models');
    addpath('./penalized/internals');
    addpath('./penalized/penalties');

    if normal

        model= glm_gaussian(row, PHI, 'nointercept');
        options = statset(statset('glmfit'));
        options.MaxIter=100000;

        b = glmfit(PHI, row,'normal','Options',options,'link','identity','constant','off');

    end

    if inverse_normal

        model= glm_inv_gaussian(row, PHI, 'nointercept');
        options = statset(statset('glmfit'));
        options.MaxIter=100000;

        b = glmfit(PHI, row,'inverse gaussian','Options',options,'link',-2,'constant','off');

    end

    if gamma

        model= glm_gamma(row, PHI, 'nointercept');
        options = statset(statset('glmfit'));
        options.MaxIter=100000;

        b = glmfit(PHI, row,'gamma','Options',options,'link','reciprocal','constant','off');

    end

    if poisson

        model= glm_poisson(row, PHI, 'nointercept');
        options = statset(statset('glmfit'));
        options.MaxIter=100000;

        b = glmfit(PHI, row,'poisson','Options',options,'link','log','constant','off');

    end

    if binomial

        model= glm_logistic(row, PHI, 'nointercept');
        options = statset(statset('glmfit'));
        options.MaxIter=100000;

        b = glmfit(PHI, row,'binomial','Options',options,'link','logit','constant','off');

    end

    fit_CV_glmfit= cv_penalized(model,@p_adaptive,'lambdamax',lambda,'lambdaminratio',0.01,'gamma',0.8,'adaptivewt',{b});

    %b_opt is the optimal coefficients regarding AdLasso with glmfit
    %intial weights

    b_opt=fit_CV_glmfit.bestbeta;

    if isempty(b_opt)
        disp('Isempty');
        b_opt=zeros(p,lag);
    end

    coefs = vec2mat(b_opt, lag);
end