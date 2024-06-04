function l = l_from_mu(model,mu)
% L_FROM_MU is an internal function used in logL and scoring.
%
% Usage:
%   l = l_from_mu(m,mu)
%
% Inputs:
%   m  : a glm_gamma model object
%   mu : a vector of expected values
%
% Outputs:
%   l  : the log-likelihood, sum(y.*(1/mu)-mu)

ok = mu~=0;
if(nnz(ok)==0)
    l=0;
else
    l = sum(model.glm_base.y(ok).*(1./mu(ok))-mu(ok));
end