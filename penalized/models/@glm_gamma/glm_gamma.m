function m = glm_gamma(y,X,varargin)
% gamma creates a gamma model.
%
% Usage:
%   p = glm_gamma(y,X,...)
%
% Inputs:
%   y  : a vector of observations
%   X  : a matrix of covariates
%   ... : options. These are:
%            'nointercept' - stops an intercept being added.
%            'center' or 'centre' - centres the columns of X when the
%                intercept is added.
%
% Outputs:
%   p  : a gamma model object
%
% Notes:
% In the gamma model, E(y)=1/(X*beta). 

m = class(struct,'glm_gamma',glm_base(y,X,varargin{:}));