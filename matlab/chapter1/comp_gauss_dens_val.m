function [ p ] = comp_gauss_dens_val( m, S, x )
%COMP_GAUSS_DENS_VAL Summary of this function goes here
%   Detailed explanation goes here
    l = size(m, 1);
    p = 1/sqrt(2*pi) ^ l * 1/sqrt(det(S)) * exp(-(x-m)' / S * (x-m)/2);

end

