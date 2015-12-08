function [ data ] = gen_gaussian( m, S, N )
%PLOT_GAUSSIAN Summary of this function goes here
%   Detailed explanation goes here
    randn('seed', 0)
    data = mvnrnd(m, S, N);
   
end

