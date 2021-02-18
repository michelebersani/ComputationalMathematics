function [f, g] = quadratic(xx)

    d = length(xx);
    sum1 = 0;
    for ii = 1:d
        xi = xx(ii);
        sum1 = sum1 + xi^6;
    end
    f = sum1;
    g = 6*xx.^5;
end

