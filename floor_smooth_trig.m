function [y, dy] = floor_smooth_trig(x, delta)
% floor_smooth_trig  “Smooth” approximation to floor(x) using a trigonometric sawtooth.
%   [y, dy] = floor_smooth_trig(x, delta)
%
%   Inputs:
%     x     = array of input values (any real, unbounded).
%     delta = small positive constant (e.g. 0.01).
%
%   Outputs:
%     y  = x minus the smooth “sawtooth” term, i.e.  y ≈ floor(x).
%     dy = ∂y/∂x, the analytic derivative of this smooth approximation.
%
%   The underlying idea is:
%     sawtooth(x) ≈ swt(x) given by a trig‐approximation, so
%     floor(x) ≈ x − swt(x).
%
%   The trig‐sawtooth is built from two pieces—
%   a “triangle‐like” term trg(·) and a “square‐like” term sqr(·):
%
%     trg(u) = 1 − (2/π) * acos((1 − δ) * sin(2πu))
%     sqr(u) = (2/π) * atan( sin(2πu) / δ )
%
%   Then the smooth sawtooth is
%     swt(x) = (1 + trg((2x − 1)/4) .* sqr(x/2)) / 2,
%   and finally
%     y = x − swt(x).
%
%   We also compute dy = d/dx [ x − swt(x) ] = 1 − swt′(x).
%
%   Usage Example:
%     x = linspace(-2,5,200);
%     [y, dy] = floor_smooth_trig(x, 0.01);
%     plot(x, y, 'b', x, floor(x), 'r--');
%     legend('smooth approx','true floor');
%
%---- 1) Compute the two auxiliary “trg” and “sqr” functions ----
% trg(u) = 1 − (2/π) * acos((1 − δ) * sin(2πu))
% sqr(u) = (2/π) * atan( sin(2πu) / δ )
%
% We will need them at two different arguments:
%
%   u1 = (2*x − 1)/4   for the trg‐piece,
%   u2 = x/2           for the sqr‐piece.
%
u1 = (2*x - 1) / 4;
u2 = x / 2;

% trg(u1)
s1 = sin(2*pi*u1);                             % sin(2πu1)
inside_acos = (1 - delta) .* s1;               % (1−δ)·sin(2πu1)
% clamp inside_acos to [−1, 1] just to avoid tiny rounding errors
inside_acos = max(min(inside_acos, 1), -1);
trg_u1 = 1 - (2/pi) .* acos(inside_acos);

% sqr(u2)
s2 = sin(2*pi*u2);                             % sin(2πu2)
sqr_u2 = (2/pi) .* atan( s2 ./ delta );         % (2/π)·atan(sin(2πu2)/δ)

%---- 2) Build the smooth sawtooth and its derivative ----
% swt(x) = (1 + trg_u1 .* sqr_u2) / 2
%
swt = 0.5 .* (1 + trg_u1 .* sqr_u2);
%---- 3) floor‐approximation and its derivative ----
y  = x - swt;            % “floor smooth” ≈ x − sawtooth(x)
%% dydx
c1 = cos(2*pi*u1);
denom1 = sqrt( max(1 - (1 - delta).^2 .* (s1.^2), eps) );
trg_pr1 = (4*(1 - delta).*c1) ./ denom1;

c2 = cos(2*pi*u2);
denom2 = delta .* (1 + (s2./delta).^2);        % <--- no π here
sqr_pr2 = (4 .* c2) ./ denom2;                 % now matches 4/(δ(1+(s/δ)^2))

% du1/dx = du2/dx = 0.5
du1dx = 0.5;
du2dx = 0.5;

swt_prime = 0.5 .* ( trg_pr1 .* du1dx .* sqr_u2 + trg_u1   .* sqr_pr2 .* du2dx );

dy = 1 - swt_prime;
end