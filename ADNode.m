classdef ADNode < handle
    %% Node in the function evalution graph

    properties
        value % function value at this node
        grad % gradient accumulator
        func % callback function to update gradient of the parent nodes
        root % input node that holds the tape
        tape % sequence of evaluation steps
    end

    methods (Static)
        function b = isreal(x)
            b = isreal(x.value);
        end
        function o = fftconvolve(x,y)
            ax = numel(x);
            ay = numel(y);
            len = ax+ay-1;
            x(length(x)+1:len) = 0;
            y(length(y)+1:len) = 0;
            x = fft(x);
            y = fft(y);
            o = x.*y;
            o = real(ifft(o));
        end
        % Helper function to find the root of the computation graph
        function root = find_ad_root(args)
            root = [];
            for i = 1:numel(args)
                if isa(args{i}, 'ADNode')
                    root = args{i}.root;
                    return;
                end
            end
        end
        function paddedArray = padArray(inputArray, padSize, dim)
            %% padArray: Pads an array with zeros on a specified dimension, or defaults to the first dimension.
            %
            %   paddedArray = padArray(inputArray, padSize)
            %   paddedArray = padArray(inputArray, padSize, dim)
            %
            %   Inputs:
            %     inputArray - The input array (can be 1D, 2D, 3D, or higher).
            %     padSize    - The number of zeros to pad on each side of the relevant dimension(s).
            %                  If padSize is a scalar, it applies to the specified 'dim'
            %                  or to the first dimension if 'dim' is not provided.
            %                  If padSize is a vector and 'dim' is NOT provided, its
            %                  length must match the number of dimensions of inputArray,
            %                  specifying padding for each.
            %                  If padSize is a vector and 'dim' IS provided, only its
            %                  first element will be used as the padding amount for 'dim'.
            %     dim        - (Optional) The specific dimension along which to pad.
            %                  If provided, only this dimension will be padded.
            %                  If not provided, padding will be applied only to the
            %                  first dimension of the array.
            %
            %   Output:
            %     paddedArray - The input array padded with zeros.
            %
            %   Examples:
            %     % Default behavior: Pads only the first dimension
            %     vec_row = [1 2 3 4 5];
            %     paddedVec_row = padArray(vec_row, 1); % Result will be 3x5 (pads dim 1)
            %     % [0 0 0 0 0]
            %     % [1 2 3 4 5]
            %     % [0 0 0 0 0]
            %
            %     % Default behavior: Pads only the first dimension
            %     vec_col = [1; 2; 3];
            %     paddedVec_col = padArray(vec_col, 1); % Result will be 5x1 (pads dim 1)
            %     % [0]
            %     % [0]
            %     % [1]
            %     % [2]
            %     % [3]
            %     % [0]
            %
            %     % Default behavior: 2D matrix pads only the first dimension (rows)
            %     mat = [1 2; 3 4];
            %     paddedMat = padArray(mat, 1);
            %     % Result:
            %     % [0 0]
            %     % [1 2]
            %     % [3 4]
            %     % [0 0]
            %
            %     % 3D array: pads only along the 3rd dimension (explicit dim)
            %     arr3D = rand(2,3,2);
            %     paddedArr3D_dim3 = padArray(arr3D, 1, 3);
            %     % The output will be 2x3x4 (original 2x3x2, padded by 1 on dim 3)
            %
            %     % 2D matrix: pads only along the 1st dimension (rows) (explicit dim)
            %     mat_pad_dim1 = padArray([1 2; 3 4], 1, 1);
            %     % Result:
            %     % [0 0]
            %     % [1 2]
            %     % [3 4]
            %     % [0 0]
            %
            %     % 2D matrix: pads only along the 2nd dimension (columns) (explicit dim)
            %     mat_pad_dim2 = padArray([1 2; 3 4], 1, 2);
            %     % Result:
            %     % [0 1 2 0]
            %     % [0 3 4 0]
            %
            %     % Different padding for different dimensions (explicit vector padSize, but still only pads dim 1 by default)
            %     % Note: If dim is NOT provided, a vector padSize will still only apply to the first dimension.
            %     % To pad multiple specific dimensions, you must call padArray multiple times or create a custom
            %     % padSizesPerDim array and pass it with dimProvided = true logic.
            %     mat2 = [1 2; 3 4];
            %     paddedMat2_default = padArray(mat2, [1 2]); % This will pad only dim 1 by 1.
            %     % Result:
            %     % [0 0]
            %     % [1 2]
            %     % [3 4]
            %     % [0 0]
            inputSize = size(inputArray);
            numDims = ndims(inputArray);
            % Determine if 'dim' argument was provided
            dimProvided = false;
            if nargin == 3
                dimProvided = true;
                targetDim = dim; % Store the specified dimension
            end
            % Initialize padSizesPerDim array to all zeros
            padSizesPerDim = zeros(1, numDims);
            if dimProvided
                % Validate the specified dimension
                if ~isscalar(targetDim) || ~isnumeric(targetDim) || targetDim < 1 || targetDim > numDims || mod(targetDim, 1) ~= 0
                    error('padArray:InvalidDimension', 'Dimension (dim) must be a positive integer within the range of array dimensions.');
                end
                % If dim is specified, padSize should ideally be a scalar for that dim.
                % If padSize is a vector, take its first element as the effective padding.
                if isscalar(padSize)
                    effectivePadAmount = padSize;
                else
                    warning('padArray:VectorPadSizeWithDim', 'When a specific dimension is provided, padSize should be a scalar. Using the first element of padSize vector.');
                    effectivePadAmount = padSize(1);
                end
                % Apply padding only for the specified dimension
                padSizesPerDim(targetDim) = effectivePadAmount;

            else % 'dim' argument was NOT provided, default to padding only the first dimension
                % If padSize is a scalar, apply it to the first dimension.
                % If padSize is a vector, take its first element for the first dimension.
                if isscalar(padSize)
                    effectivePadAmount = padSize;
                else
                    warning('padArray:VectorPadSizeNoDim', 'When no specific dimension is provided, padSize should be a scalar. Using the first element of padSize vector for the first dimension.');
                    effectivePadAmount = padSize(1);
                end
                % Apply padding only to the first dimension
                padSizesPerDim(1) = effectivePadAmount;
            end

            % Calculate the new size for the padded array
            newSize = zeros(1, numDims);
            for i = 1:numDims
                newSize(i) = inputSize(i) + 2 * padSizesPerDim(i);
            end
            % Create a new array of zeros with the calculated new size, preserving class
            paddedArray = zeros(newSize, class(inputArray));
            % Determine the indices to place the original array
            startIndices = padSizesPerDim + 1;
            endIndices = padSizesPerDim + inputSize;
            % Construct the cell array for indexing
            idx = cell(1, numDims);
            for i = 1:numDims
                idx{i} = startIndices(i):endIndices(i);
            end
            % Place the original array into the center of the new zero-padded array
            paddedArray(idx{:}) = inputArray;
        end
    end
    methods
        function y = ADNode(x, root, func)
            %% create new node
            if nargin > 1
                y.func = func;
                y.root = root;
                root.tape{end+1} = y;
            else
                y.root = y;
                y.tape = {};
                %                 y.grad = eye(numel(x));
            end
            y.value = x;
        end

        function dy = backprop(x, dy)
            %% backpropagate the gradient by evaluating the tape backwards
            if nargin > 1
                x.grad = dy;
            else
                x.grad = 1;
            end
            for k = length(x.root.tape):-1:1
                x.root.tape{k}.func(x.root.tape{k});
                x.root.tape(k) = [];
            end
            dy = x.root.grad;
            if size(dy) ~= size(x.root.value)
                if size(dy) == ones(1, ndims(dy))
                    dy = repmat(dy, size(x.root.value));
                elseif size(dy, 1) == 1 && size(x.root.value, 1) ~= 1
                    dy = repmat(dy, size(x.root.value, 1), 1);
                elseif size(dy, 2) == 1 && size(x.root.value, 2) ~= 1
                    dy = repmat(dy, 1, size(x.root.value, 2));
                end
            end
        end
        function y = reshape(x, varargin)
            y = ADNode(reshape(x.value, varargin{:}), x.root, @(y) x.add(reshape(y.grad, size(x.value))));
        end

        function y = tanh(x)
            y = ADNode(tanh(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, sech((x.value)) .^ 2)));
        end

        function y = sum(x, dim, flag)
            switch nargin
                case 3
                    y = ADNode(sum(x.value, dim, flag), x.root, @(y) x.add(y.grad .* ones(size(x.value))));
                case 2
                    y = ADNode(sum(x.value, dim), x.root, @(y) x.add(y.grad .* ones(size(x.value))));
                otherwise
                    y = ADNode(sum(x.value), x.root, @(y) x.add(y.grad .* ones(size(x.value))));
            end
        end

        function y = abs(x)
            y = ADNode(abs(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, sign(x.value))));
        end
        function y = cabs(x)
            y = ADNode(abs(x.value), x.root, @(y) x.add(bsxfun(@times, real(y.grad), bsxfun(@rdivide, x.value, abs(x.value)))));
        end

        function y = acos(x)
            y = ADNode(acos(x.value), x.root, @(y) x.add(bsxfun(@rdivide, -y.grad, sqrt(1-x.value.^2))));
        end

        function y = asin(x)
            y = ADNode(asin(x.value), x.root, @(y) x.add(bsxfun(@rdivide, y.grad, sqrt(1-x.value.^2))));
        end

        function y = atan(x)
            y = ADNode(atan(x.value), x.root, @(y) x.add(bsxfun(@rdivide, y.grad, (1+x.value.^2))));
        end

        function y = cos(x)
            y = ADNode(cos(x.value), x.root, @(y) x.add(bsxfun(@times, -y.grad, sin(x.value))));
        end
        function backprop_soft_well(x, lower, upper, scale, y, lg, d_low, d_high)
            if lg == true
                deriv = 2 * d_low ./ (scale^2 + d_low.^2) - 2*d_high ./ (scale^2 + d_high.^2);
            else
                deriv = -(2 * (lower + upper - 2 * x.value)) / scale^2;
            end
            x.add(y.grad .* deriv);
        end
        function y = soft_well(x, lower, upper, scale, lg)
            xv = x.value;
            mid_dist = (upper - lower) / 2;
            d_low = xv - lower;
            d_high = upper - xv;
            if lg == true
                bf_i = log(1 + (d_low / scale).^2) + log(1 + (d_high / scale).^2);
                C_i = 2 * log(1 + (mid_dist / scale).^2);
            else
                bf_i = (1 + (d_low / scale).^2) + (1 + (d_high / scale).^2);
                C_i = 2 * (1 + (mid_dist / scale).^2);
            end
            bf = bf_i - C_i;
            y = ADNode(bf, x.root, @(y) backprop_soft_well(x, lower, upper, scale, y, lg, d_low, d_high));
        end

        function backprop_sinc(x, y, i)
            val = cos(pi * x.value) ./ x.value - sin(pi * x.value) ./ ((x.value.^2) * pi);
            val(i) = 0;
            x.add(bsxfun(@times, y.grad, val));
        end
        function y = sinc(x)
            tmp = x.value;
            i = find(tmp == 0);
            out = sin(pi*tmp)./(pi*tmp);
            out(i) = 1;
            y = ADNode(out, x.root, @(y) backprop_sinc(x, y, i));
        end
        function backprop_atan2(x1, x2, y)
            x1.add(y.grad .* x2.value ./ (x1.value .* x1.value + x2.value .* x2.value));
            x2.add(y.grad .* -x1.value ./ (x1.value .* x1.value + x2.value .* x2.value));
        end
        function y = atan2(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(atan2(x1.value, x2.value), x1.root, @(y) backprop_atan2(x1, x2, y));
                else
                    y = ADNode(atan2(x1.value, x2), x1.root, @(y) x1.add(y.grad .* x2 ./ (x1.value .* x1.value + x2 .* x2)));
                end
            else
                y = ADNode(atan2(x1, x2.value), x2.root, @(y) x2.add(y.grad .* -x1 ./ (x1 .* x1 + x2.value .* x2.value)));
            end
        end
        function y = magPhaseCplx(mag, phi)
            % f(mag,phi) = mag .* exp(1j*phi)
            %
            % This overload handles three cases:
            % 1) both mag and phi are ADNode
            % 2) only mag is ADNode (phi is constant)
            % 3) only phi is ADNode (mag is constant)
            %
            % In all cases, the “root” pointer is taken from whichever input is an ADNode.
            if isa(mag, 'ADNode') && isa(phi, 'ADNode')
                % both are ADNode
                E = exp(1j * phi.value);
                val = mag.value .* E;
                y = ADNode(val, mag.root, @(y) backprop_magPhaseCplx(mag, phi, y, E));
            elseif isa(mag, 'ADNode')
                % mag is ADNode, phi is a plain scalar (or array)
                val = mag.value .* exp(1j * phi);
                y = ADNode(val, mag.root, @(y) mag.add(y.grad .* exp(1j * phi)));
            elseif isa(phi, 'ADNode')
                % phi is ADNode, mag is a plain scalar (or array)
                val = mag .* exp(1j * phi.value);
                y = ADNode(val, phi.root, @(y) phi.add(y.grad .* (mag .* 1j .* exp(1j * phi.value))));
            else
                % neither is an ADNode → constant output; no gradients attached
                y = mag .* exp(1j * phi);
            end
        end
        function backprop_magPhaseCplx(mag, phi, y, E)
            % “Push‐back” for y = mag .* exp(1j * phi)
            %   ∂y/∂mag =          exp(jφ)
            %   ∂y/∂φ   = mag .* j .* exp(jφ)
            %
            % y.grad is ∂L/∂y coming from upstream; we multiply by the local Jacobians
            %
            % Note: mag.value and phi.value are the stored forward values, so
            %       exp(1j*phi.value) was computed in the forward pass.
            % ∂L/∂mag += (∂L/∂y) .* (∂y/∂mag)  = y.grad .* E
            mag.add(y.grad .* E);
            % ∂L/∂φ   += (∂L/∂y) .* (∂y/∂φ)    = y.grad .* (mag.value .* j .* E)
            phi.add(y.grad .* (mag.value .* 1j .* E));
        end
        %===============================================
        % 1) Hard floor / round / ceil (zero‐gradient)
        %===============================================

        function y = floor(x)
            xv = x.value;
            out = floor(xv);
            y   = ADNode(out, x.root, @(y) backprop_floor(x, y));
        end
        function backprop_floor(x, y)
            % d floor(x)/dx = 0 almost everywhere
            x.add( zeros(size(x.value)) );
        end

        function y = round(x)
            xv = x.value;
            out = round(xv);
            y   = ADNode(out, x.root, @(y) backprop_round(x, y));
        end
        function backprop_round(x, y)
            % d round(x)/dx = 0 almost everywhere
            x.add( zeros(size(x.value)) );
        end

        function y = ceil(x)
            xv = x.value;
            out = ceil(xv);
            y   = ADNode(out, x.root, @(y) backprop_ceil(x, y));
        end
        function backprop_ceil(x, y)
            % d ceil(x)/dx = 0 almost everywhere
            x.add( zeros(size(x.value)) );
        end


        %===============================================
        % 2) Smooth floor / round / ceil via trig‐saw
        %===============================================

        function y = floor_smooth(x, delta)
            % wrapper for [yf, dyf] = floor_smooth_trig(x, delta)
            xv = x.value;
            [yf, dyf] = floor_smooth_trig(xv, delta);
            y = ADNode(yf, x.root, @(y) backprop_floor_smooth(x, y, dyf));
        end
        function backprop_floor_smooth(x, y, dyf)
            % x.add( y.grad .* dyf )
            x.add( y.grad .* dyf );
        end

        function y = round_smooth(x, delta)
            % round_smooth(x) = floor_smooth(x + 0.5)
            xv = x.value;
            [yf, dyf] = floor_smooth_trig(xv + 0.5, delta);
            y = ADNode(yf, x.root, @(y) backprop_round_smooth(x, y, dyf));
        end
        function backprop_round_smooth(x, y, dyf)
            % d/dx floor_smooth(x+0.5) = dyf * 1
            x.add( y.grad .* dyf );
        end

        function y = ceil_smooth(x, delta)
            % ceil_smooth(x) = -floor_smooth(-x)
            xv = x.value;
            [yf, dyf] = floor_smooth_trig(-xv, delta);
            out = -yf;
            y = ADNode(out, x.root, @(y) backprop_ceil_smooth(x, y, dyf));
        end
        function backprop_ceil_smooth(x, y, dyf)
            % d/dx [ -floor_smooth(-x) ] = -[ dyf * (-1) ] = dyf
            x.add( y.grad .* dyf );
        end
        %===============================================
        % 1) Hard anti‐wrap
        %===============================================
        function y = anti_wrap(x)
            % Forward: y = abs( x - round(x/(2π))⋅2π )
            xv       = x.value;
            k        = round(xv/(2*pi));
            residual = xv - k*(2*pi);
            out      = abs(residual);
            % Build ADNode, attach backprop
            y = ADNode(out, x.root, @(y) backprop_anti_wrap(x, y));
        end
        function backprop_anti_wrap(x, y)
            % Compute d/dx [ abs(x - round(x/(2π))⋅2π) ] = sign(residual)
            xv       = x.value;
            k        = round(xv/(2*pi));
            residual = xv - k*(2*pi);
            sgn      = sign(residual);   % MATLAB: sign(0)==0

            % Chain rule
            x.add( y.grad .* sgn );
        end
        %===============================================
        % 2) Smooth anti‐wrap via floor_smooth_trig
        %===============================================
        function y = anti_wrap_smooth(x, delta)
            xv       = x.value;
            two_pi   = 2*pi;
            z        = xv / two_pi;
            % 1) smooth‐round: k_smooth ≈ round(z)
            [k_smooth, dk_smooth] = floor_smooth_trig(z + 0.5, delta);
            % 2) residual = x - 2π*k_smooth
            residual = xv - two_pi * k_smooth;
            % 3) output = abs(residual)
            out = abs(residual);
            % Wrap into ADNode
            y = ADNode(out, x.root, @(y) backprop_anti_wrap_smooth(x, y, dk_smooth, residual));
        end
        function backprop_anti_wrap_smooth(x, y, dk_smooth, residual)
            % d(residual)/dx = 1 - dk_smooth
            dres_dx = 1 - dk_smooth;
            % d|r|/dr = sign(residual)
            sgn = sign(residual);
            % chain
            x.add( y.grad .* (sgn .* dres_dx) );
        end
        %         function y = ff2(x, y)
        %             x.add(conj(y.grad));
        %         end
        %         function y = conj(x)
        %             y = ADNode(conj(x.value), x.root, @(y) ff2(x, y));
        %         end
        function halfSpectrumReflect(x, y, NFFT)
            tmp = y.grad;
            if mod(NFFT,2) == 0
                halfLen = (NFFT/2)+1;
                tmp(halfLen+1:NFFT, :) = 0;
                tmp2 = fft(tmp);
                tmp2 = flipud(circshift(tmp2, -1));
            else
                halfLen = (NFFT+1)/2;
                tmp(halfLen+1:NFFT, :) = 0;
                tmp2 = fft(tmp);
                tmp2 = circshift(flipud(tmp2), 1);
            end
            tmp2 = real(tmp2);
            x.add(tmp2);
        end
        function y = rfft(x, NFFT)
            tmp = fft(x.value);
            if mod(NFFT, 2)==0
                halfLen = (NFFT/2)+1;
            else
                halfLen = (NFFT+1)/2;
            end
            y = ADNode(tmp(1 : halfLen, :), x.root, @(y) halfSpectrumReflect(x, y, NFFT));
        end
        function backprop_svfFilter(x, y, input, b0, d1, d2, c1, c2, y2, z1_A, z2_A, coeff_b0, coeff_d1, coeff_d2, coeff_c1, coeff_c2, initialValueZ1, initialValueZ2, ltv, parallelFilter, reverseFilter, reqGrad)
            if reverseFilter
                tmp2_grad = flipud(y.grad);
            else
                tmp2_grad = y.grad;
            end
            [~, nSections] = size(b0);
            [m, nSigs] = size(input);
            b0_grad = zeros(m, nSections);d1_grad = zeros(m, nSections);d2_grad = zeros(m, nSections);c1_grad = zeros(m, nSections);c2_grad = zeros(m, nSections);
            z2_A_grad = zeros(nSections, nSigs);z1_A_grad = zeros(nSections, nSigs);y2_grad = zeros(nSections, nSigs);section_grad = zeros(nSections, nSigs);input_grad = zeros(m, nSigs);
            for a = m : -1 : 2
                gd = tmp2_grad(a, :);
                for idx = nSections : -1 : 1
                    d2_grad(a, idx) = d2_grad(a, idx) + sum(gd .* z2_A(a, :, idx));
                    d1_grad(a, idx) = d1_grad(a, idx) + sum(gd .* z1_A(a, :, idx));
                    b0_grad(a, idx) = b0_grad(a, idx) + sum(gd .* y2(a, :, idx));
                    section_grad(idx, :) = y2_grad(idx, :) + gd .* b0(a, idx);
                    z2_A_grad(idx, :) = z2_A_grad(idx, :) + gd .* d2(a, idx) - section_grad(idx, :);
                    z1_A_grad(idx, :) = z1_A_grad(idx, :) + gd .* d1(a, idx) - section_grad(idx, :);
                    if parallelFilter == 0
                        gd = section_grad(idx, :);
                    end
                    c1_grad(a - 1, idx) = c1_grad(a - 1, idx) + sum(z1_A_grad(idx, :) .* y2(a - 1, :, idx));
                    y2_grad(idx, :) = z1_A_grad(idx, :) .* c1(a - 1, idx);
                    c2_grad(a - 1, idx) = c2_grad(a - 1, idx) + sum(z2_A_grad(idx, :) .* z1_A(a - 1, :, idx));
                    z1_A_grad(idx, :) = z1_A_grad(idx, :) + z2_A_grad(idx, :) .* c2(a - 1, idx);
                end
                if parallelFilter == 1
                    input_grad(a, :) = input_grad(a, :) + sum(section_grad, 1);
                else
                    input_grad(a, :) = input_grad(a, :) + section_grad(1, :);
                end
            end
            gd = tmp2_grad(1, :);
            for idx = nSections : -1 : 1
                d2_grad(1, idx) = d2_grad(1, idx) + sum(gd .* z2_A(1, :, idx));
                d1_grad(1, idx) = d1_grad(1, idx) + sum(gd .* z1_A(1, :, idx));
                b0_grad(1, idx) = b0_grad(1, idx) + sum(gd .* y2(1, :, idx));
                section_grad(idx, :) = y2_grad(idx, :) + gd * b0(1, idx);
                z2_A_grad(idx, :) = z2_A_grad(idx, :) + gd .* d2(1, idx) - section_grad(idx, :);
                z1_A_grad(idx, :) = z1_A_grad(idx, :) + gd .* d1(1, idx) - section_grad(idx, :);
                if parallelFilter == 0
                    gd = section_grad(idx, :);
                end
            end
            if parallelFilter == 1
                input_grad(1, :) = input_grad(1, :) + sum(section_grad, 1);
            else
                input_grad(1, :) = input_grad(1, :) + section_grad(1, :);
            end
            if reverseFilter
                b0_grad = flipud(b0_grad);d1_grad = flipud(d1_grad);d2_grad = flipud(d2_grad);c1_grad = flipud(c1_grad);c2_grad = flipud(c2_grad);input_grad = flipud(input_grad);
            end
            if ~ltv
                b0_grad = sum(b0_grad, 1);d1_grad = sum(d1_grad, 1);d2_grad = sum(d2_grad, 1);c1_grad = sum(c1_grad, 1);c2_grad = sum(c2_grad, 1);
            end
            z2_A_grad = z2_A_grad.';z1_A_grad = z1_A_grad.';
            if reqGrad(1)
                x.add(input_grad);
            end
            if reqGrad(2)
                coeff_b0.add(b0_grad);
            end
            if reqGrad(3)
                coeff_d1.add(d1_grad);
            end
            if reqGrad(4)
                coeff_d2.add(d2_grad);
            end
            if reqGrad(5)
                coeff_c1.add(c1_grad);
            end
            if reqGrad(6)
                coeff_c2.add(c2_grad);
            end
            if reqGrad(7)
                initialValueZ1.add(z1_A_grad);
            end
            if reqGrad(8)
                initialValueZ2.add(z2_A_grad);
            end
        end
        function y = svfFilter(x, coeff_b0, coeff_d1, coeff_d2, coeff_c1, coeff_c2, initialValueZ1, initialValueZ2, parallelFilter, reverseFilter)
            reqGrad = [isa(x,'ADNode'),isa(coeff_b0,'ADNode'),isa(coeff_d1,'ADNode'),isa(coeff_d2,'ADNode'),...
                isa(coeff_c1,'ADNode'),isa(coeff_c2,'ADNode'),isa(initialValueZ1,'ADNode'),isa(initialValueZ2,'ADNode')];
            if reqGrad(1)
                input = x.value;
            else
                input = x;
            end
            if reqGrad(2)
                b0 = coeff_b0.value;
            else
                b0 = coeff_b0;
            end
            [nCoeff, nSections] = size(b0);
            [m, nSigs] = size(input);
            if nCoeff == m
                ltv = 1;
            elseif nCoeff == 1
                ltv = 0;
            else
                error('Coefficient length must either be equal to signal length or equal to 1');
                return;
            end
            if reqGrad(3)
                d1 = coeff_d1.value;
            else
                d1 = coeff_d1;
            end
            if reqGrad(4)
                d2 = coeff_d2.value;
            else
                d2 = coeff_d2;
            end
            if reqGrad(5)
                c1 = coeff_c1.value;
            else
                c1 = coeff_c1;
            end
            if reqGrad(6)
                c2 = coeff_c2.value;
            else
                c2 = coeff_c2;
            end
            if any(~([nCoeff, nSections] == size(d1))) || any(~([nCoeff, nSections] == size(d2))) || any(~([nCoeff, nSections] == size(c1))) || any(~([nCoeff, nSections] == size(c2)))
                error('All IIR coefficients must has same size')
            end
            if ~ltv
                b0 = repmat(b0, [m, 1]);
                d1 = repmat(d1, [m, 1]);
                d2 = repmat(d2, [m, 1]);
                c1 = repmat(c1, [m, 1]);
                c2 = repmat(c2, [m, 1]);
            end
            if reqGrad(7)
                z1 = initialValueZ1.value;
            else
                z1 = initialValueZ1;
            end
            if reqGrad(8)
                z2 = initialValueZ2.value;
            else
                z2 = initialValueZ2;
            end
            z1_A = zeros(m, nSigs, nSections);z2_A = zeros(m, nSigs, nSections);
            if numel(z1) == 1 || numel(z1) == (m * nSigs * nSections)
                z1_A(1, :, :) = z1;
            else
                for idx = 1 : nSections
                    z1_A(1, :, idx) = z1(:, idx);
                end
            end
            if numel(z2) == 1 || numel(z2) == (m * nSigs * nSections)
                z2_A(1, :, :) = z2;
            else
                for idx = 1 : nSections
                    z2_A(1, :, idx) = z2(:, idx);
                end
            end
            y2 = zeros(m, nSigs, nSections);
            if reverseFilter
                input = flipud(input);b0 = flipud(b0);d1 = flipud(d1);d2 = flipud(d2);c1 = flipud(c1);c2 = flipud(c2);
            end
            % Forward filtering
            tmp2 = zeros(m, nSigs);
            if parallelFilter == 1
                out = zeros(nSections, nSigs);
                for a = 1 : m - 1
                    in = input(a, :);
                    for idx = 1 : nSections
                        y2(a, :, idx) = in - z1_A(a, :, idx) - z2_A(a, :, idx);
                        out(idx, :) = b0(a, idx) .* y2(a, :, idx) + d1(a, idx) .* z1_A(a, :, idx) + d2(a, idx) .* z2_A(a, :, idx);
                        z2_A(a + 1, :, idx) = z2_A(a, :, idx) + c2(a, idx) .* z1_A(a, :, idx);
                        z1_A(a + 1, :, idx) = z1_A(a, :, idx) + c1(a, idx) .* y2(a, :, idx);
                    end
                    tmp2(a, :) = sum(out, 1);
                end
                in = input(m, :);
                for idx = 1 : nSections
                    y2(m, :, idx) = in - z1_A(m, :, idx) - z2_A(m, :, idx);
                    out(idx, :) = b0(m, idx) .* y2(m, :, idx) + d1(m, idx) .* z1_A(m, :, idx) + d2(m, idx) .* z2_A(m, :, idx);
                end
                tmp2(m, :) = sum(out, 1);
            else
                for a = 1 : m - 1
                    in = input(a, :);
                    for idx = 1 : nSections
                        y2(a, :, idx) = in - z1_A(a, :, idx) - z2_A(a, :, idx);
                        in = b0(a, idx) .* y2(a, :, idx) + d1(a, idx) .* z1_A(a, :, idx) + d2(a, idx) .* z2_A(a, :, idx);
                        z2_A(a + 1, :, idx) = z2_A(a, :, idx) + c2(a, idx) .* z1_A(a, :, idx);
                        z1_A(a + 1, :, idx) = z1_A(a, :, idx) + c1(a, idx) .* y2(a, :, idx);
                    end
                    tmp2(a, :) = in;
                end
                in = input(m, :);
                for idx = 1 : nSections
                    y2(m, :, idx) = in - z1_A(m, :, idx) - z2_A(m, :, idx);
                    in = b0(m, idx) .* y2(m, :, idx) + d1(m, idx) .* z1_A(m, :, idx) + d2(m, idx) .* z2_A(m, :, idx);
                end
                tmp2(m, :) = in;
            end
            if reverseFilter
                tmp2 = flipud(tmp2);
            end
            y = ADNode(tmp2, coeff_b0.root, @(y) backprop_svfFilter(x, y, input, b0, d1, d2, c1, c2, y2, z1_A, z2_A, coeff_b0, coeff_d1, coeff_d2, coeff_c1, coeff_c2, initialValueZ1, initialValueZ2, ltv, parallelFilter, reverseFilter, reqGrad));
        end
        function halfSpectrumFold2(xRe, xIm, y, NFFT)
            tmp = ifft(y.grad);
            if mod(NFFT, 2) == 0
                halfLen = (NFFT/2)+1;
                tmp = tmp(1 : halfLen, :, :);
                tmp(2 : end - 1, :, :) = tmp(2 : end - 1, :, :) * 2;
            else
                halfLen = (NFFT+1)/2;
                tmp = tmp(1 : halfLen, :, :);
                tmp(2 : end, :, :) = tmp(2 : end, :, :) * 2;
            end
            tt = -imag(tmp);
            tt(1, :, :) = 0;
            tt(end, :, :) = 0;
            xRe.add(real(tmp));
            xIm.add(tt);
        end
        function y = separate_irfft(xRe, xIm, NFFT)
            tmp1 = xRe.value;
            tmp2 = xIm.value;
            tmp2(1, :, :) = 0;
            tmp2(end, :, :) = 0;
            if mod(NFFT,2) == 0
                halfLen = (NFFT/2)+1;
                tmp1(halfLen+1:NFFT, :, :) = tmp1(halfLen-1:-1:2, :, :);
                tmp2(halfLen+1:NFFT, :, :) = -tmp2(halfLen-1:-1:2, :, :);
            else
                halfLen = (NFFT+1)/2;
                tmp1(halfLen+1:NFFT, :, :) = tmp1(halfLen:-1:2, :, :);
                tmp2(halfLen+1:NFFT, :, :) = -tmp2(halfLen:-1:2, :, :);
            end
            y = ADNode(ifft(tmp1 + tmp2 * 1j), xRe.root, @(y) halfSpectrumFold2(xRe, xIm, y, NFFT));
        end
        function y = fft(x)
            y = ADNode(fft(x.value), x.root, @(y) x.add(fft(y.grad)));
        end
        function y = ifft(x)
            y = ADNode(ifft(x.value), x.root, @(y) x.add(ifft(y.grad)));
        end
        function y = real(x)
            y = ADNode(real(x.value), x.root, @(y) x.add(real(y.grad)));
        end
        function y = imag(x)
            y = ADNode(imag(x.value), x.root, @(y) x.add(complex(0, y.grad)));
        end
        function y = conj(x)
            y = ADNode(conj(x.value), x.root, @(y) x.add(conj(y.grad)));
        end
        function y = revWndAcc(x, y, frameSize, hop)
            tmp = y.grad;
            rec = buffer(tmp, frameSize, frameSize - hop);
            cutted = rec(:, frameSize / hop : end);
            x.add(cutted);
        end
        function y = fold(x, frameSize, hop) % Sum sliding window
            y = ADNode(overlapAdd(x.value, hop), x.root, @(y) revWndAcc(x, y, frameSize, hop));
        end
        function y = revWndAcc3D(x, y, frameSize, hop)
            tmp = y.grad;
            % idealBufferOutLen = ceil((size(tmp, 1)-(frameSize - hop))/(frameSize-(frameSize - hop)));
            requiredBufferOutLen = size(x, 2);
            cutLen = ceil(size(tmp, 1) / hop) - requiredBufferOutLen;
            rec = zeros(frameSize, ceil(size(tmp, 1) / hop), size(tmp, 2));
            for idx = 1 : size(tmp, 2)
                rec(:, :, idx) = buffer(tmp(:, idx), frameSize, frameSize - hop);
            end
            % cutted = rec(:, frameSize / hop : end, :);
            cutted = rec(:, cutLen + 1 : end, :);
            x.add(cutted);
        end
        function y = fold3D(x, frameSize, hop) % Sum sliding window
            y = ADNode(overlapAdd3D(x.value, hop), x.root, @(y) revWndAcc3D(x, y, frameSize, hop));
        end
        function backprop_logFourier(b0, b1, b2, a1, a2, y, phi, lg10, bSumV, aSumV, numerator, denominator, term2V, term3V, term6V, b0V, b1V, b2V, a1V)
            numBands = length(b0.value);
            reducedSumGrad = lg10 * repmat(y.grad, numBands, 1);
            %% Logarithmic Fourier transform partial derivatives
            term11Grad = reducedSumGrad ./ max(numerator, eps);
            term20Grad = -reducedSumGrad ./ max(denominator, eps);
            b0Grad = term11Grad .* bSumV * 2;
            b1Grad = term11Grad .* bSumV * 2;
            b2Grad = term11Grad .* bSumV * 2;
            a1Grad = term20Grad .* aSumV * 2;
            a2Grad = term20Grad .* aSumV * 2;
            term5Grad = term11Grad .* phi;
            term1Grad = term5Grad .* phi;
            termGrad = term20Grad .* phi;
            term5b1V = -term5Grad .* b1V;
            b1.grad = b1Grad + -term5Grad .* term2V;
            b0Grad = b0Grad + term5b1V + 4 * -term5Grad .* b2V;
            b2Grad = b2Grad + term5b1V + -term5Grad .* term3V;
            b0.grad = b0Grad + term1Grad .* b2V;
            b2.grad = b2Grad + term1Grad .* b0V;
            a1.grad = a1Grad + -termGrad .* term6V;
            a2.grad = a2Grad + 4 * -termGrad + -termGrad .* a1V + termGrad .* phi;
        end
        function y = logFourier(b0, b1, b2, a1, a2, phi, lg10)
            a1V = a1.value;
            a2V = a2.value;
            b0V = b0.value;
            b1V = b1.value;
            b2V = b2.value;
            %%
            aSumV = 1 + a1V + a2V;
            bSumV = b0V + b1V + b2V;
            term1 = b0V .* b2V;
            term2V = b0V + b2V;
            term3V = 4 * b0V;
            term4 = term3V .* b2V;
            numerator = bSumV .* bSumV + ((term1 .* phi - b1V .* term2V - term4) .* phi);
            term6V = 1 + a2V;
            denominator = aSumV .* aSumV + (a2V .* phi - (a1V .* term6V + a2V * 4)) .* phi;
            numerator(numerator <= 0) = realmin; % eps? realmin?
            denominator(denominator <= 0) = realmin; % eps? realmin?
            eq_op = log(numerator) - log(denominator);
            reducedSum = sum(eq_op * lg10, 1);
            y = ADNode(reducedSum, b0.root, @(y) backprop_logFourier(b0, b1, b2, a1, a2, y, phi, lg10, bSumV, aSumV, numerator, denominator, term2V, term3V, term6V, b0V, b1V, b2V, a1V));
        end
        function backprop_logFourier2(b0V_node, b1V_node, b2V_node, a1V_node, a2V_node, reducedSum_node, ...
                dreducedSum_db0V, dreducedSum_db1V, dreducedSum_db2V, dreducedSum_da1V, dreducedSum_da2V)
            % Get the incoming gradient from the next layer (dL/d(reducedSum))
            dL_dreducedSum = reducedSum_node.grad;
            % Apply the chain rule: dL/d(input_i) += dL/d(reducedSum) * d(reducedSum)/d(input_i)
            b0V_node.add(dL_dreducedSum .* dreducedSum_db0V);
            b1V_node.add(dL_dreducedSum .* dreducedSum_db1V);
            b2V_node.add(dL_dreducedSum .* dreducedSum_db2V);
            a1V_node.add(dL_dreducedSum .* dreducedSum_da1V);
            a2V_node.add(dL_dreducedSum .* dreducedSum_da2V);
        end
        function reducedSum_node = logFourier2(b0V_node, b1V_node, b2V_node, a1V_node, a2V_node, phi, lg10)
            b0V_val = b0V_node.value;
            b1V_val = b1V_node.value;
            b2V_val = b2V_node.value;
            a1V_val = a1V_node.value;
            a2V_val = a2V_node.value;
            aSumV = 1 + a1V_val + a2V_val;
            bSumV = b0V_val + b1V_val + b2V_val;
            term1 = b0V_val .* b2V_val;
            term2V = b0V_val + b2V_val;
            term3V = 4 * b0V_val;
            term4 = term3V .* b2V_val;
            a2v2 = 2 * a2V_val;
            a2v4 = 2 * a2v2;
            t1phi = term1 .* phi;
            t2phi = a2V_val .* phi;
            t3phi = b1V_val .* phi;
            bbSum = bSumV .* bSumV;
            b1t2 = b1V_val .* term2V;
            numerator = bbSum + ((t1phi - b1t2 - term4) .* phi);
            term6V = 1 + a2V_val;
            denominator = aSumV .* aSumV + (t2phi - (a1V_val .* term6V + a2v4)) .* phi;
            numerator(numerator <= 0) = realmin; % eps? realmin?
            denominator(denominator <= 0) = realmin; % eps? realmin?
            eq_op = log(numerator) - log(denominator);
            reducedSumV = sum(eq_op * lg10, 1); % sum(scalar, 1) is just the scalar
            pp4_2 = phi .* (phi - 4) + 2;
            common_den_b = (phi.*(4 * term1 + b1t2 - t1phi) - bbSum) / lg10;
            common_den_b = min(common_den_b, -eps);
            twob0b1 = 2 * (b0V_val + b1V_val);
            % Derivative with respect to b0V
            dreducedSum_db0V = -((twob0b1 - t3phi) + b2V_val.* pp4_2)./common_den_b;
            % Derivative with respect to b1V
            dreducedSum_db1V = -((twob0b1 + 2*b2V_val) - phi.*term2V)./common_den_b;
            % Derivative with respect to b2V
            dreducedSum_db2V = -((2 * (b1V_val + b2V_val) - t3phi) + b0V_val.* pp4_2)./common_den_b;
            % Common denominator for a-coefficient derivatives
            common_den_a = ((a1V_val + term6V).^2 - phi .* (a2v4 - t2phi + a1V_val.*term6V)) / lg10;
            common_den_a = max(common_den_a, eps);
            % Derivative with respect to a1V
            dreducedSum_da1V = -(2*a1V_val + a2v2 - phi .* term6V + 2)./common_den_a;
            % Derivative with respect to a2V
            dreducedSum_da2V = -((a2v2 + pp4_2) - a1V_val.*(phi - 2))./common_den_a;
            reducedSum_node = ADNode(reducedSumV, b0V_node.root, @(y_node) backprop_logFourier2(b0V_node, b1V_node, b2V_node, a1V_node, a2V_node, y_node, ...
                dreducedSum_db0V, dreducedSum_db1V, dreducedSum_db2V, dreducedSum_da1V, dreducedSum_da2V));
        end
        function gdB = grpdelay(b0_arg, b1_arg, b2_arg, a1_arg, a2_arg, sw, sw2, cw, cw2)
            is_b0_node = isa(b0_arg, 'ADNode');
            is_b1_node = isa(b1_arg, 'ADNode');
            is_b2_node = isa(b2_arg, 'ADNode');
            is_a1_node = isa(a1_arg, 'ADNode');
            is_a2_node = isa(a2_arg, 'ADNode');

            b0_val = b0_arg;
            b1_val = b1_arg;
            b2_val = b2_arg;
            a1_val = a1_arg;
            a2_val = a2_arg;

            if is_b0_node, b0_val = b0_arg.value; end
            if is_b1_node, b1_val = b1_arg.value; end
            if is_b2_node, b2_val = b2_arg.value; end
            if is_a1_node, a1_val = a1_arg.value; end
            if is_a2_node, a2_val = a2_arg.value; end

            % Numerator GD
            u_b = b0_val .* sw2 + b1_val .* sw;
            v_b = b0_val .* cw2 + b1_val .* cw + b2_val;
            du_b = 2.0 .* b0_val .* cw2 + b1_val .* cw;
            dv_b = -(2.0 .* b0_val .* sw2 + b1_val .* sw);
            u2v2_b = (b0_val.^2) + (b1_val.^2) + (b2_val.^2) + 2.0 .* (b0_val.*b1_val + b1_val.*b2_val) .* cw + 2.0 .* (b0_val.*b2_val) .* cw2;
            gdB_num = (2.0 - (v_b .* du_b - u_b .* dv_b) ./ u2v2_b);

            % Denominator GD
            u_a = sw2 + a1_val .* sw;
            v_a = cw2 + a1_val .* cw + a2_val;
            du_a = 2.0 .* cw2 + a1_val .* cw;
            dv_a = -(2.0 .* sw2 + a1_val .* sw);
            u2v2_a = 1.0 + (a1_val.^2) + (a2_val.^2) + 2.0 .* (a1_val + a1_val .* a2_val) .* cw + 2.0 .* a2_val .* cw2;
            gdB_den = (2.0 - (v_a .* du_a - u_a .* dv_a) ./ u2v2_a);

            % Final group delay is the difference
            gdB_val = gdB_num - gdB_den;

            % --- 3. Determine if we need to create an ADNode output ---
            if ~is_b0_node && ~is_b1_node && ~is_b2_node && ~is_a1_node && ~is_a2_node
                gdB = gdB_val;
                return;
            end

            % --- 4. Get the root node from the ADNode inputs ---
            ad_inputs = {b0_arg, b1_arg, b2_arg, a1_arg, a2_arg};
            root_node = ADNode.find_ad_root(ad_inputs);

            % --- 5. Create the ADNode output with the custom backprop function ---
            gdB = ADNode(gdB_val, root_node, @(y_node) gd_backprop(...
                b0_arg, b1_arg, b2_arg, a1_arg, a2_arg, y_node, ...
                sw, sw2, cw, cw2, ...
                u_b, v_b, du_b, dv_b, u2v2_b, ...
                u_a, v_a, du_a, dv_a, u2v2_a));
        end
        function gd_backprop(b0_arg, b1_arg, b2_arg, a1_arg, a2_arg, y_node, ...
                sw, sw2, cw, cw2, ...
                u_b, v_b, du_b, dv_b, u2v2_b, ...
                u_a, v_a, du_a, dv_a, u2v2_a)

            % gd_backprop: Backpropagation function for the group delay calculation.

            % Incoming gradient dL/d(gdB)
            dL_d_gdB = y_node.grad;

            % Pre-compute common terms from the symbolic derivations to improve readability
            b0 = b0_arg.value;
            b1 = b1_arg.value;
            b2 = b2_arg.value;
            a1 = a1_arg.value;
            a2 = a2_arg.value;

            % Numerator-specific terms
            num_part1 = (b1 .* cw + 2 .* b0 .* cw2);
            num_part2 = (b2 + b1 .* cw + b0 .* cw2);
            num_part3 = (b1 .* sw + b0 .* sw2);
            num_part4 = (b1 .* sw + 2 .* b0 .* sw2);
            num_u2v2_denom = ( (2 .* b0 .* b1) + (2 .* b1 .* b2) ) .* cw + (b0.^2) + (b1.^2) + (b2.^2) + (2 .* b0 .* b2) .* cw2;

            % Denominator-specific terms
            den_u2v2_denom = (2 .* a1 + 2 .* a1 .* a2) .* cw + (2 .* a2) .* cw2 + a1.^2 + a2.^2 + 1;
            den_part1 = (sw2 + a1 .* sw);
            den_part2 = (2 .* sw2 + a1 .* sw);
            den_part3 = (2 .* cw2 + a1 .* cw);
            den_part4 = (a2 + cw2 + a1 .* cw);

            % --- Gradients with respect to the b_i coefficients (Numerator) ---
            if isa(b0_arg, 'ADNode')
                % Correct derivative d(gdB)/db0
                J_b0_sym_term1 = (num_part1 .* num_part2 + num_part3 .* num_part4) .* (2 .* b0 + 2 .* b1 .* cw + 2 .* b2 .* cw2);
                J_b0_sym_term2 = (2 .* b2 .* cw2 + 3 .* b1 .* (cw .* cw2 + sw .* sw2) + 4 .* b0 .* (cw2.^2 + sw2.^2));
                J_b0 = J_b0_sym_term1 ./ (num_u2v2_denom.^2) - J_b0_sym_term2 ./ num_u2v2_denom;
                b0_arg.add(dL_d_gdB .* J_b0);
            end

            if isa(b1_arg, 'ADNode')
                % Correct derivative d(gdB)/db1
                J_b1_sym_term1 = (num_part1 .* num_part2 + num_part3 .* num_part4) .* (2 .* b1 + cw .* (2 .* b0 + 2 .* b2));
                J_b1_sym_term2 = (b2 .* cw + 3 .* b0 .* (cw .* cw2 + sw .* sw2) + 2 .* b1 .* (cw.^2 + sw.^2));
                J_b1 = J_b1_sym_term1 ./ (num_u2v2_denom.^2) - J_b1_sym_term2 ./ num_u2v2_denom;
                b1_arg.add(dL_d_gdB .* J_b1);
            end

            if isa(b2_arg, 'ADNode')
                % Correct derivative d(gdB)/db2
                J_b2_sym_term1 = (num_part1 .* num_part2 + num_part3 .* num_part4) .* (2 .* b2 + 2 .* b1 .* cw + 2 .* b0 .* cw2);
                J_b2_sym_term2 = (b1 .* cw + 2 .* b0 .* cw2);
                J_b2 = J_b2_sym_term1 ./ (num_u2v2_denom.^2) - J_b2_sym_term2 ./ num_u2v2_denom;
                b2_arg.add(dL_d_gdB .* J_b2);
            end

            % --- Gradients with respect to the a_i coefficients (Denominator) ---
            if isa(a1_arg, 'ADNode')
                % Correct derivative d(gdB)/da1
                J_a1_sym_term1 = (den_part1 .* den_part2 + den_part3 .* den_part4) .* (2 .* a1 + 2 .* cw .* (a2 + 1));
                J_a1_sym_term2 = (a2 .* cw + 3 .* cw .* cw2 + 3 .* sw .* sw2 + 2 .* a1 .* (cw.^2 + sw.^2));
                J_a1 = J_a1_sym_term2 ./ den_u2v2_denom - J_a1_sym_term1 ./ (den_u2v2_denom.^2);
                a1_arg.add(dL_d_gdB .* J_a1);
            end

            if isa(a2_arg, 'ADNode')
                % Correct derivative d(gdB)/da2
                J_a2_sym_term1 = (den_part1 .* den_part2 + den_part3 .* den_part4) .* (2 .* a2 + 2 .* cw2 + 2 .* a1 .* cw);
                J_a2_sym_term2 = (2 .* cw2 + a1 .* cw);
                J_a2 = J_a2_sym_term2 ./ den_u2v2_denom - J_a2_sym_term1 ./ (den_u2v2_denom.^2);
                a2_arg.add(dL_d_gdB .* J_a2);
            end

        end
        function backprop_error_to_signal(reducedSum_node, loss_node, dloss_dreducedSum_val)
            dL_dloss = loss_node.grad;
            reducedSum_node.add(dL_dloss .* dloss_dreducedSum_val);
        end
        function loss_node = error_to_signal(target_val, y_pred)
            y_pred_val = y_pred.value;
            squared_error_val = (target_val - y_pred_val).^2;
            error_sum_sq_val = sum(squared_error_val, 'all');
            true_sum_sq_val = sum(target_val.^2, 'all');
            if true_sum_sq_val == 0
                if error_sum_sq_val == 0
                    loss_val = 0;
                else
                    loss_val = inf;
                end
            else
                loss_val = error_sum_sq_val / true_sum_sq_val;
            end
            if true_sum_sq_val == 0
                dloss_dreducedSum_val = zeros(size(y_pred_val));
            else
                diff_terms = y_pred_val - target_val;
                dloss_dreducedSum_val = (2 * diff_terms) ./ true_sum_sq_val;
            end
            loss_node = ADNode(loss_val, y_pred.root, @(y_node) backprop_error_to_signal(y_pred, y_node, dloss_dreducedSum_val));
        end
        function backprop_filter(b_arg, x_arg, y_node)
            dL_dy = y_node.grad;
            if isa(b_arg, 'ADNode')
                b_val = b_arg.value;
                if isa(x_arg, 'ADNode')
                    x_val = x_arg.value;
                else
                    x_val = x_arg;
                end
                N_x = numel(x_val);
                M_b = numel(b_val);
                % db_grad_full_conv = ADNode.fftconvolve(dL_dy, flip(x_val));
                % db_grad = db_grad_full_conv(N_x : N_x + M_b - 1);
                db_grad = zeros(size(b_val));
                if M_b <= 64
                    L = numel(dL_dy);
                    if size(b_val, 2) == 1
                        x_pad = [ flip(x_val); zeros(M_b-1, 1) ];
                        for k = 0:(M_b-1)
                            idx = (N_x + k) - (0:(L-1));    % = N_x+k + (1:-1:-L+1)
                            db_grad(k+1) = dL_dy(:).' * x_pad(idx);
                        end
                    else
                        x_pad = [ flip(x_val), zeros(1, M_b-1) ];
                        for k = 0:(M_b-1)
                            idx = (N_x + k) - (0:(L-1));    % = N_x+k + (1:-1:-L+1)
                            db_grad(k+1) = x_pad(idx) * dL_dy(:);
                        end
                    end
                else
                    obj_conv = FFTConvolver;
                    obj_conv.load(M_b, flip(x_val));
                    N_total  = numel(dL_dy) + M_b;
                    nBlocks  = ceil(N_total / M_b);
                    if size(b_val, 2) == 1
                        out = zeros(nBlocks*M_b, 1);
                        signal = [dL_dy; zeros(M_b, 1)];
                    else
                        out = zeros(1, nBlocks*M_b);
                        signal = [dL_dy, zeros(1, M_b)];
                    end
                    for b = 1:nBlocks
                        startIdx = (b-1)*M_b + 1;
                        endIdx   = min(b*M_b, N_total);
                        lenBlk   = endIdx - startIdx + 1;
                        blk = zeros(M_b,1);
                        blk(1:lenBlk) = signal(startIdx:endIdx);
                        if endIdx + M_b >= N_x
                            yblk = obj_conv.process(blk);
                            out(startIdx : startIdx+M_b-1) = yblk;
                        else
                            obj_conv.processNoReturn(blk);
                        end
                    end
                    db_grad = out(N_x : N_x + M_b - 1);
                end
                b_arg.add(db_grad);
            end
            if isa(x_arg, 'ADNode')
                b_val = b_arg.value;
                N_x = numel(x_arg.value);
                M_b = numel(b_val);
                % dx_grad_full = conv(dL_dy, flip(b_val));
                dx_grad = zeros(size(x_arg.value));
                if N_x <= 64
                    dx_grad_full = ADNode.fftconvolve(dL_dy, flip(b_val));
                    dx_grad(:) = dx_grad_full(M_b : M_b + N_x - 1);
                else
                    obj_conv = FFTConvolver;
                    obj_conv.load(N_x, flip(b_val));
                    N_total  = numel(dL_dy) + N_x;
                    nBlocks  = ceil(N_total / N_x);
                    if size(x_arg.value, 2) == 1
                        out = zeros(nBlocks*N_x, 1);
                        signal = [dL_dy; zeros(N_x, 1)];
                    else
                        out = zeros(1, nBlocks*N_x);
                        signal = [dL_dy; zeros(1, N_x)];
                    end
                    for b = 1:nBlocks
                        startIdx = (b-1)*N_x + 1;
                        endIdx   = min(b*N_x, N_total);
                        lenBlk   = endIdx - startIdx + 1;
                        blk = zeros(N_x,1);
                        blk(1:lenBlk) = signal(startIdx:endIdx);
                        if endIdx + N_x >= M_b
                            yblk = obj_conv.process(blk);
                            out(startIdx : startIdx+N_x-1) = yblk;
                        else
                            obj_conv.processNoReturn(blk);
                        end
                    end
                    dx_grad = out(M_b : M_b + N_x - 1);
                end
                x_arg.add(dx_grad);
            end
        end
        function y = filter(b_arg, a_arg, x_arg)
            if ~isnumeric(a_arg) || a_arg ~= 1
                y = builtin('filter', b_arg, a_arg, x_arg);
                return;
            end
            is_b_node = isa(b_arg, 'ADNode');
            is_x_node = isa(x_arg, 'ADNode');
            b_val = b_arg;
            x_val = x_arg;
            if is_b_node
                b_val = b_arg.value;
            end
            if is_x_node
                x_val = x_arg.value;
            end
            y_val = builtin('filter', b_val, 1, x_val);
            if is_b_node
                rt = b_arg.root;
            elseif is_x_node
                rt = x_arg.root;
            else
                y = y_val;
                return;
            end
            y = ADNode(y_val, rt, @(y_node) backprop_filter(b_arg, x_arg, y_node));
        end
        function backprop_conv(b_arg, x_arg, y_node)
            dL_dy = y_node.grad; % Incoming gradient from the next layer (dL/dy)

            % Get original dimensions for trimming
            N_x = numel(x_arg.value); % Original length of input signal x
            M_b = numel(b_arg.value); % Original length of coefficients b

            % Gradients with respect to b (filter coefficients)
            if isa(b_arg, 'ADNode')
                x_val = x_arg.value; % Numerical value of the input signal

                % The convolution dL/db is conv(dL_dy, flip(x_val)).
                % This will produce a result of length: length(dL_dy) + length(x_val) - 1.
                % We need the gradient to be the same size as b_val (M_b).
                db_grad_full_conv = conv(dL_dy, flip(x_val));

                % Trim the result to match the original size of b_val.
                % The correct portion starts from index length(x_val) and goes for M_b elements.
                db_grad = db_grad_full_conv(N_x : N_x + M_b - 1);

                b_arg.add(db_grad);
            end

            % Gradients with respect to x (input signal)
            if isa(x_arg, 'ADNode')
                b_val = b_arg.value; % Numerical value of the coefficients

                % The convolution dL/dx is conv(dL_dy, flip(b_val)).
                % This will produce a result of length: length(dL_dy) + length(b_val) - 1.
                % We need the gradient to be the same size as x_val (N_x).
                dx_grad_full_conv = conv(dL_dy, flip(b_val));

                % Trim the result to match the original size of x_val.
                % The correct portion starts from index length(b_val) and goes for N_x elements.
                dx_grad = dx_grad_full_conv(M_b : M_b + N_x - 1);

                x_arg.add(dx_grad);
            end
        end
        function y = conv(b_arg, x_arg)
            % Determine which arguments are ADNodes and extract their numerical values
            is_b_node = isa(b_arg, 'ADNode');
            is_x_node = isa(x_arg, 'ADNode');

            % Get numerical values for the forward pass
            b_val = b_arg; % Assume numeric initially
            x_val = x_arg; % Assume numeric initially

            if is_b_node
                b_val = b_arg.value;
            end
            if is_x_node
                x_val = x_arg.value;
            end

            % --- Perform the forward pass using your custom my_fir_filter (now full conv) ---
            y_val = conv(b_val, x_val);

            % Determine the root node for the new ADNode.
            if is_b_node
                rt = b_arg.root;
            elseif is_x_node
                rt = x_arg.root;
            else
                % If neither b_arg nor x_arg is an ADNode, return a plain numeric result.
                y = y_val;
                return;
            end
            y = ADNode(y_val, rt, @(y_node) backprop_conv(b_arg, x_arg, y_node));
        end
        function backprop_diff(x_node, y_node, n, dim)
            dL_dy = y_node.grad; % Incoming gradient from the output of diff
            x_val = x_node.value; % Numerical value of the original input x
            % Handle edge cases where diff output is empty
            if isempty(dL_dy) || numel(x_val) <= 1 % diff of empty or single element is empty. No gradients.
                return;
            end
            dL_dx_temp = diff(ADNode.padArray(dL_dy, n, dim), n, dim) * ((-1) ^ n);
            % Add the computed gradient to the original input node
            x_node.add(dL_dx_temp);
        end
        function out = diff(obj, n, dim)
            x_val = obj.value;
            try
                y_val = diff(x_val, n, dim);
            catch ME
                % Propagate errors from builtin diff if inputs are invalid
                rethrow(ME);
            end
            % Ensure y_val is a row vector if x_val was a row/column vector
            % This simplifies backprop logic for now.
            if isvector(x_val) && isrow(x_val) && iscolumn(y_val)
                y_val = y_val'; % Make output consistently row vector if input was row
            elseif isvector(x_val) && iscolumn(x_val) && isrow(y_val)
                % Keep as column if original was column, diff(col) gives col
            end
            out = ADNode(y_val, obj.root, @(y_node) backprop_diff(obj, y_node, n, dim));
        end
        % This function will be called for each normalized output during backprop
        % y_node: The ADNode for the normalized output (e.g., b0_norm)
        % num_node: The ADNode for the numerator (e.g., b0)
        % den_node: The ADNode for the denominator (e.g., a0)
        % num_val: The *value* of the numerator at the time of forward pass
        % den_val: The *value* of the denominator at the time of forward pass
        % type: A string indicating which coefficient this is ('b0', 'b1', etc.)
        function backprop_biquadNorm(y_node, num_node, den_node, num_val, den_val)
            % y_node.grad is the incoming gradient for this normalized output (e.g., dL/db0_norm)
            dL_dy_norm = y_node.grad;
            % d(y_norm)/d(num) = 1 / den_val
            if isa(num_node, 'ADNode')
                num_node.add(dL_dy_norm ./ den_val);
            end
            % d(y_norm)/d(den) = -num_val / (den_val ^ 2)
            if isa(den_node, 'ADNode')
                den_node.add(dL_dy_norm .* (-num_val ./ (den_val .^ 2)));
            end
        end
        function [b0_norm, b1_norm, b2_norm, a1_norm, a2_norm] = biquadNorm(b0, b1, b2, a0, a1, a2)
            %% Normalization
            % Determine the root node for the tape. It should be consistent
            % across all output ADNodes. We can pick any of the input ADNodes
            % that are indeed ADNodes.
            rootNode = [];
            if isa(b0, 'ADNode')
                rootNode = b0.root;
            elseif isa(b1, 'ADNode')
                rootNode = b1.root;
            elseif isa(b2, 'ADNode')
                rootNode = b2.root;
            elseif isa(a0, 'ADNode')
                rootNode = a0.root;
            elseif isa(a1, 'ADNode')
                rootNode = a1.root;
            elseif isa(a2, 'ADNode')
                rootNode = a2.root;
            end
            % If no ADNode inputs, no need for AD, return plain values
            if isempty(rootNode)
                b0_norm = b0 ./ a0;
                b1_norm = b1 ./ a0;
                b2_norm = b2 ./ a0;
                a1_norm = a1 ./ a0;
                a2_norm = a2 ./ a0;
                return;
            end
            if isa(b0, 'ADNode')
                b0_val = b0.value;
            else
                b0_val = b0;
            end
            if isa(b1, 'ADNode')
                b1_val = b1.value;
            else
                b1_val = b1;
            end
            if isa(b2, 'ADNode')
                b2_val = b2.value;
            else
                b2_val = b2;
            end
            if isa(a0, 'ADNode')
                a0_val = a0.value;
            else
                a0_val = a0;
            end
            if isa(a1, 'ADNode')
                a1_val = a1.value;
            else
                a1_val = a1;
            end
            if isa(a2, 'ADNode')
                a2_val = a2.value;
            else
                a2_val = a2;
            end
            b0V = b0_val ./ a0_val;
            b1V = b1_val ./ a0_val;
            b2V = b2_val ./ a0_val;
            a1V = a1_val ./ a0_val;
            a2V = a2_val ./ a0_val;
            b0_norm = ADNode(b0V, rootNode, @(y_node) backprop_biquadNorm(y_node, b0, a0, b0_val, a0_val));
            b1_norm = ADNode(b1V, rootNode, @(y_node) backprop_biquadNorm(y_node, b1, a0, b1_val, a0_val));
            b2_norm = ADNode(b2V, rootNode, @(y_node) backprop_biquadNorm(y_node, b2, a0, b2_val, a0_val));
            a1_norm = ADNode(a1V, rootNode, @(y_node) backprop_biquadNorm(y_node, a1, a0, a1_val, a0_val));
            a2_norm = ADNode(a2V, rootNode, @(y_node) backprop_biquadNorm(y_node, a2, a0, a2_val, a0_val));
        end
        function backprop_pkfb0(alpha_node, A_node, cs_node, b0_node, db0_dalpha, db0_dA, db0_dcs)
            %% Backpropagates gradients for b0.
            % b0_node.grad contains dL/db0.
            % This function calculates dL/d(alpha), dL/d(A), dL/d(cs) from b0.

            % Get the incoming gradient from the output node (dL/db0)
            dL_db0 = b0_node.grad;

            % Apply the chain rule: dL/d(input) += dL/d(output) * d(output)/d(input)
            alpha_node.add(dL_db0 .* db0_dalpha);
            A_node.add(dL_db0 .* db0_dA);
            cs_node.add(dL_db0 .* db0_dcs);
        end
        function backprop_pkfb1(alpha_node, A_node, cs_node, b1_node, db1_dalpha, db1_dA, db1_dcs)
            %% Backpropagates gradients for b1.
            % b1_node.grad contains dL/db1.

            dL_db1 = b1_node.grad;

            alpha_node.add(dL_db1 .* db1_dalpha);
            A_node.add(dL_db1 .* db1_dA);
            cs_node.add(dL_db1 .* db1_dcs);
        end
        function backprop_pkfb2(alpha_node, A_node, cs_node, b2_node, db2_dalpha, db2_dA, db2_dcs)
            %% Backpropagates gradients for b2.
            % b2_node.grad contains dL/db2.

            dL_db2 = b2_node.grad;

            alpha_node.add(dL_db2 .* db2_dalpha);
            A_node.add(dL_db2 .* db2_dA);
            cs_node.add(dL_db2 .* db2_dcs);
        end
        function backprop_pkfa0(alpha_node, A_node, cs_node, a0_node, da0_dalpha, da0_dA, da0_dcs)
            %% Backpropagates gradients for a0.
            % a0_node.grad contains dL/da0.

            dL_da0 = a0_node.grad;

            alpha_node.add(dL_da0 .* da0_dalpha);
            A_node.add(dL_da0 .* da0_dA);
            cs_node.add(dL_da0 .* da0_dcs);
        end
        function backprop_pkfa1(alpha_node, A_node, cs_node, a1_node, da1_dalpha, da1_dA, da1_dcs)
            %% Backpropagates gradients for a1.
            % a1_node.grad contains dL/da1.

            dL_da1 = a1_node.grad;

            alpha_node.add(dL_da1 .* da1_dalpha);
            A_node.add(dL_da1 .* da1_dA);
            cs_node.add(dL_da1 .* da1_dcs);
        end
        function backprop_pkfa2(alpha_node, A_node, cs_node, a2_node, da2_dalpha, da2_dA, da2_dcs)
            %% Backpropagates gradients for a2.
            % a2_node.grad contains dL/da2.

            dL_da2 = a2_node.grad;

            alpha_node.add(dL_da2 .* da2_dalpha);
            A_node.add(dL_da2 .* da2_dA);
            cs_node.add(dL_da2 .* da2_dcs);
        end
        function [b0, b1, b2, a0, a1, a2] = peaking_filter(alpha_node, A_node, cs_node)
            alpha_val = alpha_node.value;
            A_val = A_node.value;
            cs_val = cs_node.value;

            % Calculate the filter coefficients numerically (denoted with 'V' for Value)
            b0V = 1 + (alpha_val .* A_val);
            b1V = -2 * cs_val;
            b2V = 1 - (alpha_val .* A_val);
            a0V = 1 + (alpha_val ./ A_val);
            a1V = -2 * cs_val;
            a2V = 1 - (alpha_val ./ A_val);

            % --- Pre-compute the partial derivatives (from your Jacobian) ---
            % These values are computed once during the forward pass and passed
            % to the backpropagation functions via closure.

            % Derivatives for b0 = 1 + alpha * A
            db0_dalpha = A_val;
            db0_dA = alpha_val;
            db0_dcs = 0;

            % Derivatives for b1 = -2 * cs
            db1_dalpha = 0;
            db1_dA = 0;
            db1_dcs = -2;

            % Derivatives for b2 = 1 - alpha * A
            db2_dalpha = -A_val;
            db2_dA = -alpha_val;
            db2_dcs = 0;

            % Derivatives for a0 = 1 + alpha / A
            da0_dalpha = 1 ./ A_val;
            da0_dA = -alpha_val ./ (A_val.^2);
            da0_dcs = 0;

            % Derivatives for a1 = -2 * cs
            da1_dalpha = 0;
            da1_dA = 0;
            da1_dcs = -2;

            % Derivatives for a2 = 1 - alpha / A
            da2_dalpha = -1 ./ A_val;
            da2_dA = alpha_val ./ (A_val.^2);
            da2_dcs = 0;

            % Create individual ADNodes for each output coefficient.
            % Each ADNode's 'func' property points to its specific backpropagation function,
            % capturing the input nodes (alpha_node, A_node, cs_node) and the pre-computed
            % partial derivatives. The 'y' in the anonymous function is the output node itself.
            b0 = ADNode(b0V, alpha_node.root, ...
                @(y) backprop_pkfb0(alpha_node, A_node, cs_node, y, db0_dalpha, db0_dA, db0_dcs));
            b1 = ADNode(b1V, alpha_node.root, ...
                @(y) backprop_pkfb1(alpha_node, A_node, cs_node, y, db1_dalpha, db1_dA, db1_dcs));
            b2 = ADNode(b2V, alpha_node.root, ...
                @(y) backprop_pkfb2(alpha_node, A_node, cs_node, y, db2_dalpha, db2_dA, db2_dcs));
            a0 = ADNode(a0V, alpha_node.root, ...
                @(y) backprop_pkfa0(alpha_node, A_node, cs_node, y, da0_dalpha, da0_dA, da0_dcs));
            a1 = ADNode(a1V, alpha_node.root, ...
                @(y) backprop_pkfa1(alpha_node, A_node, cs_node, y, da1_dalpha, da1_dA, da1_dcs));
            a2 = ADNode(a2V, alpha_node.root, ...
                @(y) backprop_pkfa2(alpha_node, A_node, cs_node, y, da2_dalpha, da2_dA, da2_dcs));
        end
        % shelf_filter_forward.m
        function backprop_shfb0(A_node, cs_node, beta_node, b0_node, db0_dA, db0_dcs, db0_dbeta)
            %% Backpropagates gradients for b0 of the shelf filter.
            % b0_node.grad contains dL/db0.
            % This function calculates dL/d(A), dL/d(cs), dL/d(dir), dL/d(beta) from b0.

            % Get the incoming gradient from the output node (dL/db0)
            dL_db0 = b0_node.grad;

            % Apply the chain rule: dL/d(input) += dL/d(output) * d(output)/d(input)
            A_node.add(dL_db0 .* db0_dA);
            cs_node.add(dL_db0 .* db0_dcs);
            beta_node.add(dL_db0 .* db0_dbeta);
        end
        function backprop_shfb1(A_node, cs_node, beta_node, b1_node, db1_dA, db1_dcs, db1_dbeta)
            %% Backpropagates gradients for b1 of the shelf filter.
            % b1_node.grad contains dL/db1.
            dL_db1 = b1_node.grad;
            A_node.add(dL_db1 .* db1_dA);
            cs_node.add(dL_db1 .* db1_dcs);
            beta_node.add(dL_db1 .* db1_dbeta);
        end
        function backprop_shfb2(A_node, cs_node, beta_node, b2_node, db2_dA, db2_dcs, db2_dbeta)
            %% Backpropagates gradients for b2 of the shelf filter.
            % b2_node.grad contains dL/db2.
            dL_db2 = b2_node.grad;
            A_node.add(dL_db2 .* db2_dA);
            cs_node.add(dL_db2 .* db2_dcs);
            beta_node.add(dL_db2 .* db2_dbeta);
        end
        function backprop_shfa0(A_node, cs_node, beta_node, a0_node, da0_dA, da0_dcs, da0_dbeta)
            %% Backpropagates gradients for a0 of the shelf filter.
            % a0_node.grad contains dL/da0.
            dL_da0 = a0_node.grad;
            A_node.add(dL_da0 .* da0_dA);
            cs_node.add(dL_da0 .* da0_dcs);
            beta_node.add(dL_da0 .* da0_dbeta);
        end
        function backprop_shfa1(A_node, cs_node, beta_node, a1_node, da1_dA, da1_dcs, da1_dbeta)
            %% Backpropagates gradients for a1 of the shelf filter.
            % a1_node.grad contains dL/da1.
            dL_da1 = a1_node.grad;
            A_node.add(dL_da1 .* da1_dA);
            cs_node.add(dL_da1 .* da1_dcs);
            beta_node.add(dL_da1 .* da1_dbeta);
        end
        function backprop_shfa2(A_node, cs_node, beta_node, a2_node, da2_dA, da2_dcs, da2_dbeta)
            %% Backpropagates gradients for a2 of the shelf filter.
            % a2_node.grad contains dL/da2.
            dL_da2 = a2_node.grad;
            A_node.add(dL_da2 .* da2_dA);
            cs_node.add(dL_da2 .* da2_dcs);
            beta_node.add(dL_da2 .* da2_dbeta);
        end
        function [b0, b1, b2, a0, a1, a2] = shelf_filter(A_node, cs_node, beta_node, dir_val)
            A_val = A_node.value;
            cs_val = cs_node.value;
            beta_val = beta_node.value;
            % Calculate intermediate numerical values for filter coefficients
            Ap1 = A_val + 1;
            Am1 = A_val - 1;
            Ap1_cs = Ap1 .* cs_val;
            Am1_cs = Am1 .* cs_val;
            twoA = 2 .* A_val;

            b02_1 = dir_val .* Am1_cs;
            b02_2 = Ap1 + b02_1;
            b0_sum = b02_2 + beta_val;
            b0V = A_val .* b0_sum;

            b2_diff = b02_2 - beta_val;
            b2V = A_val .* b2_diff;

            b1_1 = dir_val .* Ap1_cs;
            b1_2 = Am1 + b1_1;
            b1_pre = twoA .* b1_2;
            b1_pos = dir_val .* b1_pre;
            b1V = -1 * b1_pos;

            a02_1 = dir_val .* Am1_cs;
            a02_2 = Ap1 - a02_1;
            a0V = a02_2 + beta_val;

            a2V = a02_2 - beta_val;

            a1_1 = dir_val .* Ap1_cs;
            a1_2 = Am1 - a1_1;
            a1_pre = 2 .* a1_2;
            a1V = dir_val .* a1_pre;

            % --- Pre-compute the partial derivatives (elements of the Jacobian) ---
            % These values are computed once during the forward pass and passed
            % to the backpropagation functions via closure.

            % J = jacobian([b0; b1; b2; a0; a1; a2], [A; cs; dir; beta]);
            % J is given by the user:
            % [ 2*A + beta - cs*dir + 2*A*cs*dir + 1,        A*dir*(A - 1),  A]
            % [-2*dir*(2*A + cs*dir + 2*A*cs*dir - 1),   -2*A*dir^2*(A + 1), 0]
            % [ 2*A - beta - cs*dir + 2*A*cs*dir + 1,        A*dir*(A - 1), -A]
            % [                          1 - cs*dir,           -dir*(A - 1), 1]
            % [                     -2*dir*(cs*dir - 1),   -2*dir^2*(A + 1), 0]
            % [                          1 - cs*dir,          -dir*(A - 1), -1]

            % Row 1: Derivatives for b0
            db0_dA = 2*A_val + beta_val - cs_val.*dir_val + 2*A_val.*cs_val.*dir_val + 1;
            db0_dcs = A_val.*dir_val.*(A_val - 1);
            db0_dbeta = A_val;

            % Row 2: Derivatives for b1
            db1_dA = -2*dir_val.*(2*A_val + cs_val.*dir_val + 2*A_val.*cs_val.*dir_val - 1);
            db1_dcs = -2*A_val.*dir_val.^2.*(A_val + 1);
            db1_dbeta = 0;

            % Row 3: Derivatives for b2
            db2_dA = 2*A_val - beta_val - cs_val.*dir_val + 2.*A_val.*cs_val.*dir_val + 1;
            db2_dcs = A_val.*dir_val.*(A_val - 1);
            db2_dbeta = -A_val;

            % Row 4: Derivatives for a0
            da0_dA = 1 - cs_val.*dir_val;
            da0_dcs = -dir_val.*(A_val - 1);
            da0_dbeta = 1;

            % Row 5: Derivatives for a1
            da1_dA = -2*dir_val.*(cs_val.*dir_val - 1);
            da1_dcs = -2*dir_val.^2.*(A_val + 1);
            da1_dbeta = 0;

            % Row 6: Derivatives for a2
            da2_dA = 1 - cs_val.*dir_val;
            da2_dcs = -dir_val.*(A_val - 1);
            da2_dbeta = -1;
            % Create individual ADNodes for each output coefficient.
            % Each ADNode's 'func' property points to its specific backpropagation function,
            % capturing the input nodes and the pre-computed partial derivatives.
            % The 'y' in the anonymous function is the output node itself.
            % The root is taken from one of the input nodes (e.g., A_node.root).
            b0 = ADNode(b0V, A_node.root, @(y) backprop_shfb0(A_node, cs_node, beta_node, y, ...
                db0_dA, db0_dcs, db0_dbeta));
            b1 = ADNode(b1V, A_node.root, @(y) backprop_shfb1(A_node, cs_node, beta_node, y, ...
                db1_dA, db1_dcs, db1_dbeta));
            b2 = ADNode(b2V, A_node.root, @(y) backprop_shfb2(A_node, cs_node, beta_node, y, ...
                db2_dA, db2_dcs, db2_dbeta));
            a0 = ADNode(a0V, A_node.root, @(y) backprop_shfa0(A_node, cs_node, beta_node, y, ...
                da0_dA, da0_dcs, da0_dbeta));
            a1 = ADNode(a1V, A_node.root, @(y) backprop_shfa1(A_node, cs_node, beta_node, y, ...
                da1_dA, da1_dcs, da1_dbeta));
            a2 = ADNode(a2V, A_node.root, @(y) backprop_shfa2(A_node, cs_node, beta_node, y, ...
                da2_dA, da2_dcs, da2_dbeta));
        end% backprop_b0_allpass.m
        function backprop_b0_allpass(cs_node, alpha_node, b0_node, db0_dcs, db0_dalpha)
            %% Backpropagates gradients for b0 of the Allpass filter.
            % b0_node.grad contains dL/db0.
            dL_db0 = b0_node.grad;
            cs_node.add(dL_db0 .* db0_dcs);
            alpha_node.add(dL_db0 .* db0_dalpha);
        end% backprop_b1_allpass.m
        function backprop_b1_allpass(cs_node, alpha_node, b1_node, db1_dcs, db1_dalpha)
            %% Backpropagates gradients for b1 of the Allpass filter.
            % b1_node.grad contains dL/db1.
            dL_db1 = b1_node.grad;
            cs_node.add(dL_db1 .* db1_dcs);
            alpha_node.add(dL_db1 .* db1_dalpha);
        end% backprop_b2_allpass.m
        function backprop_b2_allpass(cs_node, alpha_node, b2_node, db2_dcs, db2_dalpha)
            %% Backpropagates gradients for b2 of the Allpass filter.
            % b2_node.grad contains dL/db2.
            dL_db2 = b2_node.grad;
            cs_node.add(dL_db2 .* db2_dcs);
            alpha_node.add(dL_db2 .* db2_dalpha);
        end% backprop_a0_allpass.m
        function backprop_a0_allpass(cs_node, alpha_node, a0_node, da0_dcs, da0_dalpha)
            %% Backpropagates gradients for a0 of the Allpass filter.
            % a0_node.grad contains dL/da0.
            dL_da0 = a0_node.grad;
            cs_node.add(dL_da0 .* da0_dcs);
            alpha_node.add(dL_da0 .* da0_dalpha);
        end% backprop_a1_allpass.m
        function backprop_a1_allpass(cs_node, alpha_node, a1_node, da1_dcs, da1_dalpha)
            %% Backpropagates gradients for a1 of the Allpass filter.
            % a1_node.grad contains dL/da1.
            dL_da1 = a1_node.grad;
            cs_node.add(dL_da1 .* da1_dcs);
            alpha_node.add(dL_da1 .* da1_dalpha);
        end% backprop_a2_allpass.m
        function backprop_a2_allpass(cs_node, alpha_node, a2_node, da2_dcs, da2_dalpha)
            %% Backpropagates gradients for a2 of the Allpass filter.
            % a2_node.grad contains dL/da2.
            dL_da2 = a2_node.grad;
            cs_node.add(dL_da2 .* da2_dcs);
            alpha_node.add(dL_da2 .* da2_dalpha);
        end
        function [b0, b1, b2, a0, a1, a2] = allpass_filter(cs_node, alpha_node)
            %% Computes the forward pass of the Allpass filter and creates individual ADNodes for each coefficient.
            % cs_node: ADNode for the cs parameter.
            % alpha_node: ADNode for the alpha parameter.
            % Returns: b0, b1, b2, a0, a1, a2 as individual ADNodes.
            % Extract numerical values from the input ADNodes
            cs_val = cs_node.value;
            alpha_val = alpha_node.value;

            % Calculate the filter coefficients numerically (denoted with 'V' for Value)
            b0V = 1 - alpha_val;
            b1V = -2 * cs_val;
            b2V = 1 + alpha_val;
            a0V = 1 + alpha_val;
            a1V = -2 * cs_val;
            a2V = 1 - alpha_val;

            % --- Pre-compute the partial derivatives (elements of the Jacobian) ---
            % These values are computed once during the forward pass and passed
            % to the backpropagation functions via closure.

            % J = jacobian([b0; b1; b2; a0; a1; a2], [cs; alpha]);
            % J (simplified by user):
            % [ 0, -1]   % db0/dcs, db0/dalpha
            % [-2,  0]   % db1/dcs, db1/dalpha
            % [ 0,  1]   % db2/dcs, db2/dalpha
            % [ 0,  1]   % da0/dcs, da0/dalpha
            % [-2,  0]   % da1/dcs, da1/dalpha
            % [ 0, -1]   % da2/dcs, da2/dalpha

            % Row 1: Derivatives for b0 = 1 - alpha
            db0_dcs = 0;
            db0_dalpha = -1;

            % Row 2: Derivatives for b1 = -2 * cs
            db1_dcs = -2;
            db1_dalpha = 0;

            % Row 3: Derivatives for b2 = 1 + alpha
            db2_dcs = 0;
            db2_dalpha = 1;

            % Row 4: Derivatives for a0 = 1 + alpha
            da0_dcs = 0;
            da0_dalpha = 1;

            % Row 5: Derivatives for a1 = -2 * cs
            da1_dcs = -2;
            da1_dalpha = 0;

            % Row 6: Derivatives for a2 = 1 - alpha
            da2_dcs = 0;
            da2_dalpha = -1;

            % Create individual ADNodes for each output coefficient.
            % Each ADNode's 'func' property points to its specific backpropagation function,
            % capturing the input nodes and the pre-computed partial derivatives.
            % The 'y' in the anonymous function is the output node itself.
            % The root is taken from one of the input nodes (e.g., cs_node.root).
            b0 = ADNode(b0V, cs_node.root, ...
                @(y) backprop_b0_allpass(cs_node, alpha_node, y, db0_dcs, db0_dalpha));
            b1 = ADNode(b1V, cs_node.root, ...
                @(y) backprop_b1_allpass(cs_node, alpha_node, y, db1_dcs, db1_dalpha));
            b2 = ADNode(b2V, cs_node.root, ...
                @(y) backprop_b2_allpass(cs_node, alpha_node, y, db2_dcs, db2_dalpha));
            a0 = ADNode(a0V, cs_node.root, ...
                @(y) backprop_a0_allpass(cs_node, alpha_node, y, da0_dcs, da0_dalpha));
            a1 = ADNode(a1V, cs_node.root, ...
                @(y) backprop_a1_allpass(cs_node, alpha_node, y, da1_dcs, da1_dalpha));
            a2 = ADNode(a2V, cs_node.root, ...
                @(y) backprop_a2_allpass(cs_node, alpha_node, y, da2_dcs, da2_dalpha));
        end
        % phase_function_forward.m
        % backprop_re.m
        function backprop_re(b0_node, b1_node, b2_node, a1_node, a2_node, re_node, ...
                dre_db0, dre_db1, dre_db2, dre_da1, dre_da2)
            %% Backpropagates gradients for 're' from the phase function.
            % re_node.grad contains dL/dre.
            % This function calculates dL/d(b0), dL/d(b1), dL/d(b2), dL/d(a1), dL/d(a2) from 're'.

            % Get the incoming gradient from the output node (dL/dre)
            dL_dre = re_node.grad;

            % Apply the chain rule: dL/d(input) += dL/d(output) * d(output)/d(input)
            b0_node.add(dL_dre .* dre_db0);
            b1_node.add(dL_dre .* dre_db1);
            b2_node.add(dL_dre .* dre_db2);
            a1_node.add(dL_dre .* dre_da1);
            a2_node.add(dL_dre .* dre_da2);
        end
        % backprop_im.m
        function backprop_im(b0_node, b1_node, b2_node, a1_node, a2_node, im_node, ...
                dim_db0, dim_db1, dim_db2, dim_da1, dim_da2)
            %% Backpropagates gradients for 'im' from the phase function.
            % im_node.grad contains dL/dim.

            dL_dim = im_node.grad;

            b0_node.add(dL_dim .* dim_db0);
            b1_node.add(dL_dim .* dim_db1);
            b2_node.add(dL_dim .* dim_db2);
            a1_node.add(dL_dim .* dim_da1);
            a2_node.add(dL_dim .* dim_da2);
        end
        function [re, im] = fourierPhase(b0_node, b1_node, b2_node, a1_node, a2_node, s_R_val, s_I_val)
            %% Computes the forward pass of the phase function and creates individual ADNodes for re and im.
            % b0_node, b1_node, b2_node: ADNodes for numerator coefficients.
            % a1_node, a2_node: ADNodes for denominator coefficients.
            % s_R_val, s_I_val: Numerical values for the real and imaginary parts of 's'.
            % Returns: re, im as individual ADNodes.

            % Extract numerical values from the input ADNodes
            b0_val = b0_node.value;
            b1_val = b1_node.value;
            b2_val = b2_node.value;
            a1_val = a1_node.value;
            a2_val = a2_node.value;

            % Calculate intermediate numerical values for NR, NI, DR, DI
            s_R_sq = s_R_val .* s_R_val;
            s_I_sq = s_I_val .* s_I_val;

            NR = b0_val * (s_R_sq - s_I_sq) + b1_val * s_R_val + b2_val;
            NI = s_I_val .* (2 * s_R_val .* b0_val + b1_val);
            DR = (s_R_sq - s_I_sq) + a1_val * s_R_val + a2_val;
            DI = s_I_val .* (2 * s_R_val + a1_val);

            % Calculate the numerical values of the outputs re and im
            reV = NR .* DR + NI .* DI;
            imV = NI .* DR - NR .* DI;

            % --- Direct Port of the provided Jacobian terms ---
            % J = jacobian([re; im], [b0; b1; b2; a1; a2]);
            % Note: The .* operator is used for clarity, even for scalars, as it's common in MATLAB.

            % Row 1: Derivatives for re (dre/d[b0, b1, b2, a1, a2])
            dre_db0 = 2.*s_I_val.^2.*s_R_val.*(a1_val + 2.*s_R_val) - (s_I_val.^2 - s_R_val.^2).*(- s_I_val.^2 + s_R_val.^2 + a1_val.*s_R_val + a2_val);
            dre_db1 = s_I_val.^2.*(a1_val + 2.*s_R_val) + s_R_val.*(- s_I_val.^2 + s_R_val.^2 + a1_val.*s_R_val + a2_val);
            dre_db2 = - s_I_val.^2 + s_R_val.^2 + a1_val.*s_R_val + a2_val;
            dre_da1 = s_I_val.^2.*(b1_val + 2.*b0_val.*s_R_val) + s_R_val.*(b2_val - b0_val.*(s_I_val.^2 - s_R_val.^2) + b1_val.*s_R_val);
            dre_da2 = b2_val - b0_val.*(s_I_val.^2 - s_R_val.^2) + b1_val.*s_R_val;

            % Row 2: Derivatives for im (dim/d[b0, b1, b2, a1, a2])
            dim_db0 = s_I_val.*(a1_val.*s_I_val.^2 + a1_val.*s_R_val.^2 + 2.*a2_val.*s_R_val);
            dim_db1 = -s_I_val.*(s_I_val.^2 + s_R_val.^2 - a2_val);
            dim_db2 = -s_I_val.*(a1_val + 2.*s_R_val);
            dim_da1 = s_I_val.*(b0_val.*s_I_val.^2 + b0_val.*s_R_val.^2 - b2_val);
            dim_da2 = s_I_val.*(b1_val + 2.*b0_val.*s_R_val);

            % Create individual ADNodes for each output (re and im).
            % Each ADNode's 'func' property points to its specific backpropagation function,
            % capturing the input nodes and the pre-computed partial derivatives.
            % The root is taken from one of the input nodes (e.g., b0_node.root).
            re = ADNode(reV, b0_node.root, ...
                @(y) backprop_re(b0_node, b1_node, b2_node, a1_node, a2_node, y, ...
                dre_db0, dre_db1, dre_db2, dre_da1, dre_da2));
            im = ADNode(imV, b0_node.root, ...
                @(y) backprop_im(b0_node, b1_node, b2_node, a1_node, a2_node, y, ...
                dim_db0, dim_db1, dim_db2, dim_da1, dim_da2));
        end
        function backprop_notch(alpha_node, cs_node, y_node, dcoeff_dalpha, dcoeff_dcs)
            %% Generic backpropagation function for a Notch Filter coefficient.
            % It adds the gradient contribution from 'y_node' to 'alpha_node' and 'cs_node'.
            %
            % Inputs:
            %   alpha_node    : ADNode for the 'alpha' parameter.
            %   cs_node       : ADNode for the 'cs' parameter.
            %   y_node        : The ADNode representing the coefficient whose gradient is being propagated.
            %                   y_node.grad contains the incoming gradient (dL/d_coefficient).
            %   dcoeff_dalpha : Pre-calculated partial derivative of the coefficient w.r.t. alpha.
            %   dcoeff_dcs    : Pre-calculated partial derivative of the coefficient w.r.t. cs.

            dL_dcoeff = y_node.grad; % Incoming gradient from the next layer (dL/d_coefficient)

            % Apply the chain rule: dL/d(input) += dL/d(output) * d(output)/d(input)
            alpha_node.add(dL_dcoeff .* dcoeff_dalpha);
            cs_node.add(dL_dcoeff .* dcoeff_dcs);
        end
        function [b0_node, b1_node, b2_node, a0_node, a1_node, a2_node] = notch_filter(alpha_node, cs_node)
            %% Computes the forward pass for Notch Filter coefficients and creates ADNodes for them.
            %
            % Inputs:
            %   alpha_node : ADNode representing the 'alpha' parameter.
            %   cs_node    : ADNode representing 'cos(theta)' (cs) parameter.
            %
            % Outputs:
            %   b0_node, b1_node, b2_node : ADNodes for numerator coefficients.
            %   a0_node, a1_node, a2_node : ADNodes for denominator coefficients.

            % Extract numerical values from the input ADNodes
            alpha_val = alpha_node.value;
            cs_val = cs_node.value;

            % Calculate numerical values of the coefficients
            b0_val = 1;
            b1_val = -2 * cs_val;
            b2_val = 1;
            a0_val = 1 + alpha_val;
            a1_val = -2 * cs_val;
            a2_val = 1 - alpha_val;

            % Define partial derivatives for backpropagation based on the provided Jacobian:
            % J = [db0/dalpha, db0/dcs;
            %      db1/dalpha, db1/dcs;
            %      db2/dalpha, db2/dcs;
            %      da0/dalpha, da0/dcs;
            %      da1/dalpha, da1/dcs;
            %      da2/dalpha, da2/dcs];

            % All ADNodes will share the same root for tape management.
            % We assume alpha_node's root is the primary root for this graph.
            root_node = alpha_node.root;

            % Create ADNode for each coefficient.
            % The 'func' property for each ADNode is an anonymous function that calls
            % the generic 'notch_backprop' with the specific pre-calculated partial derivatives.
            b0_node = ADNode(b0_val, root_node, @(y_node) backprop_notch(alpha_node, cs_node, y_node, 0, 0));
            b1_node = ADNode(b1_val, root_node, @(y_node) backprop_notch(alpha_node, cs_node, y_node, 0, -2));
            b2_node = ADNode(b2_val, root_node, @(y_node) backprop_notch(alpha_node, cs_node, y_node, 0, 0));
            a0_node = ADNode(a0_val, root_node, @(y_node) backprop_notch(alpha_node, cs_node, y_node, 1, 0));
            a1_node = ADNode(a1_val, root_node, @(y_node) backprop_notch(alpha_node, cs_node, y_node, 0, -2));
            a2_node = ADNode(a2_val, root_node, @(y_node) backprop_notch(alpha_node, cs_node, y_node, -1, 0));
        end
        function backprop_bandpass(alpha_node, cs_node, y_node, dcoeff_dalpha, dcoeff_dcs)
            %% Generic backpropagation function for filter coefficients.
            % It adds the gradient contribution from 'y_node' to 'alpha_node' and 'cs_node'.
            %
            % Inputs:
            %   alpha_node    : ADNode for the 'alpha' parameter.
            %   cs_node       : ADNode for the 'cs' parameter.
            %   y_node        : The ADNode representing the coefficient whose gradient is being propagated.
            %                   y_node.grad contains the incoming gradient (dL/d_coefficient).
            %   dcoeff_dalpha : Pre-calculated partial derivative of the coefficient w.r.t. alpha.
            %   dcoeff_dcs    : Pre-calculated partial derivative of the coefficient w.r.t. cs.

            dL_dcoeff = y_node.grad; % Incoming gradient from the next layer (dL/d_coefficient)

            % Apply the chain rule: dL/d(input) += dL/d(output) * d(output)/d(input)
            alpha_node.add(dL_dcoeff .* dcoeff_dalpha);
            cs_node.add(dL_dcoeff .* dcoeff_dcs);
        end
        function [b0_node, b1_node, b2_node, a0_node, a1_node, a2_node] = bandpass_filter(alpha_node, cs_node)
            %% Computes the forward pass for Bandpass Filter coefficients and creates ADNodes for them.
            %
            % Inputs:
            %   alpha_node : ADNode representing the 'alpha' parameter.
            %   cs_node    : ADNode representing 'cos(theta)' (cs) parameter.
            %
            % Outputs:
            %   b0_node, b1_node, b2_node : ADNodes for numerator coefficients.
            %   a0_node, a1_node, a2_node : ADNodes for denominator coefficients.

            % Extract numerical values from the input ADNodes
            alpha_val = alpha_node.value;
            cs_val = cs_node.value;

            % Calculate numerical values of the coefficients
            b0_val = alpha_val;
            b1_val = 0;
            b2_val = -alpha_val;
            a0_val = 1 + alpha_val;
            a1_val = -2 * cs_val;
            a2_val = 1 - alpha_val;

            % Define partial derivatives for backpropagation based on the provided Jacobian:
            % J = [db0/dalpha, db0/dcs;
            %      db1/dalpha, db1/dcs;
            %      db2/dalpha, db2/dcs;
            %      da0/dalpha, da0/dcs;
            %      da1/dalpha, da1/dcs;
            %      da2/dalpha, da2/dcs];

            % All ADNodes will share the same root for tape management.
            % We assume alpha_node's root is the primary root for this graph.
            root_node = alpha_node.root;

            % Create ADNode for each coefficient.
            % The 'func' property for each ADNode is an anonymous function that calls
            % the generic 'filter_coeff_backprop' with the specific pre-calculated partial derivatives.
            b0_node = ADNode(b0_val, root_node, @(y_node) backprop_bandpass(alpha_node, cs_node, y_node, 1, 0)); % db0/dalpha=1, db0/dcs=0
            b1_node = ADNode(b1_val, root_node, @(y_node) backprop_bandpass(alpha_node, cs_node, y_node, 0, 0)); % db1/dalpha=0, db1/dcs=0
            b2_node = ADNode(b2_val, root_node, @(y_node) backprop_bandpass(alpha_node, cs_node, y_node, -1, 0)); % db2/dalpha=-1, db2/dcs=0
            a0_node = ADNode(a0_val, root_node, @(y_node) backprop_bandpass(alpha_node, cs_node, y_node, 1, 0)); % da0/dalpha=1, da0/dcs=0
            a1_node = ADNode(a1_val, root_node, @(y_node) backprop_bandpass(alpha_node, cs_node, y_node, 0, -2)); % da1/dalpha=0, da1/dcs=-2
            a2_node = ADNode(a2_val, root_node, @(y_node) backprop_bandpass(alpha_node, cs_node, y_node, -1, 0)); % da2/dalpha=-1, da2/dcs=0
        end
        function backprop_lowhighpass(alpha_node, cs_node, y_node, dcoeff_dalpha, dcoeff_dcs)
            %% Generic backpropagation function for filter coefficients with 3 inputs.
            % It adds the gradient contribution from 'y_node' to 'alpha_node', 'cs_node', and 'dir_node'.

            dL_dcoeff = y_node.grad; % Incoming gradient from the next layer (dL/d_coefficient)

            % Apply the chain rule: dL/d(input) += dL/d(output) * d(output)/d(input)
            alpha_node.add(dL_dcoeff .* dcoeff_dalpha);
            cs_node.add(dL_dcoeff .* dcoeff_dcs);
        end
        function [b0_node, b1_node, b2_node, a0_node, a1_node, a2_node] = lowhighpass_filter(alpha_node, cs_node, dir_val)
            %% Computes the forward pass for Custom Filter coefficients and creates ADNodes for them.
            %
            % Inputs:
            %   alpha_node : ADNode representing the 'alpha' parameter.
            %   cs_node    : ADNode representing 'cos(theta)' (cs) parameter.
            %   dir_node   : ADNode representing the 'dir' parameter.
            %
            % Outputs:
            %   b0_node, b1_node, b2_node : ADNodes for numerator coefficients.
            %   a0_node, a1_node, a2_node : ADNodes for denominator coefficients.
            % Extract numerical values from the input ADNodes
            alpha_val = alpha_node.value;
            cs_val = cs_node.value;
            % Calculate numerical values of the coefficients
            b0_val = (1 - dir_val .* cs_val) / 2;
            b1_val = dir_val .* (1 - dir_val .* cs_val);
            b2_val = (1 - dir_val .* cs_val) / 2;
            a0_val = 1 + alpha_val;
            a1_val = -2 * cs_val;
            a2_val = 1 - alpha_val;

            % Define partial derivatives for backpropagation based on the provided Jacobian:
            % J = [db0/dalpha, db0/dcs, db0/ddir;
            %      db1/dalpha, db1/dcs, db1/ddir;
            %      db2/dalpha, db2/dcs, db2/ddir;
            %      da0/dalpha, da0/dcs, da0/ddir;
            %      da1/dalpha, da1/dcs, da1/ddir;
            %      da2/dalpha, da2/dcs, da2/ddir];

            % All ADNodes will share the same root for tape management.
            root_node = alpha_node.root;
            % Create ADNode for each coefficient with their specific partial derivatives.
            % The backprop function now handles 3 input nodes.
            b0_node = ADNode(b0_val, root_node, @(y_node) backprop_lowhighpass(alpha_node, cs_node, y_node, 0, -dir_val/2));
            b1_node = ADNode(b1_val, root_node, @(y_node) backprop_lowhighpass(alpha_node, cs_node, y_node, 0, -dir_val.^2));
            b2_node = ADNode(b2_val, root_node, @(y_node) backprop_lowhighpass(alpha_node, cs_node, y_node, 0, -dir_val/2));
            a0_node = ADNode(a0_val, root_node, @(y_node) backprop_lowhighpass(alpha_node, cs_node, y_node, 1, 0));
            a1_node = ADNode(a1_val, root_node, @(y_node) backprop_lowhighpass(alpha_node, cs_node, y_node, 0, -2));
            a2_node = ADNode(a2_val, root_node, @(y_node) backprop_lowhighpass(alpha_node, cs_node, y_node, -1, 0));
        end
        function backprop_breakpoint(y)
        end
        function y = breakpoint(x)
            y = ADNode(x.value, x.root, @(y) backprop_breakpoint(y));
        end

        function y = exp(x)
            y = ADNode(exp(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, exp((x.value)))));
        end

        function y = log(x)
            y = ADNode(log(x.value), x.root, @(y) x.add(bsxfun(@rdivide, y.grad, (x.value))));
        end
        function y = log10(x)
            y = ADNode(log10(x.value), x.root, @(y) x.add(bsxfun(@rdivide, (1/log(10)) * y.grad, (x.value))));
        end

        function y = sin(x)
            y = ADNode(sin(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, cos(x.value))));
        end

        function y = sqrt(x)
            y = ADNode(sqrt(x.value), x.root, @(y) x.add(bsxfun(@rdivide, y.grad, 2*sqrt(x.value))));
        end

        function y = tan(x)
            y = ADNode(tan(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, sec(x.value) .^ 2)));
        end

        function y = besseli(nu, x)
            y = ADNode(besseli(nu, x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, besseli(nu - 1, x.value) - (nu * besseli(nu, x.value)) / x.value)));
        end
        function y = besselj(nu, x)
            y = ADNode(besselj(nu, x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, besselj(nu - 1, x.value) - (nu * besselj(nu, x.value)) / x.value)));
        end
        function y = besselk(nu, x)
            y = ADNode(besselk(nu, x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, -besselk(nu - 1, x.value) - (nu * besselk(nu, x.value)) / x.value)));
        end
        function y = bessely(nu, x)
            y = ADNode(bessely(nu, x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, bessely(nu - 1, x.value) - (nu * bessely(nu, x.value)) / x.value)));
        end
        function y = besselh(nu, K, x)
            y = ADNode(besselh(nu, K, x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, besselh(nu - 1, K, x.value) - (nu * besselh(nu, K, x.value)) / x.value)));
        end
        function y = gamma(x)
            y = ADNode(gamma(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, gamma(x.value) * psi(x.value))));
        end
        function y = gammaln(x)
            y = ADNode(gammaln(x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, psi(x.value))));
        end
        function y = psi(k, x)
            y = ADNode(psi(k, x.value), x.root, @(y) x.add(bsxfun(@times, y.grad, psi(k + 1, x.value))));
        end
        function y = wrightOmega(x)
            val = arrayfun(@wrightOmegaq, x.value);
            y = ADNode(val, x.root, @(y) x.add(bsxfun(@times, y.grad, val ./ (val + 1))));
        end

        function y = uminus(x)
            y = ADNode(-x.value, x.root, @(y) x.add(-y.grad));
        end

        function y = uplus(x)
            y = ADNode(x.value, x.root, @(y) x.add(y.grad));
        end

        function [varargout] = subsref(x, s)
            switch s(1).type
                case '()'
                    varargout{1} = ADNode(x.value(s.subs{:}), x.root, @(y) x.subs_add(s.subs, y));
                otherwise
                    [varargout{1:nargout}] = builtin('subsref', x, s);
            end
        end

        function y = subsasgn(x, s, varargin)
            switch s(1).type
                case '()'
                    if isa(varargin{1}, 'ADNode')
                        inputOrigSize = size(x.value);
                        x.value(s.subs{:}) = varargin{1}.value;
                        t = ADNode(x.value(s.subs{:}), x.root, @(y) varargin{1}.subs_move(s.subs, x, inputOrigSize));
                        y = x;
                    else
                        x.value(s.subs{:}) = varargin{1};
                        t = ADNode(x.value(s.subs{:}), x.root, @(y) x.subs_clear(s.subs));
                        y = x;
                    end
                otherwise
                    y = builtin('subsagn', x, s, varargin);
            end
        end

        function y = plus(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@plus, x1.value, x2.value), x1.root, @(y) y.plus_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@plus, x1.value, x2), x1.root, @(y) x1.add(y.grad));
                end
            else
                y = ADNode(bsxfun(@plus, x1, x2.value), x2.root, @(y) x2.add(y.grad));
            end
        end

        function y = minus(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@minus, x1.value, x2.value), x1.root, @(y) y.minus_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@minus, x1.value, x2), x1.root, @(y) x1.add(y.grad));
                end
            else
                y = ADNode(bsxfun(@minus, x1, x2.value), x2.root, @(y) x2.add(-y.grad));
            end
        end

        function y = mtimes(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value * x2.value, x1.root, @(y) y.mtimes_backprop(x1, x2));
                else
                    y = ADNode(x1.value * x2, x1.root, @(y) x1.add(y.grad * x2'));
                end
            else
                y = ADNode(x1 * x2.value, x2.root, @(y) x2.add(x1' * y.grad));
            end
        end
        function backprog_pagemtimes(x1, x2, y, NFFT)
            tmp = pagemtimes(y.grad, 'none', x2, 'transpose');
            tmp2 = sum(tmp, 3);
            x1.add(tmp2)
        end
        function y = pagemtimes(x1, x2)
            y = ADNode(pagemtimes(x1.value, x2), x1.root, @(y) backprog_pagemtimes(x1, x2, y));
        end
        function y = pinv_backward(x, y)
            y.value = y.value;
            [m, n] = size(x.value);
            pinvAh = y.value';
            if (m <= n)
                K = y.grad * y.value;
                KpinvAh = K * pinvAh;
                res = -(y.value * K)' + KpinvAh - ((x.value * y.value) * KpinvAh) + ((pinvAh * y.value) * (y.grad - K * x.value));
            else
                K = y.value * y.grad;
                pinvAhK = pinvAh * K;
                res = -(K * y.value)' + (((y.grad - x.value * K) * y.value) * pinvAh) + pinvAhK - ((pinvAhK * y.value) * x.value);
            end
            x.add(res)
        end
        function y = pinv(x)
            y = ADNode(pinv(x.value), x.root, @(y) pinv_backward(x, y));
        end
        function y = inv(x)
            y = ADNode(inv(x.value), x.root, @(y) x.add(-(y.value * (y.grad * y.value))'));
        end

        function y = times(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@times, x1.value, x2.value), x1.root, @(y) y.times_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@times, x1.value, x2), x1.root, @(y) x1.add(bsxfun(@times, y.grad, (x2))));
                end
            else
                y = ADNode(bsxfun(@times, x1, x2.value), x2.root, @(y) x2.add(bsxfun(@times, y.grad, (x1))));
            end
        end

        function y = rdivide(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(bsxfun(@rdivide, x1.value, x2.value), x1.root, @(y) y.rdivide_backprop(x1, x2));
                else
                    y = ADNode(bsxfun(@rdivide, x1.value, x2), x1.root, @(y) x1.add(bsxfun(@rdivide, y.grad, x2)));
                end
            else
                y = ADNode(bsxfun(@rdivide, x1, x2.value), x2.root, @(y) x2.add(-y.grad .* bsxfun(@rdivide, x1, x2.value .^ 2)));
            end
        end

        function y = mrdivide(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value / x2.value, x1.root, @(y) y.mrdivide_backprop(x1, x2));
                else
                    y = ADNode(x1.value / x2, x1.root, @(y) x1.add(y.grad / x2));
                end
            else
                y = ADNode(x1 / x2.value, x2.root, @(y) x2.add(- y.grad .* x1 / x2.value .^ 2));
            end
        end

        function y = mpower(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value ^ x2.value, x1.root, @(y) y.mpower_backprop(x1, x2));
                else
                    switch x2
                        case 1
                            y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad));
                        case 2
                            y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad * x1.value * 2));
                        otherwise
                            y = ADNode(x1.value ^ x2, x1.root, @(y) x1.add(y.grad * x1.value ^ (x2-1) * x2));
                    end
                end
            else
                t = x1 ^ x2.value;
                y = ADNode(t, x2.root, @(y) x2.add(y.grad * t * log(x1)));
            end
        end

        function y = power(x1, x2)
            if isa(x1, 'ADNode')
                if isa(x2, 'ADNode')
                    y = ADNode(x1.value .^ x2.value, x1.root, @(y) y.power_backprop(x1, x2));
                else
                    switch x2
                        case 1
                            y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add(y.grad));
                        case 2
                            y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add(bsxfun(@times, y.grad, x1.value * 2)));
                        otherwise
                            y = ADNode(x1.value .^ x2, x1.root, @(y) x1.add(bsxfun(@times, y.grad, x1.value .^ (x2-1) .* x2)));
                    end
                end
            else
                t = x1 .^ x2.value;
                y = ADNode(t, x2.root, @(y) x2.add(bsxfun(@times, y.grad, t .* log(x1))));
            end
        end

        function y = transpose(x)
            y = ADNode(x.value', x.root, @(y) x.transp(y.grad));
        end

        function y = length(adn)
            y = length(adn.value);
        end

        function y = size(adn, dim)
            if nargin < 2;
                y = size(adn.value);
            else
                y = size(adn.value, dim);
            end
        end

        function y = bsxfun(op, x1, x2)
            switch func2str(op)
                case 'minus'
                    y = minus(x1, x2);
                case 'plus'
                    y = plus(x1, x2);
                case 'times'
                    y = times(x1, x2);
                case 'rdivide'
                    y = rdivide(x1, x2);
                otherwise
                    assert(false, 'not implemented');
            end
        end

        function y = min(x1, x2)
            if nargin < 2
                [m, k] = min(x1.value);
                y = ADNode(m, x1.root, @(y) x1.subs_add({k}, y));
            else
                if isa(x1, 'ADNode')
                    if isa(x2, 'ADNode')
                        m = min(x1.value, x2.value);
                        y = ADNode(m, x1.root, @(y) y.minmax_backprop(x1, x2));
                    else
                        m = min(x1.value, x2);
                        y = ADNode(m, x1.root, @(y) x1.subs_match({find(m == x1.value)}, y));
                    end
                else
                    m = min(x1, x2.value);
                    y = ADNode(m, x2.root, @(y) x2.subs_match({find(m == x2.value)}, y));
                end
            end
        end

        function y = max(x1, x2)
            if nargin < 2
                [m, k] = max(x1.value);
                y = ADNode(m, x1.root, @(y) x1.subs_add({k}, y));
            else
                if isa(x1, 'ADNode')
                    if isa(x2, 'ADNode')
                        m = max(x1.value, x2.value);
                        y = ADNode(m, x1.root, @(y) y.minmax_backprop(x1, x2));
                    else
                        m = max(x1.value, x2);
                        y = ADNode(m, x1.root, @(y) x1.subs_match({find(m == x1.value)}, y));
                    end
                else
                    m = max(x1, x2.value);
                    y = ADNode(m, x2.root, @(y) x2.subs_match({find(m == x2.value)}, y));
                end
            end
        end

        function y = norm(x, d)
            if (nargin==1) d = 2; end
            y = sum(abs(x) .^ d) .^ (1/d);
        end

        function y = end(adn, dim, n)
            if n == 1
                y = length(adn.value);
            else
                y = size(adn.value, dim);
            end
        end

        function z = eq(x, y)
            if isa(y, 'ADNode')
                if isa(x, 'ADNode')
                    z = x.value == y.value;
                else
                    z = x == y.value;
                end
            else
                z = x.value == y;
            end
        end

        function z = ne(x, y)
            if isa(y, 'ADNode')
                if isa(x, 'ADNode')
                    z = x.value ~= y.value;
                else
                    z = x ~= y.value;
                end
            else
                z = x.value ~= y;
            end
        end

        function z = sign(x)
            z = sign(x.value);
        end

        function z = ge(x, y)
            if isa(y, 'ADNode')
                if isa(x, 'ADNode')
                    z = x.value >= y.value;
                else
                    z = x >= y.value;
                end
            else
                z = x.value >= y;
            end
        end

        function z = gt(x, y)
            if isa(y, 'ADNode')
                if isa(x, 'ADNode')
                    z = x.value > y.value;
                else
                    z = x > y.value;
                end
            else
                z = x.value > y;
            end
        end

        function z = le(x, y)
            if isa(y, 'ADNode')
                if isa(x, 'ADNode')
                    z = x.value <= y.value;
                else
                    z = x <= y.value;
                end
            else
                z = x.value <= y;
            end
        end

        function z = lt(x, y)
            if isa(y, 'ADNode')
                if isa(x, 'ADNode')
                    z = x.value < y.value;
                else
                    z = x < y.value;
                end
            else
                z = x.value < y;
            end
        end
        % sort
        % dummy
        function backprop_dummy(A, y)
            grad_A = y.grad;
            A.add(grad_A);
        end
        function y = dummy(A)
            y = ADNode(A.value, A.root, @(y) backprop_dummy(A, y));
        end
        % Concatenation
        function backprop_vertcat(A, B, y, reqGrad)
            if isa(A,'ADNode')
                sz_A = size(A.value);
            else
                sz_A = size(A);
            end
            grad_A = y.grad(1 : sz_A(1), :);
            grad_B = y.grad(sz_A(1) + 1 : size(y.grad, 1), :);
            if reqGrad(1)
                A.add(grad_A);
            end
            if reqGrad(2)
                B.add(grad_B);
            end
        end
        function y = vertcat(A, B)
            reqGrad = [isa(A,'ADNode'),isa(B,'ADNode')];
            if reqGrad(1)
                if reqGrad(2)
                    y = ADNode(vertcat(A.value, B.value), A.root, @(y) backprop_vertcat(A, B, y, reqGrad));
                else
                    y = ADNode(vertcat(A.value, B), A.root, @(y) backprop_vertcat(A, B, y, reqGrad));
                end
            else
                y = ADNode(vertcat(A, B.value), B.root, @(y) backprop_vertcat(A, B, y, reqGrad));
            end
        end
        function backprop_horzcat(A, B, y, reqGrad)
            if isa(A,'ADNode')
                sz_A = size(A.value);
            else
                sz_A = size(A);
            end
            grad_A = y.grad(:, 1 : sz_A(2));
            grad_B = y.grad(:, sz_A(2) + 1 : size(y.grad, 2));
            if reqGrad(1)
                A.add(grad_A);
            end
            if reqGrad(2)
                B.add(grad_B);
            end
        end
        function y = horzcat(A, B)
            reqGrad = [isa(A,'ADNode'),isa(B,'ADNode')];
            if reqGrad(1)
                if reqGrad(2)
                    y = ADNode(horzcat(A.value, B.value), A.root, @(y) backprop_horzcat(A, B, y, reqGrad));
                else
                    y = ADNode(horzcat(A.value, B), A.root, @(y) backprop_horzcat(A, B, y, reqGrad));
                end
            else
                y = ADNode(horzcat(A, B.value), B.root, @(y) backprop_horzcat(A, B, y, reqGrad));
            end
        end
    end

    methods (Access = private)
        function add(x, grad)
            %% accumulate the gradient, take sum of dimensions if needed
            if isempty(x.grad)
                if ndims(x.value) == 3
                    x.grad = grad;
                    return;
                end
                if size(x.value) == ones(1, ndims(x.value))
                    x.grad = sum(sum(grad));
                elseif size(x.value, 1) == 1
                    x.grad = sum(grad, 1);
                elseif size(x.value, 2) == 1
                    x.grad = sum(grad, 2);
                else
                    x.grad = grad;
                end
            else
                if size(x.grad) == ones(1, ndims(x.grad))
                    x.grad = x.grad + sum(sum(grad));
                elseif size(x.grad, 1) == 1
                    x.grad = x.grad + sum(grad, 1);
                elseif size(x.grad, 2) == 1
                    x.grad = x.grad + sum(grad, 2);
                else
                    x.grad = bsxfun(@plus, x.grad, grad);
                end
            end
        end
        function transp(x, grad)
            x.grad = grad';
        end

        function subs_add(x, subs, y)
            %% accumulate the gradient with subscripts
            gradt = y.grad;
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            old = x.grad(subs{:});
            if size(old, 1) == 1 && size(old, 2) == 1
                x.grad(subs{:}) = old + sum(sum(gradt));
            elseif size(old, 1) == 1
                x.grad(subs{:}) = old + sum(gradt, 1);
            elseif size(old, 2) == 1
                sm = sum(gradt(1 : min(size(old, 1), size(gradt, 1)), :), 2);
                if ~isempty(sm)
                    x.grad(subs{:}) = old + sm;
                else
                    x.grad(subs{:}) = old;
                end
            else
                x.grad(subs{:}) = old + gradt;
            end
        end

        function subs_match(x, subs, y)
            %% accumulate the gradient with subscripts
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            if size(x.grad) == ones(1, ndims(x.grad))
                x.grad = x.grad + sum(y.grad(subs{:}));
            else
                x.grad(subs{:}) = x.grad(subs{:}) + y.grad(subs{:});
            end
        end

        function subs_clear(x, subs)
            %% clear the gradient with subscripts
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            x.grad(subs{:}) = 0;
        end

        function subs_move(x, subs, y, inputOrigSize)
            %% accumulate the gradient with subscripts
            if size(y.grad) == ones(1, ndims(y.grad))
                y.grad = repmat(y.grad, size(y.value));
            end
            gradt = y.grad(subs{:});
            y.grad(subs{:}) = 0;
            if any(size(y.grad) ~= inputOrigSize) % Remove excessive gradient data
                y.grad(subs{:}) = [];
            end
            if isempty(x.grad)
                x.grad = zeros(size(x.value));
            end
            old = x.grad;
            if size(old, 1) == 1 && size(old, 2) == 1
                x.grad = old + sum(sum(gradt));
            elseif size(old, 1) == 1
                x.grad = old + sum(gradt, 1);
            elseif size(old, 2) == 1
                x.grad = old + sum(gradt, 2);
            else
                x.grad = old + gradt;
            end
        end

        function plus_backprop(y, x1, x2)
            x1.add(y.grad);
            x2.add(y.grad);
        end

        function minus_backprop(y, x1, x2)
            x1.add(y.grad);
            x2.add(-y.grad);
        end

        function mtimes_backprop(y, x1, x2)
            x1.add(y.grad * x2.value');
            x2.add(x1.value' * y.grad);
        end

        function times_backprop(y, x1, x2)
            x1.add(bsxfun(@times, y.grad, x2.value));
            x2.add(bsxfun(@times, y.grad, x1.value));
        end

        function rdivide_backprop(y, x1, x2)
            x1.add(bsxfun(@rdivide, y.grad, x2.value));
            x2.add(-y.grad .* bsxfun(@rdivide, x1.value, x2.value .^ 2));
        end

        function mrdivide_backprop(y, x1, x2)
            x1.add(y.grad / x2.value);
            x2.add(-y.grad .* x1.value / x2.value .^ 2);
        end

        function mpower_backprop(y, x1, x2)
            x1.add(y.grad * x1.value ^ (x2.value-1) * x2.value);
            x2.add(y.grad * y.value * log(x1.value));
        end

        function power_backprop(y, x1, x2)
            x1.add(y.grad .* x1.value .^ (x2.value-1) .* x2.value);
            x2.add(y.grad .* y.value .* log(x1.value));
        end

        function minmax_backprop(y, x1, x2)
            x1.subs_match({find(y.value == x1.value)}, y);
            x2.subs_match({find(y.value == x2.value)}, y);
        end
    end

end