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
        function y = reshape(x, shp)
            y = ADNode(reshape(x.value, shp), x.root, @(y) x.add(reshape(y.grad, size(x.value))));
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