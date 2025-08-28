obj_conv = FFTConvolver;
rng(1)
ir = randn(102400, 1);
ir = ir / max(abs(ir)) * 0.1;
ir(1) = 1;
%% Demo 1 regular single large block convolution
obj_conv.load(32, ir);
signal = [1; zeros(2048, 1)];
rng(1)
signal = [randn(513, 1) * 0.1; zeros(1535, 1)];
out = obj_conv.process(signal);
subplot(2, 1, 1)
plot(signal)
hold on
plot(out)
hold off
axis tight
title('Regular convolution')
%% Demo 2 multiple small block convolution, change IR in the middle
obj_conv.load(32, ir);
signal = [1; zeros(512, 1)];
rng(1)
signal = randn(size(signal)) * 0.1;
out = obj_conv.process(signal); % Convolve 513 samples
%% Refresh IR
ir = randn(size(ir));
ir = ir / max(abs(ir)) * 0.1;
ir(1) = 1;
obj_conv.update(ir);
%% Convolve as normal
signal(:) = 0;
out = [out; obj_conv.process(signal)]; % Convolve another 513 samples
out = [out; obj_conv.process(signal)]; % Convolve another 513 samples
out = [out; obj_conv.process(signal)]; % Convolve another 513 samples
subplot(2, 1, 2)
plot(out)
axis tight
title('Convolution, change IR in the middle')