%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% This MATLAB-Script allows to generate Figs. 2 and 3. (Fig. 4 with some small adjustments). 
%% Line 13 has to be adjusted in order to use different data sets.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; close all; clear all; rng(300);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% generate synthetic data
%%
dim = 512;
samples_per_class = [50, 50, 50, 50];
train_test_split = 0.5;
for lidx = 1:4
    c = 2^(lidx-1);
    factor = 1/(c+1);
    arimacoeffs{lidx} = {0, factor*ones([1, c]), factor*ones([1, c])};
end
[X, X_t, y, y_t] = generate_ARIMA_data(arimacoeffs, dim, samples_per_class, train_test_split);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CDA
%% normalize
X = normc(X);
X_t = normc(X_t);
%% general stuff
[D, N] = size(X);
c_labels = unique(y);
C = length(c_labels);
% circulant "filterkernel" width
L = 8;
%% split classes
X_c = cell([C, 1]);
X_c_t = cell([C, 1]);
for c = 1:C
    X_c{c} = X(:, y==c_labels(c));
    X_c_t{c} = X_t(:, y_t==c_labels(c));
end
%% a priori class probablities
P_c = zeros([1, C]);
for c = 1:C
    P_c(c) = sum(y==c_labels(c))/N;
end
%% class-specific and overall sample mean
mean_0 = mean(X, 2);
mean_c = zeros([D, C]);
for c = 1:C
    mean_c(:, c) = mean(X_c{c}, 2);
end
%% within-class scatter matrix
Z_W = zeros(L);
for c = 1:C
    z = ifft(sum(fft(X_c{c} - mean_c(:, c)).*conj(fft(X_c{c} - mean_c(:, c))), 2));
    Z_W = Z_W + P_c(c)*toeplitz(z(1:L));
end
%% beneath-class scatter matrix
Z_B = zeros(L);
for c = 1:C
    z = ifft(sum(fft(mean_c(:, c) - mean_0).*conj(fft(mean_c(:, c) - mean_0)), 2));
    Z_B = Z_B + P_c(c)*toeplitz(z(1:L));
end
%% filters
[vec, val] = eig(pinv(Z_W)*Z_B);
[~, idx] = sort(diag(val), 'descend');
vec = real(vec(:, idx));
%% project data
X_p = zeros([C-1, size(X, 2)]);
X_p_t = zeros([C-1, size(X_t, 2)]);
for c = 1:C-1
    X_p(c, :) = sum(ifft(fft([vec(:, c); zeros([D-L, 1])]).*fft(X)).^2, 1).^(1/2);
end
for c = 1:C-1
    X_p_t(c, :) = sum(ifft(fft([vec(:, c); zeros([D-L, 1])]).*fft(X_t)).^2, 1).^(1/2);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LDA
%% scatter matrices
B = zeros(D);
W = zeros(D);
for c = 1:C
    B = B + P_c(c)*(mean_c(:, c) - mean_0)*(mean_c(:, c) - mean_0)';
end
for c = 1:C
    W = W + P_c(c)*cov((X_c{c} - mean_c(:, c))');
end
[vec_, val] = eig(pinv(W)*B);
[val, idx] = sort(diag(val), 'descend');
vec_ = real(vec_(:, idx));
g = vec_(:, 1:C-1);
%% projections:
proj_cd_train = cell([C, C-1]);
proj_cd_test = cell([C, C-1]);
for c = 1:C
    for i = 1:C-1
        class_data_train = X(:, y==c_labels(c));
        class_data_test = X_t(:, y_t==c_labels(c));
        proj_cd_train{c, i} = (class_data_train'*g(:, i))';
        proj_cd_test{c, i} = (class_data_test'*g(:, i))';
    end
end
%% estimate class-densities
psd_c = zeros([D, C]);
for c = 1:C
    psd = sum(fft(X_c{c} - mean_c(:, c)).*conj(fft(X_c{c} - mean_c(:, c))), 2);
    psd_c(:, c) = psd;
end
psd_c = normc(psd_c);   % scaling
%% scatterplot projected data
some_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b', 'c', 'm', 'y', 'k'];
figure(1)
title("Fig. 2")
for c = 1:C
    subplot(1,3,1)
    plot(X_c{c}(:, 1) + c, some_colors(c), 'linewidth', 2); hold on;
    plot(X_c{c}(:, 2) + c, some_colors(c), 'linewidth', 0.25); hold on; axis off;
    subplot(1,3,2)
    scatter3(X_p(1, y==c_labels(c)), X_p(2, y==c_labels(c)), X_p(3, y==c_labels(c)), some_colors(c), 'Displayname', int2str(c), 'marker', 'o'); hold on;
    scatter3(X_p_t(1, y_t==c_labels(c)), X_p_t(2, y_t==c_labels(c)), X_p_t(3, y_t==c_labels(c)), some_colors(c), 'marker', 'x', 'Displayname', int2str(c));
    subplot(1,3,3)
    scatter3(proj_cd_train{c, 1}, proj_cd_train{c, 2}, proj_cd_train{c, 3}, some_colors(c), 'Displayname', int2str(c), 'marker', 'o'); hold on;
    scatter3(proj_cd_test{c, 1}, proj_cd_test{c, 2}, proj_cd_test{c, 3}, some_colors(c), 'marker', 'x', 'Displayname', int2str(c));
end
subplot(1,3,1)
subplot(1,3,2); subtitle("CDA"); view(45,45);
subplot(1,3,3); subtitle("LDA");
%% psd and discriminants
figure(2)
subplot(1,2,1)
plot(vec(:, 1) + 1); hold on; axis off
plot(vec(:, 2) + 2);
plot(vec(:, 3) + 3);
subplot(1,2,2)
discr_spec = normc(abs(fft(vec, dim)));
plot(discr_spec(:, 1)); hold on; axis off
plot(discr_spec(:, 2), '-')
plot(discr_spec(:, 3), '-.'); xlim([0, ceil(dim/2)])
for c = 1:C
    plot(psd_c(:, c), some_colors(c))
end
