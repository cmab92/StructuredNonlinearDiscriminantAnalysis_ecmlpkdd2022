function [trD, teD, trL, teL] = generate_ARIMA_data(arimacoeffs, dim, spc, tts)
%% dim:= data dimension
%% spc:= array with integers determining the number of samples per class
%% tts:= train-test-split
%% ...
    % num of classes:
    C = length(arimacoeffs);
    if C == length(spc)
        cspc = cumsum(spc);
        X = zeros([dim, cspc(end)]);
        X_labels = zeros([1, cspc(end)]);
        for c = 1:C
            coeffs = arimacoeffs{c};
            Mdl = arima('Constant',coeffs{1},'MA',coeffs{2},'AR',coeffs{3},'Variance',1);            
            X_ = simulate(Mdl, spc(c)*dim ); 
            X(:, cspc(c)-spc(c)+1:cspc(c)) = reshape(X_, [dim, spc(c)]);
            X_labels(cspc(c)-spc(c)+1:cspc(c)) = c;
        end
        idx = randperm(cspc(end));
        trD = X(:, idx(1:ceil(cspc(end)*tts)));
        teD = X(:, idx(ceil(cspc(end)*tts):end));
        trL = X_labels(idx(1:ceil(cspc(end)*tts)));
        teL = X_labels(idx(ceil(cspc(end)*tts):end));
    else
        error("Number of samples doesnt fit to number of filterkernels")
    end
    if length(unique(teL))>length(unique(trL))
        error("More classes in test data than in training data...")
    end
end