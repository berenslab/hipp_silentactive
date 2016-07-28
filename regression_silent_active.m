%% Introduction
%
% Here, we are trying to find out whether morphological features of
% hippocampal granule cells are predictive of whether these cells are
% active or silent.
%
% There are primary anatomical features, mainly the total dendritic length
% of dendrites at a certain branching order, and derived indices that
% combine several primary metrics.
% 
% For the analysis we use the primary measures, because they are easier to
% interpret and the derived metrics are quite correlated with those.

%% Regression analysis
%
% The regression analysis is based on sparse logistic regression with an
% elastic-net penalty. We evaluate multiple sparsity trade-offs via nested
% cross-validation. 

[data, txt] = xlsread('burgalossi_data.xlsx');

X = zscore(data(:,12:21)); % primary morphological parameters
y = data(:,3) > 0;         % label: silent or active

varnames = {'Total', 'Order 1', 'Order 2', 'Order 3', 'Order 4', ...
            'Order 5', 'Order 6', 'Order 7', '# primary dendrites', ...
            '# dendritic endings'};

N = size(X,1);

% sparsity parameters
alpha = [.05 .1 .5 .9 .95];
A = length(alpha);

% intialize
yhat = NaN(N,A);
w = NaN(size(X,2),N,A);
pc = NaN(A,1);

% fix seed
rng(0);

for a=1:A
    for i=1:N
        disp([a,i])
        idx = setdiff(1:N, i);  % leave-one-out cross validation
        [b,stats] = lassoglm(X(idx,:),y(idx),'binomial','CV',10,'alpha',alpha(a));
        w(:,i,a) = b(:,stats.Index1SE);
        preds = (X*w(:,i,a)) > 0;
        yhat(i,a) = preds(i);
    end
    pc(a) = mean(yhat(:,a)==y);
end

%% Visualization
%
% Plot the weights with SEM over cross-validation folds

f  = Figure('size',[60 50]);

wm = squeeze(mean(w,2));
ws = squeeze(std(w,[],2)/sqrt(N));
wx = bsxfun(@plus,repmat((1:size(X,2))',1,5),-.2:0.1:.2);

errorbar(wx,wm,ws,'.')

xlim([0 11])
ylim([-.5 1.5])

set(gca,'xtick',1:size(X,2))
set(gca,'xticklabel',varnames,'XTickLabelRotation',45)

line([0 11],[0 0],'color','k','linestyle',':')

l = {};
for a=1:length(alpha)
    l{a} = sprintf('\\alpha=%.2f\n',alpha(a));
end
ll = legend(l);
set(ll,'box','off','location','NorthWest')

f.cleanup()

%% Shuffle analysis
%
% To assess whether the classification performance is significantly better
% than chance, we perform a shuffle analysis on the labels. For each
% iteration, we shuffle the labels and reclassify using the same
% cross-validation procedure as before.

rng(0);

pc_shuffle = NaN(500,1);

for j=1:500
    ys = y(randperm(length(y)));        % randomly assign labels
    yshat = NaN(N,1);
    for i=1:N
        disp([j,i])
        idx = setdiff(1:N, i);
        [b,stats] = lassoglm(X(idx,:),ys(idx),'binomial','CV',10,'alpha',alpha(2));
        ww = b(:,stats.Index1SE);
        preds = (X*ww) > 0;
        yshat(i) = preds(i);
    end
    pc_shuffle(j) = mean(yshat==ys);    
end

