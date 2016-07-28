[data, txt] = xlsread('burgalossi_data2.xlsx');
varnames = txt(2,1:end-1);

X = zscore(data(:,12:19)); % primary morphological parameters
y = data(:,3) > 0;         % label: silent or active

N = size(X,1);

alpha = [.05 .1 .5 .9 .95];
A = length(alpha);

yhat = NaN(N,A);
w = NaN(size(X,2),N,A);

rng(0);

for a=1:A
    for i=1:N
        disp([a,i])
        idx = setdiff(1:N, i);
        [b,stats] = lassoglm(X(idx,:),y(idx),'binomial','CV',10,'alpha',alpha(a));
        w(:,i,a) = b(:,stats.Index1SE);
        preds = (X*w(:,i,a)) > 0;
        yhat(i,a) = preds(i);
    end
    pc(a) = mean(yhat(:,a)==y);
end


%%
f  = Figure('size',[60 40]);

wm = squeeze(mean(w,2));
ws = squeeze(std(w,[],2)/sqrt(N));
wx = bsxfun(@plus,repmat((1:8)',1,5),-.2:0.1:.2);

errorbar(wx,wm,ws,'.')

xlim([0 11])
ylim([-.5 1.5])

set(gca,'xtick',1:8)

line([0 11],[0 0],'color','k','linestyle',':')

l = {};
for a=1:length(alpha)
    l{a} = sprintf('\\alpha=%.2f\n',alpha(a));
end
ll = legend(l);
set(ll,'box','off')

f.cleanup()

%% shuffle analysis
rng(0);

for j=1:100
    ys = y(randperm(length(y)));
    yshat = NaN(N,1);
    for i=1:N
        disp([j,i])
        idx = setdiff(1:N, i);
        [b,stats] = lassoglm(X(idx,:),ys(idx),'binomial','CV',10,'alpha',alpha(4));
        ww = b(:,stats.Index1SE);
        preds = (X*ww) > 0;
        yshat(i) = preds(i);
    end
    pc(j) = mean(yshat==ys);    
end

