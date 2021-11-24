% load training data
train = load('data\TrainingSamplesDCT_subsets_8.mat');
strategy1 = load('data\Prior_1.mat');
strategy2 = load('data\Prior_2.mat');
load('data\Alpha.mat');

% Extract features
cheetah = imread('data/cheetah.bmp');
cheetah = im2double(cheetah);
[l, w] = size(cheetah);
feature = zeros(64,(l-8)*(w-8));
M = readmatrix('Zig-Zag Pattern.txt');
% get 8x8 blocks with sliding window
for i = 1:w-8
    for j = 1:l-8
        block = cheetah(j:j+7,i:i+7);
        feature(:,j+(i-1)*(l-8)) = pre_process(block,M)';
    end
end

mask = cast(imread('data/cheetah_mask.bmp')/250, 'int8');
P_error_Bayes = zeros(1,length(alpha));
P_error_ML = zeros(1,length(alpha));
P_error_MAP = zeros(1,length(alpha));

for d = 1:4
    D_BG = "D"+num2str(d)+"_BG";
    D_FG = "D"+num2str(d)+"_FG";
    % Prior probability
    [P_cheetah, P_grass] = get_priors(train.(D_BG), train.(D_FG));
    
    % Classification - stratergy 1
    for i = 1:length(alpha)
        var_prior = diag(strategy1.W0.*alpha(i));
        [predictive_mean_BG, predictive_var_BG, sample_mean_BG, sample_var_BG, posterior_mean_BG] = get_predictive_mean_variance(train.(D_BG), var_prior, strategy1.mu0_BG);
        [predictive_mean_FG, predictive_var_FG, sample_mean_FG, sample_var_FG, posterior_mean_FG] = get_predictive_mean_variance(train.(D_FG), var_prior, strategy1.mu0_FG);
        % Bayes estimate
        A_Bayes = BDR(feature, predictive_mean_FG, predictive_var_FG, predictive_mean_BG, predictive_var_BG, P_grass, P_cheetah);
        P_error_Bayes(i) = get_error(mask, A_Bayes, l, w);
        % ML estimate
        A_ML = BDR(feature, sample_mean_FG', sample_var_FG, sample_mean_BG', sample_var_BG, P_grass, P_cheetah);
        P_error_ML(i) = get_error(mask, A_ML, l, w);
        % MAP estimate
        A_MAP = BDR(feature, posterior_mean_FG, sample_var_FG, posterior_mean_BG, sample_var_BG, P_grass, P_cheetah);
        P_error_MAP(i) = get_error(mask, A_MAP, l, w);
    end
    figure;
    plot(alpha,P_error_Bayes, alpha, P_error_ML, alpha, P_error_MAP)
    set(gca,'XScale', 'log')
    title("PE vs alpha for stratergy 1 for dataset "+d);
    legend('Bayes', 'MLE', 'MAP');
    
    % stratergy 2
    for i = 1:length(alpha)
        var_prior = diag(strategy2.W0.*alpha(i));
        [predictive_mean_BG, predictive_var_BG, sample_mean_BG, sample_var_BG, posterior_mean_BG] = get_predictive_mean_variance(train.(D_BG), var_prior, strategy2.mu0_BG);
        [predictive_mean_FG, predictive_var_FG, sample_mean_FG, sample_var_FG, posterior_mean_FG] = get_predictive_mean_variance(train.(D_FG), var_prior, strategy2.mu0_FG);
        % Bayes estimate
        A_Bayes = BDR(feature, predictive_mean_FG, predictive_var_FG, predictive_mean_BG, predictive_var_BG, P_grass, P_cheetah);
        P_error_Bayes(i) = get_error(mask, A_Bayes, l, w);
        % ML estimate
        A_ML = BDR(feature, sample_mean_FG', sample_var_FG, sample_mean_BG', sample_var_BG, P_grass, P_cheetah);
        P_error_ML(i) = get_error(mask, A_ML, l, w);
        %  MAP estimate
        A_MAP = BDR(feature, posterior_mean_FG, sample_var_FG, posterior_mean_BG, sample_var_BG, P_grass, P_cheetah);
        P_error_MAP(i) = get_error(mask, A_MAP, l, w);
    end
    figure;
    plot(alpha,P_error_Bayes, alpha, P_error_ML, alpha, P_error_MAP)
    set(gca,'XScale', 'log')
    title("PE vs alpha for stratergy 2 for dataset "+d);
    legend('Bayes', 'MLE', 'MAP');
   
end


function [P_error] = get_error(mask,A, l, w)
    A = reshape(A,[l-8,w-8]);
    A = padarray(A,[8,8],0,'post');
    error = cast(A,'int8')-mask;
    P_error = (sum(error(error>0))-sum(error(error<0)))/numel(mask);
end

function [P_1, P_0] = get_priors(D_BG, D_FG)
    n = length(D_FG)+length(D_BG);
    P_1 = length(D_FG)/n;
    P_0 = length(D_BG)/n;
end

function [predictive_mean, predictive_var, mu, var, posterior_mean] = get_predictive_mean_variance(data, var_prior, mean_prior)
    N = length(data);
    mu = mean(data);
    var = cov(data);
    sigma_n = 1/N*var;
    posterior_mean = var_prior*(inv(var_prior + sigma_n)*mu') + sigma_n*(inv(var_prior + sigma_n)*mean_prior');
    posterior_var = var_prior*(inv(var_prior + sigma_n)*sigma_n);
    predictive_mean = posterior_mean;
    predictive_var = var + posterior_var;
    % predictive_var = max(posterior_var,var);
end


function [feature] = pre_process(image_block, M)
    %This function performs DCT, falttens the output with zig-zag scan and
    %returns the 64D feature for the 8x8 image block input
    % Returns:
    % A 64D feature for each 8x8 block
    DCT_output = dct2(image_block);
    % zig-zag scan
    zig_zag_dct(M+1) = DCT_output;
    feature = zig_zag_dct;
end

function [Y] = BDR(x, P_x1_mean, P_x1_var, P_x0_mean, P_x0_var, P_grass, P_Cheetah)
    %This function assigns output class of 1 for cheetah and 0 for grass
    %   Returns:
    %   Y - the state variable based on Bayesian Decision Rule
    Y = log_gaussian_pdf(x,P_x0_mean,P_x0_var)-2*log(P_grass) >= log_gaussian_pdf(x,P_x1_mean,P_x1_var)-2*log(P_Cheetah);
end

function [Y] = log_gaussian_pdf(X,m,s)
    % utility function returns the log of the Gaussian distribution y for given mean and
    % variance
    % input X - feature matrix of dimension dxn where d is number of dimensions
    % and n is number of samples
    sigma_inv = inv(s);
    mean_adjusted = X-m;
    const = size(X,1)*log(2*pi) + log(det(s));
    prod = mean_adjusted'*sigma_inv;
    square_form = zeros(1,size(X,1));
    for i = 1:size(X,2)
        square_form(i) = prod(i,:)*mean_adjusted(:,i);
    end
    Y = const + square_form;
end