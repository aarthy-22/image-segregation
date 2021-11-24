%%% Image Segmentation using BDR
load('data/TrainingSamplesDCT_8_new.mat')

% Extract features
cheetah = imread('cheetah.bmp');
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

%%% Prior Probability
class_distribution = [ones(length(TrainsampleDCT_FG),1); zeros(length(TrainsampleDCT_BG),1)];
figure;
title('Histogram of prior')
histogram(class_distribution, 'Normalization','pdf')
% ML Estimation for binomial prior distribution
n = length(TrainsampleDCT_FG)+length(TrainsampleDCT_BG);
C1 = length(class_distribution(class_distribution == 1));
C0 = length(class_distribution(class_distribution == 0));
P_cheetah = C1/n;
P_grass = C0/n;
fprintf('Prior probabilities of Y are: P(Y=cheetah) = %.3f and P(Y=grass)= %.3f',P_cheetah,P_grass);

%%% MLE for class conditional densities
P_x0_mean = 1/C0.*sum(TrainsampleDCT_BG);
P_x1_mean = 1/C1.*sum(TrainsampleDCT_FG);
P_x0_var = 1/C0.*(TrainsampleDCT_BG - P_x0_mean)'*(TrainsampleDCT_BG - P_x0_mean);
P_x1_var = 1/C1.*(TrainsampleDCT_FG - P_x1_mean)'*(TrainsampleDCT_FG - P_x1_mean);
training_data_set = [TrainsampleDCT_BG; TrainsampleDCT_FG];

%Classifier 
A_64D = BDR(feature, P_x1_mean, P_x1_var, P_x0_mean, P_x0_var, P_grass, P_cheetah);
% assign output to top left pixel of the blockdiag
A_64D = reshape(A_64D,[l-8,w-8]);
A_64D = padarray(A_64D,[8,8],0,'post');

%%% Classified Image
figure;
colormap(gray(255))
imagesc(A_64D)
title('Classification mask with 64-D Gaussian')

mask = cast(imread('cheetah_mask.bmp')/250, 'int8');
error_64D = cast(A_64D,'int8')-mask;
P_error = (sum(error_64D(error_64D>0))-sum(error_64D(error_64D<0)))/numel(A_64D);
fprintf('The error for 64-D Gaussian is %.3f \n',P_error)

 
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
mean_adjusted = X-m';
const = size(X,1)*log(2*pi) + log(det(s));
% wrong dimensions - the last multiplication has to be with each individual
% row only
prod = mean_adjusted'*sigma_inv;
square_form = zeros(1,size(X,1));
for i = 1:size(X,2)
    square_form(i) = prod(i,:)*mean_adjusted(:,i);
end
Y = const + square_form;
end