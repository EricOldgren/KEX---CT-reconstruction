% Example script for working with Helsinki Tomography Challenge 2022 data.
% To run this script, you will need to add the following MATLAB toolboxes
% to your path:
%
% HelTomo - Helsinki Tomography Toolbox
% https://github.com/Diagonalizable/HelTomo
%
% The ASTRA Toolbox
% https://www.astra-toolbox.com/
%
% Spot – A Linear-Operator Toolbox
% https://www.cs.ubc.ca/labs/scl/spot/
%
% Note that using the above toolboxes for the Challenge is by no means
% compulsory: the metadata for each dataset contains a full specification
% of the measurement geometry, and the competitors are free to use any and
% all computational tools they want to in computing the reconstructions and
% segmentations.
%
%   Alexander Meaney, University of Helsinki
%   Created:            5.8.2022
%   Last edited:        12.8.2022

% Workspace cleanup
close all;
clear;
clc;

% Specify phantom data, one from {'ta', 'td', 'tc', 'td, 'solid_disc'}
dataset = 'ta';

% Load full dataset of teaching phantom
load(['htc2022_' dataset '_full']);

% Load reference reconstruction and segmentation
load(['htc2022_' dataset '_full_recon_fbp']);
load(['htc2022_' dataset '_full_recon_fbp_seg']);

% Specify difficulty level 1-7 for limited angle data
difficulty = 7;

% Lmited angle data opening angle
switch difficulty
   case 1
      openingAngle = 90;
   case 2
      openingAngle = 80;
   case 3
      openingAngle = 70;
   case 4
      openingAngle = 60;
   case 5
      openingAngle = 50;
   case 6
      openingAngle = 40;
   case 7
      openingAngle = 30;
   otherwise
      error('Invalid difficulty level.');
end

% Choose random starting angle
startingAngle = randi([0 (360-openingAngle)]);

% Specify projection angles
angles = startingAngle : 0.5 : (startingAngle + openingAngle);

% Create limited angle dataset
CtDataLimited = subsample_sinogram(CtDataFull, angles);

% IMPORTANT NOTE:
% The data structure called CtDataLimited is what your limited-angle
% reconstruction algorithm should take in as the data. The actual test data
% for the challenge will have this format and variable name. Your algorithm
% does not need subsample the test data as this has already been done. The 
% above procedure allows you, the competitor, to freely choose the type of
% subsampling schemes (number of projections, location and size of the 
% angular range etc.) you want to use to teach your algorithm and to 
% compare your limited-angle reconstructions to the ground truth 
% reconstruction and its segmentation.

% Specify reconstruction edge length (fixed at 512 for HTC 2022)
n = 512;

% Create CT operator modeling the limited angle imaging geometry
% (choose suitable function -  CUDA version is recommended)
%A = create_ct_operator_2d_fan_astra(CtDataLimited, n, n);
A = create_ct_operator_2d_fan_astra_cuda(CtDataLimited, n, n);

% Create the simplest possible reconstruction: direct back-projection
sinogramLimited = CtDataLimited.sinogram(:);
reconLimitedBp  = A' * sinogramLimited;
reconLimitedBp  = reshape(reconLimitedBp, n, n);

% Compare reconstruction to test reconstruction and segmentation visually
figure('Name', 'Simplest possible limited angle reconstruction');
imshow(reconLimitedBp, []);

figure('Name', 'FBP reference reconstruction');
imshow(reconFullFbp, []);

figure('Name', 'FBP reference reconstruction, segmentation');
imshow(reconFullFbpSeg, []);


