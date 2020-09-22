function main(testId)

% Clear the workspace
clc;
close all;
addpath('./source');

% Create a matlabpool of size 20
matlabPoolSize = 20;

PC = parcluster('local');
PC.JobStorageLocation = ['/users/bsevilmi/scratch/AlignmentByComposition/' num2str(testId)];
matlabpool(PC, matlabPoolSize);
pctRunOnAll warning off

% Auxiliaries
auxiliariesStruct = setAuxiliaryFilePaths();

if ( exist([auxiliariesStruct.resultsFolder '/' num2str(testId) '.mat']) ~= 0 )
   matlabpool('close');
   return;
end

% Extract Object Proposals
disp(' '); disp('***OBJECT PROPOSAL EXTRACTION***');
objectProposalParameters = [];
objectProposalParameters.testId = testId;
objectProposalParameters.areaRatio = 0.5; %0.5, use 1 when treating the full image as the bounding box
objectProposalParameters.inclusionRatio = 0.8;
objectProposalStruct = extractObjectProposals(objectProposalParameters, auxiliariesStruct);

% Colocalization
disp(' '); disp('***COLOCALIZATION***');
colocalizationStruct = colocalizeCommonObject(objectProposalStruct);

% Coarse sliding window search
disp(' '); disp('***COARSE SEARCH***');
coarseSearchParameters = [];
coarseSearchParameters.stride = 8;
coarseSearchStruct = coarseSearch(coarseSearchParameters, objectProposalStruct, colocalizationStruct);

% Optimization
disp(' ');disp('***OPTIMIZATION***');
optimizationParameters = [];
optimizationParameters.bruteForceSearchSize = 50;
optimizationParameters.kBestMatches = 20; % 5
optimizationParameters.beamSize = 4; % 32
optimizationParameters.sigma = 1;
optimizationParameters.numberOfBestSolutions = 1;
optimizationParameters.rotationThreshold = 5; % degrees
optimizationParameters.distortionThreshold = 1;
optimizationParameters.fineSearchSigma = 5;
optimizationParameters.fineSearchMaxNumLabels = 10;
optimizationStruct = optimization(optimizationParameters, coarseSearchParameters, coarseSearchStruct, objectProposalStruct, colocalizationStruct);  

% Flow estimation
disp(' ');disp('***FLOW ESTIMATION***');
flowEstimationParameters = [];
flowEstimationParameters.wBilateral = 10;
flowEstimationParameters.sigmaBilateral = [50 30];
flowEstimationStruct = flowEstimation(flowEstimationParameters, coarseSearchStruct, objectProposalStruct, optimizationStruct);

% Save results
save([auxiliariesStruct.resultsFolder '/' num2str(objectProposalParameters.testId) '.mat'], 'flowEstimationStruct','colocalizationStruct');

matlabpool('close');

end
