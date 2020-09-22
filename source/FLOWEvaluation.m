% Measure flow accuracy

clc; clear all; close all;

dataset = 'TSS_CVPR2016';

resultsDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/results/' dataset];
%resultsDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/results/' dataset '/withDetectionRotationFixed'];

groundTruthFileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/gt.txt'];
groundTruthFid = fopen(groundTruthFileName, 'r');
groundTruthCell = textscan(groundTruthFid, '%s\n');
groundTruthCell = groundTruthCell{1};
fclose(groundTruthFid);

% pair1FileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/pair1.txt'];
% pair1Fid = fopen(pair1FileName, 'r');
% pair1Cell = textscan(pair1Fid, '%s\n');
% fclose(pair1Fid);
% 
% pair2FileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/pair2.txt'];
% pair2Fid = fopen(pair2FileName, 'r');
% pair2Cell = textscan(pair2Fid, '%s\n');
% fclose(pair2Fid);

groundTruthDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/GroundTruth/' dataset];

% imageDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/Images/' dataset];

FG3DCarLinear = [];
FG3DCarLinearBilateralFiltered = [];
FG3DCarDeformable = [];
FG3DCarDeformableBilateralFiltered = [];
FG3DCarTPS = [];
FG3DCarNatural = [];

JODSLinear = [];
JODSLinearBilateralFiltered = [];
JODSDeformable = [];
JODSDeformableBilateralFiltered = [];
JODSTPS = [];
JODSNatural = [];

PASCALLinear = [];
PASCALLinearBilateralFiltered = [];
PASCALDeformable = [];
PASCALDeformableBilateralFiltered = [];
PASCALTPS = [];
PASCALNatural = [];

threshold = 5;
maxSize = 100;

FG3DCarImageRange = 1:386;
JODSImageRange = 387:530;
PASCALImageRange = 531:682;

for i = 1:682
    
    if ( exist([resultsDirectory '/' num2str(i) '.mat']) )
        currentResult = load( [resultsDirectory '/' num2str(i) '.mat'] );
    else
        continue;
    end
                
    % load the ground truth flow file
    load( [groundTruthDirectory '/' groundTruthCell{i} '.mat'] );
    
    scaledThreshold = threshold / maxSize * max(size(gtFlow,1),size(gtFlow,2));
    
    % measure performance
    fvalidGT = gtFlow(:,:,1) < 1e9 & gtFlow(:,:,2) < 1e9;
    nvalidGT = sum(fvalidGT(:));
    
    
    flowLinearX = currentResult.flowEstimationStruct.flowLinearX;
    flowLinearY = currentResult.flowEstimationStruct.flowLinearY;
    
    flowLinearXBilateralFiltered = currentResult.flowEstimationStruct.flowLinearXBilateralFiltered;
    flowLinearYBilateralFiltered = currentResult.flowEstimationStruct.flowLinearYBilateralFiltered;
    
    flowDeformableX = currentResult.flowEstimationStruct.flowDeformableX;
    flowDeformableY = currentResult.flowEstimationStruct.flowDeformableY;
    
    flowDeformableXBilateralFiltered = currentResult.flowEstimationStruct.flowDeformableXBilateralFiltered;
    flowDeformableYBilateralFiltered = currentResult.flowEstimationStruct.flowDeformableYBilateralFiltered;
    
    flowTPSX = currentResult.flowEstimationStruct.flowTPSX;
    flowTPSY = currentResult.flowEstimationStruct.flowTPSY;
    
    flowNaturalX = currentResult.flowEstimationStruct.flowNaturalX;
    flowNaturalY = currentResult.flowEstimationStruct.flowNaturalY;
    
        
    ferrorLinear = sum((cat(3,flowLinearX, flowLinearY) - gtFlow).^2, 3).^0.5;
    faccsLinear = 1.0 - sum(ferrorLinear(fvalidGT) > scaledThreshold)/nvalidGT;
    
    ferrorLinearFiltered = sum((cat(3,flowLinearXBilateralFiltered, flowLinearYBilateralFiltered) - gtFlow).^2, 3).^0.5;
    faccsLinearFiltered = 1.0 - sum(ferrorLinearFiltered(fvalidGT) > scaledThreshold)/nvalidGT;
    
    ferrorDeformable = sum((cat(3,flowDeformableX, flowDeformableY) - gtFlow).^2, 3).^0.5;
    faccsDeformable = 1.0 - sum(ferrorDeformable(fvalidGT) > scaledThreshold)/nvalidGT;
    
    ferrorDeformableFiltered = sum((cat(3,flowDeformableXBilateralFiltered, flowDeformableYBilateralFiltered) - gtFlow).^2, 3).^0.5;
    faccsDeformableFiltered = 1.0 - sum(ferrorDeformableFiltered(fvalidGT) > scaledThreshold)/nvalidGT;
    
    ferrorTPS = sum((cat(3,flowTPSX, flowTPSY) - gtFlow).^2, 3).^0.5;
    faccsTPS = 1.0 - sum(ferrorTPS(fvalidGT) > scaledThreshold)/nvalidGT;
    
    ferrorNatural = sum((cat(3,flowNaturalX, flowNaturalY) - gtFlow).^2, 3).^0.5;
    faccsNatural = 1.0 - sum(ferrorNatural(fvalidGT) > scaledThreshold)/nvalidGT;
    
    if ( ismember(i, FG3DCarImageRange ) )
        
        FG3DCarLinear = [FG3DCarLinear; faccsLinear];
        FG3DCarLinearBilateralFiltered = [FG3DCarLinearBilateralFiltered; faccsLinearFiltered];
        FG3DCarDeformable = [FG3DCarDeformable; faccsDeformable];
        FG3DCarDeformableBilateralFiltered = [FG3DCarDeformableBilateralFiltered; faccsDeformableFiltered];
        FG3DCarTPS = [FG3DCarTPS; faccsTPS];
        FG3DCarNatural = [FG3DCarNatural; faccsNatural];
                
    elseif ( ismember(i, JODSImageRange ) )
        
        JODSLinear = [JODSLinear; faccsLinear];
        JODSLinearBilateralFiltered = [JODSLinearBilateralFiltered; faccsLinearFiltered];
        JODSDeformable = [JODSDeformable; faccsDeformable];
        JODSDeformableBilateralFiltered = [JODSDeformableBilateralFiltered; faccsDeformableFiltered];
        JODSTPS = [JODSTPS; faccsTPS];
        JODSNatural = [JODSNatural; faccsNatural];
        
    elseif ( ismember(i, PASCALImageRange ) )
        
        PASCALLinear = [PASCALLinear; faccsLinear];
        PASCALLinearBilateralFiltered = [PASCALLinearBilateralFiltered; faccsLinearFiltered];
        PASCALDeformable = [PASCALDeformable; faccsDeformable];
        PASCALDeformableBilateralFiltered = [PASCALDeformableBilateralFiltered; faccsDeformableFiltered];
        PASCALTPS = [PASCALTPS; faccsTPS];
        PASCALNatural = [PASCALNatural; faccsNatural];
        
    end
    
    disp([num2str(i) ' done...']);
    clear currentResult;
    clear gtFlow;
    
end
    
    
if( ~isempty(FG3DCarLinear) )   
    disp(['FG3DCarLinear: ' num2str(mean(FG3DCarLinear))]);    
end
if( ~isempty(FG3DCarLinearBilateralFiltered) )   
    disp(['FG3DCarLinearBilateralFiltered: ' num2str(mean(FG3DCarLinearBilateralFiltered))]);    
end
if( ~isempty(FG3DCarDeformable) )   
    disp(['FG3DCarDeformable: ' num2str(mean(FG3DCarDeformable))]);    
end
if( ~isempty(FG3DCarDeformableBilateralFiltered) )   
    disp(['FG3DCarDeformableBilateralFiltered: ' num2str(mean(FG3DCarDeformableBilateralFiltered))]);    
end
if( ~isempty(FG3DCarTPS) )   
    disp(['FG3DCarTPS: ' num2str(mean(FG3DCarTPS))]);    
end
if( ~isempty(FG3DCarNatural) )   
    disp(['FG3DCarNatural: ' num2str(mean(FG3DCarNatural))]);    
end

disp(' ');

if( ~isempty(JODSLinear) )   
    disp(['JODSLinear: ' num2str(mean(JODSLinear))]);    
end
if( ~isempty(JODSLinearBilateralFiltered) )   
    disp(['JODSLinearBilateralFiltered: ' num2str(mean(JODSLinearBilateralFiltered))]);    
end
if( ~isempty(JODSDeformable) )   
    disp(['JODSDeformable: ' num2str(mean(JODSDeformable))]);    
end
if( ~isempty(JODSDeformableBilateralFiltered) )   
    disp(['JODSDeformableBilateralFiltered: ' num2str(mean(JODSDeformableBilateralFiltered))]);    
end
if( ~isempty(JODSTPS) )   
    disp(['JODSTPS: ' num2str(mean(JODSTPS))]);    
end
if( ~isempty(JODSNatural) )   
    disp(['JODSNatural: ' num2str(mean(JODSNatural))]);    
end

disp(' ');

if( ~isempty(PASCALLinear) )   
    disp(['PASCALLinear: ' num2str(mean(PASCALLinear))]);    
end
if( ~isempty(PASCALLinearBilateralFiltered) )   
    disp(['PASCALLinearBilateralFiltered: ' num2str(mean(PASCALLinearBilateralFiltered))]);    
end
if( ~isempty(PASCALDeformable) )   
    disp(['PASCALDeformable: ' num2str(mean(PASCALDeformable))]);    
end
if( ~isempty(PASCALDeformableBilateralFiltered) )   
    disp(['PASCALDeformableBilateralFiltered: ' num2str(mean(PASCALDeformableBilateralFiltered))]);    
end
if( ~isempty(PASCALTPS) )   
    disp(['PASCALTPS: ' num2str(mean(PASCALTPS))]);    
end
if( ~isempty(PASCALNatural) )   
    disp(['PASCALNatural: ' num2str(mean(PASCALNatural))]);    
end
    
    
    
    
    
    
