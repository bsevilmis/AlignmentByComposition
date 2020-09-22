% Measure PCK score

clc;clear all;close all;

dataset = 'PF-dataset';

resultsDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/results/' dataset '/withDetectionRotationFixed'];

groundTruthFileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/gt.txt'];
groundTruthFid = fopen(groundTruthFileName, 'r');
groundTruthCell = textscan(groundTruthFid, '%s %s\n');
fclose(groundTruthFid);

classNamesFileName = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/ImageFileNames/' dataset '/classNames.txt'];
classNamesFid = fopen(classNamesFileName, 'r');
classNamesCell = textscan(classNamesFid,'%s\n');
fclose(classNamesFid);

groundTruthKeypointsDirectory = ['/users/bsevilmi/scratch/AlignmentByComposition/datasets/GroundTruth/' dataset];

PCKScoresLinear = [];
PCKScoresLinearBilateralFiltered = [];
PCKScoresDeformable = [];
PCKScoresDeformableBilateralFiltered = [];
PCKScoresTPS = [];
PCKScoresNatural = [];

alpha = 0.1;

for i = 1:900
           
    %if ( ~exist([resultsDirectory '/' num2str(i) '.mat'] ) )
    %    continue;
    %end
    
    currentResult = load( [resultsDirectory '/' num2str(i) '.mat'] );
           
    keypoints1 = load( [groundTruthKeypointsDirectory '/' groundTruthCell{1}{i} '.mat'] );
    keypoints1 = keypoints1.pts_coord;
        
    keypoints2 = load( [groundTruthKeypointsDirectory '/' groundTruthCell{2}{i} '.mat'] );
    keypoints2 = keypoints2.pts_coord;
    
    assert( numel(keypoints1) == numel(keypoints2) );
    
    distancesLinear = zeros(1,size(keypoints1,2));
    distancesLinearBilateralFiltered = zeros(1,size(keypoints1,2));
    distancesDeformable = zeros(1,size(keypoints1,2));
    distancesDeformableBilateralFiltered = zeros(1,size(keypoints1,2));
    distancesTPS = zeros(1,size(keypoints1,2));
    distancesNatural = zeros(1,size(keypoints1,2));
    
    bbox = [min(keypoints2(1,:)) min(keypoints2(2,:)) max(keypoints2(1,:)) max(keypoints2(2,:))];
    
    for j = 1:size(keypoints1,2)
       
        px = round(keypoints1(1,j));
        py = round(keypoints1(2,j));


        if ( isnan(px) || isnan(py) || isnan(keypoints2(1,j)) || isnan(keypoints2(2,j)) )

           disp('Warning point is NaN');

        end


        
        distancesLinear(j) = sqrt( ( (keypoints1(1,j) + currentResult.flowEstimationStruct.flowLinearX(py,px)) - keypoints2(1,j) )^2 + ...
            ( (keypoints1(2,j) + currentResult.flowEstimationStruct.flowLinearY(py,px)) - keypoints2(2,j) )^2 );                
        
        distancesLinearBilateralFiltered(j) = sqrt( ( (keypoints1(1,j) + currentResult.flowEstimationStruct.flowLinearXBilateralFiltered(py,px)) - keypoints2(1,j) )^2 + ...
            ( (keypoints1(2,j) + currentResult.flowEstimationStruct.flowLinearYBilateralFiltered(py,px)) - keypoints2(2,j) )^2 );                
        
        distancesDeformable(j) = sqrt( ( (keypoints1(1,j) + currentResult.flowEstimationStruct.flowDeformableX(py,px)) - keypoints2(1,j) )^2 + ...
            ( (keypoints1(2,j) + currentResult.flowEstimationStruct.flowDeformableY(py,px)) - keypoints2(2,j) )^2 );                
        
        distancesDeformableBilateralFiltered(j) = sqrt( ( (keypoints1(1,j) + currentResult.flowEstimationStruct.flowDeformableXBilateralFiltered(py,px)) - keypoints2(1,j) )^2 + ...
            ( (keypoints1(2,j) + currentResult.flowEstimationStruct.flowDeformableYBilateralFiltered(py,px)) - keypoints2(2,j) )^2 );                
        
        distancesTPS(j) = sqrt( ( (keypoints1(1,j) + currentResult.flowEstimationStruct.flowTPSX(py,px)) - keypoints2(1,j) )^2 + ...
            ( (keypoints1(2,j) + currentResult.flowEstimationStruct.flowTPSY(py,px)) - keypoints2(2,j) )^2 );                
        
        distancesNatural(j) = sqrt( ( (keypoints1(1,j) + currentResult.flowEstimationStruct.flowNaturalX(py,px)) - keypoints2(1,j) )^2 + ...
            ( (keypoints1(2,j) + currentResult.flowEstimationStruct.flowNaturalY(py,px)) - keypoints2(2,j) )^2 );                
        
    end
    
    distancesLinear = distancesLinear ./ max( bbox(3:4) - bbox(1:2) );
    distancesLinearBilateralFiltered = distancesLinearBilateralFiltered ./ max( bbox(3:4) - bbox(1:2) );
    distancesDeformable = distancesDeformable ./ max( bbox(3:4) - bbox(1:2) );
    distancesDeformableBilateralFiltered = distancesDeformableBilateralFiltered ./ max( bbox(3:4) - bbox(1:2) );
    distancesTPS = distancesTPS ./ max( bbox(3:4) - bbox(1:2) );
    distancesNatural = distancesNatural ./ max( bbox(3:4) - bbox(1:2) );
    
            
    PCKScoresLinear = [PCKScoresLinear; nnz( distancesLinear <= alpha )];
    PCKScoresLinearBilateralFiltered = [PCKScoresLinearBilateralFiltered; nnz( distancesLinearBilateralFiltered <= alpha )];
    PCKScoresDeformable = [PCKScoresDeformable; nnz( distancesDeformable <= alpha )];
    PCKScoresDeformableBilateralFiltered = [PCKScoresDeformableBilateralFiltered; nnz( distancesDeformableBilateralFiltered <= alpha )];
    PCKScoresTPS = [PCKScoresTPS; nnz( distancesTPS <= alpha )];
    PCKScoresNatural = [PCKScoresNatural; nnz( distancesNatural <= alpha )];
    
    
    disp([num2str(i) ' done...']);

            
end

disp([dataset ' mean PCK(Linear): ' num2str(mean(PCKScoresLinear/10))]);
disp([dataset ' mean PCK(LinearBilateralFiltered): ' num2str(mean(PCKScoresLinearBilateralFiltered/10))]);
disp([dataset ' mean PCK(Deformable): ' num2str(mean(PCKScoresDeformable/10))]);
disp([dataset ' mean PCK(DeformableBilateralFiltered): ' num2str(mean(PCKScoresDeformableBilateralFiltered/10))]);
disp([dataset ' mean PCK(TPS): ' num2str(mean(PCKScoresTPS/10))]);
disp([dataset ' mean PCK(Natural): ' num2str(mean(PCKScoresNatural/10))]);

for i = 1:90:900
 
    disp(' ');
    disp(classNamesCell{1}{i});
    
    disp([dataset ' mean PCK(Linear): ' num2str(mean(PCKScoresLinear(i:i+89)/10))]);
    disp([dataset ' mean PCK(LinearBilateralFiltered): ' num2str(mean(PCKScoresLinearBilateralFiltered(i:i+89)/10))]);
    disp([dataset ' mean PCK(Deformable): ' num2str(mean(PCKScoresDeformable(i:i+89)/10))]);
    disp([dataset ' mean PCK(DeformableBilateralFiltered): ' num2str(mean(PCKScoresDeformableBilateralFiltered(i:i+89)/10))]);
    disp([dataset ' mean PCK(TPS): ' num2str(mean(PCKScoresTPS(i:i+89)/10))]);
    disp([dataset ' mean PCK(Natural): ' num2str(mean(PCKScoresNatural(i:i+89)/10))]);
            
end
