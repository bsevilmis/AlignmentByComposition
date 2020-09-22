% Measure PCK score

clc;clear all;close all;

dataset = 'PF-dataset-PASCAL';

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

classIds = [];


alpha = 0.1;

for i = 1:1351
           
    %if ( ~exist([resultsDirectory '/' num2str(i) '.mat'] ) )
    %    continue;
    %end
    
    currentResult = load( [resultsDirectory '/' num2str(i) '.mat'] );
           
    keypoints1 = load( [groundTruthKeypointsDirectory '/' groundTruthCell{1}{i} '.mat'] );
    classIds1 = keypoints1.class;
    keypoints1 = keypoints1.kps;
        
    keypoints2 = load( [groundTruthKeypointsDirectory '/' groundTruthCell{2}{i} '.mat'] );
    bbox = keypoints2.bbox;
    classIds2 = keypoints2.class;
    keypoints2 = keypoints2.kps;
    
    if ( ~isequal(classIds1, classIds2) )
	continue;
    end

    assert( numel(keypoints1) == numel(keypoints2) );
    
    classIds = [classIds; classIds1];
    
    distancesLinear = [];
    distancesLinearBilateralFiltered = [];
    distancesDeformable = [];
    distancesDeformableBilateralFiltered = [];
    distancesTPS = [];
    distancesNatural = [];
            
    for j = 1:size(keypoints1,1)
       
        px = round(keypoints1(j,1));
        py = round(keypoints1(j,2));
        

        if ( isnan(px) || isnan(py) || isnan(keypoints2(j,1)) || isnan(keypoints2(j,2)) )

           continue;

	else

        	distancesLinear = [distancesLinear; sqrt( ( (keypoints1(j,1) + currentResult.flowEstimationStruct.flowLinearX(py,px)) - keypoints2(j,1) )^2 + ...
            ( (keypoints1(j,2) + currentResult.flowEstimationStruct.flowLinearY(py,px)) - keypoints2(j,2) )^2 ) ];                
        
        	distancesLinearBilateralFiltered = [distancesLinearBilateralFiltered; sqrt( ( (keypoints1(j,1) + currentResult.flowEstimationStruct.flowLinearXBilateralFiltered(py,px)) - keypoints2(j,1) )^2 + ...
            ( (keypoints1(j,2) + currentResult.flowEstimationStruct.flowLinearYBilateralFiltered(py,px)) - keypoints2(j,2) )^2 ) ];                
        
        	distancesDeformable = [distancesDeformable; sqrt( ( (keypoints1(j,1) + currentResult.flowEstimationStruct.flowDeformableX(py,px)) - keypoints2(j,1) )^2 + ...
            ( (keypoints1(j,2) + currentResult.flowEstimationStruct.flowDeformableY(py,px)) - keypoints2(j,2) )^2 ) ];                
        
        	distancesDeformableBilateralFiltered = [distancesDeformableBilateralFiltered; sqrt( ( (keypoints1(j,1) + currentResult.flowEstimationStruct.flowDeformableXBilateralFiltered(py,px)) - keypoints2(j,1) )^2 + ...
            ( (keypoints1(j,2) + currentResult.flowEstimationStruct.flowDeformableYBilateralFiltered(py,px)) - keypoints2(j,2) )^2 ) ];                
        
        	distancesTPS = [distancesTPS; sqrt( ( (keypoints1(j,1) + currentResult.flowEstimationStruct.flowTPSX(py,px)) - keypoints2(j,1) )^2 + ...
            ( (keypoints1(j,2) + currentResult.flowEstimationStruct.flowTPSY(py,px)) - keypoints2(j,2) )^2 ) ];                
        
        	distancesNatural = [distancesNatural; sqrt( ( (keypoints1(j,1) + currentResult.flowEstimationStruct.flowNaturalX(py,px)) - keypoints2(j,1) )^2 + ...
            ( (keypoints1(j,2) + currentResult.flowEstimationStruct.flowNaturalY(py,px)) - keypoints2(j,2) )^2 ) ];                

	end
        
    end
    
    distancesLinear = distancesLinear ./ max( bbox(3:4) - bbox(1:2) );
    distancesLinearBilateralFiltered = distancesLinearBilateralFiltered ./ max( bbox(3:4) - bbox(1:2) );
    distancesDeformable = distancesDeformable ./ max( bbox(3:4) - bbox(1:2) );
    distancesDeformableBilateralFiltered = distancesDeformableBilateralFiltered ./ max( bbox(3:4) - bbox(1:2) );
    distancesTPS = distancesTPS ./ max( bbox(3:4) - bbox(1:2) );
    distancesNatural = distancesNatural ./ max( bbox(3:4) - bbox(1:2) );
    
            
    PCKScoresLinear = [PCKScoresLinear; nnz( distancesLinear <= alpha )/numel(distancesLinear)];
    PCKScoresLinearBilateralFiltered = [PCKScoresLinearBilateralFiltered; nnz( distancesLinearBilateralFiltered <= alpha )/numel(distancesLinearBilateralFiltered)];
    PCKScoresDeformable = [PCKScoresDeformable; nnz( distancesDeformable <= alpha )/numel(distancesDeformable)];
    PCKScoresDeformableBilateralFiltered = [PCKScoresDeformableBilateralFiltered; nnz( distancesDeformableBilateralFiltered <= alpha )/numel(distancesDeformableBilateralFiltered)];
    PCKScoresTPS = [PCKScoresTPS; nnz( distancesTPS <= alpha )/numel(distancesTPS)];
    PCKScoresNatural = [PCKScoresNatural; nnz( distancesNatural <= alpha )/numel(distancesNatural)];
    
    
    disp([num2str(i) ' done...']);

            
end


for i = 1:20
 
    disp(' ');
    disp(classNamesCell{1}{i});
    
    disp([dataset ' mean PCK(Linear): ' num2str(mean(PCKScoresLinear(classIds == i)))]);
    disp([dataset ' mean PCK(LinearBilateralFiltered): ' num2str(mean(PCKScoresLinearBilateralFiltered(classIds == i)))]);
    disp([dataset ' mean PCK(Deformable): ' num2str(mean(PCKScoresDeformable(classIds == i)))]);
    disp([dataset ' mean PCK(DeformableBilateralFiltered): ' num2str(mean(PCKScoresDeformableBilateralFiltered(classIds == i)))]);
    disp([dataset ' mean PCK(TPS): ' num2str(mean(PCKScoresTPS(classIds == i)))]);
    disp([dataset ' mean PCK(Natural): ' num2str(mean(PCKScoresNatural(classIds == i)))]);
            
end
