function output = colocalizeCommonObject(objectProposalStruct)

% --PARAMETER EXTRACTION

box1Features = objectProposalStruct.box1Features;
box2Features = objectProposalStruct.box2Features;


% --CODE

hogSimilarityMatrix = computeHOGSimilarityMatrix(box1Features, box2Features);
hogSimilarityMatrix = (hogSimilarityMatrix - min(min(hogSimilarityMatrix))) / ( max(max(hogSimilarityMatrix)) - min(min(hogSimilarityMatrix)) );

%similarityMatrix = computeSimilarityMatrix(box1Features, box2Features, hogSimilarityMatrix);
similarityMatrix = computeSimilarityMatrixSymmetricVersion(box1Features, box2Features, hogSimilarityMatrix);

% VOTING SCHEME
% convert to [x1 y1 width height] format
proposals1Converted = zeros(size(box1Features,1),4);
for r = 1:size(box1Features,1)
   proposals1Converted(r,:) = [box1Features{r}.coordinates(2),...
       box1Features{r}.coordinates(1),...
       box1Features{r}.coordinates(4)-box1Features{r}.coordinates(2) + 1,...
       box1Features{r}.coordinates(3)-box1Features{r}.coordinates(1) + 1]; 
end

proposals2Converted = zeros(size(box2Features,1),4);
for r = 1:size(box2Features,1)
   proposals2Converted(r,:) = [box2Features{r}.coordinates(2),...
       box2Features{r}.coordinates(1),...
       box2Features{r}.coordinates(4)-box2Features{r}.coordinates(2) + 1,...
       box2Features{r}.coordinates(3)-box2Features{r}.coordinates(1) + 1]; 
end

overlapMatrix1 = zeros(size(proposals1Converted,1),size(proposals1Converted,1));
for rrr = 1:size(proposals1Converted,1)
    
    intersectionScores = rectint(proposals1Converted(rrr,:), proposals1Converted);
    validIndices = find(intersectionScores > 0);
    
    for ccc = validIndices
        
        if ( rrr == ccc || overlapMatrix1(rrr,ccc) > 0 )
            continue;
        else
            currentIntersectionScore = intersectionScores(ccc);
            currentUnionScore = ( proposals1Converted(rrr,3) * proposals1Converted(rrr,4) )  + ( proposals1Converted(ccc,3) * proposals1Converted(ccc,4) )...
                - currentIntersectionScore;
            overlapMatrix1(rrr,ccc) = max(0, currentIntersectionScore / currentUnionScore);
            overlapMatrix1(ccc,rrr) = max(0, currentIntersectionScore / currentUnionScore);
        end
        
    end        
end
overlapMatrix1(sub2ind([size(overlapMatrix1,1),size(overlapMatrix1,1)],(1:size(overlapMatrix1,1))',(1:size(overlapMatrix1,1))')) = 1;


overlapMatrix2 = zeros(size(proposals2Converted,1),size(proposals2Converted,1));
for rrr = 1:size(proposals2Converted,1)
    
    intersectionScores = rectint(proposals2Converted(rrr,:), proposals2Converted);
    validIndices = find(intersectionScores > 0);
    
    for ccc = validIndices
        
        if ( rrr == ccc || overlapMatrix2(rrr,ccc) > 0 )
            continue;
        else
            currentIntersectionScore = intersectionScores(ccc);
            currentUnionScore = ( proposals2Converted(rrr,3) * proposals2Converted(rrr,4) )  + ( proposals2Converted(ccc,3) * proposals2Converted(ccc,4) )...
                - currentIntersectionScore;
            overlapMatrix2(rrr,ccc) = max(0, currentIntersectionScore / currentUnionScore);
            overlapMatrix2(ccc,rrr) = max(0, currentIntersectionScore / currentUnionScore);
        end
        
    end        
end
overlapMatrix2(sub2ind([size(overlapMatrix2,1),size(overlapMatrix2,1)],(1:size(overlapMatrix2,1))',(1:size(overlapMatrix2,1))')) = 1;

% FAST METHOD
overlapMatrix1Copy = (overlapMatrix1 >= 0 ) .* overlapMatrix1;
overlapMatrix2Copy = (overlapMatrix2 >= 0 ) .* overlapMatrix2;

tic
A = overlapMatrix1Copy*similarityMatrix;
ASum = sum(overlapMatrix1Copy,2);
B = overlapMatrix2Copy*similarityMatrix';
BSum = sum(overlapMatrix2Copy,2);
similarityMatrixModified = (A + B') ./ ( repmat(ASum, 1,size(similarityMatrix,2)) + ...
    repmat(BSum', size(similarityMatrix,1), 1) ); 
toc

[bestId1, bestId2] = find( similarityMatrixModified == max(max(similarityMatrixModified)) );

if ( isempty(bestId1) || isempty(bestId2) )
    bestId1 = 1;
    bestId2 = 1;
else
    bestId1 = bestId1(1); % avoid multiple solutions
    bestId2 = bestId2(1);
end

%% PICK THE IMAGE ITSELF AS THE BOUNDING BOX
%bestId1 = 1;
%bestId2 = 1;


output.bestId1 = bestId1;
output.bestId2 = bestId2;


end
