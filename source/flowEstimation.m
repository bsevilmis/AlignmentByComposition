function output = flowEstimation(flowEstimationParameters, coarseSearchStruct, objectProposalStruct, optimizationStruct)

% --PARAMETERS

image1 = objectProposalStruct.image1;
image2 = objectProposalStruct.image2;
searchFlag = optimizationStruct.flag;

if (searchFlag == false)
    % do linear warping
    [XGridImage1, YGridImage1] = meshgrid(1:size(image1,2),1:size(image1,1));
    XDestination = (XGridImage1 - 1) * (size(image2,2)-1)/(size(image1,2)-1) + 1;
    YDestination = (YGridImage1 - 1) * (size(image2,1)-1)/(size(image1,1)-1) + 1;
    
    flowLinearXCompleted = (XDestination-XGridImage1);
    flowLinearYCompleted = (YDestination-YGridImage1);
    warpedLinearCompleted = uint8(warpImageModified(double(image2),flowLinearXCompleted, flowLinearYCompleted));
    
    flowLinearXCompletedBilateralFiltered = flowLinearXCompleted;
    flowLinearYCompletedBilateralFiltered = flowLinearYCompleted;
    warpedLinearCompletedFiltered = warpedLinearCompleted;
    
    flowDeformableXCompleted = flowLinearXCompleted;
    flowDeformableYCompleted = flowLinearYCompleted;
    warpedDeformableCompleted = warpedLinearCompleted;
    
    flowDeformableXCompletedBilateralFiltered = flowLinearXCompleted;
    flowDeformableYCompletedBilateralFiltered = flowLinearYCompleted;
    warpedDeformableCompletedFiltered = warpedLinearCompleted;
    
    flowTPSX = flowLinearXCompleted;
    flowTPSY = flowLinearYCompleted;
    warpedTPS = warpedLinearCompleted;
    
    flowNaturalX = flowLinearXCompleted;
    flowNaturalY = flowLinearYCompleted;
    warpedNatural = warpedLinearCompleted;
    
    
    output.warpedLinear = warpedLinearCompleted;
    output.flowLinearX = flowLinearXCompleted;
    output.flowLinearY = flowLinearYCompleted;
    
    output.warpedLinearBilateralFiltered = warpedLinearCompletedFiltered;
    output.flowLinearXBilateralFiltered = flowLinearXCompletedBilateralFiltered;
    output.flowLinearYBilateralFiltered = flowLinearYCompletedBilateralFiltered;
    
    output.warpedDeformable = warpedDeformableCompleted;
    output.flowDeformableX = flowDeformableXCompleted;
    output.flowDeformableY = flowDeformableYCompleted;
    
    output.warpedDeformableBilateralFiltered = warpedDeformableCompletedFiltered;
    output.flowDeformableXBilateralFiltered = flowDeformableXCompletedBilateralFiltered;
    output.flowDeformableYBilateralFiltered = flowDeformableYCompletedBilateralFiltered;
    
    output.warpedTPS = warpedTPS;
    output.flowTPSX = flowTPSX;
    output.flowTPSY = flowTPSY;
    
    output.warpedNatural = warpedNatural;
    output.flowNaturalX = flowNaturalX;
    output.flowNaturalY = flowNaturalY;
        
    return;

end


box1Features = objectProposalStruct.box1Features;
coarseSolutionProposalIds = optimizationStruct.coarseSolutionProposalIds;
bestMatchLabels = optimizationStruct.bestMatchLabels;
decisionMatchLabels = optimizationStruct.decisionMatchLabels;
R = coarseSearchStruct.R;
T = coarseSearchStruct.T;
wBilateral = flowEstimationParameters.wBilateral;
sigmaBilateral = flowEstimationParameters.sigmaBilateral;


% --CODE

% Linear Flow Field
flowFieldPartLabels = zeros(size(image1,1),size(image1,2));
for r = 1:size(image1,1)
    for c = 1:size(image1,2)
        
        currentPartLabel = NaN;
        currentPartScore = -Inf;
        
        for k = 1:numel(coarseSolutionProposalIds)
            
            currentPartCoordinates = box1Features{coarseSolutionProposalIds(k)}.coordinates;
            
            if ( r >= currentPartCoordinates(1) && c >= currentPartCoordinates(2) && ...
                    r <= currentPartCoordinates(3) && c <= currentPartCoordinates(4) )
                % we are inside the part
                %                 if ( bestMatchLabels{kkk}(decisionMatchLabels(kkk), end) > currentPartScore )
                %                     currentPartLabel = kkk;
                %                     currentPartScore = bestMatchLabels{kkk}(decisionMatchLabels(kkk), end);
                %                 end
                areaPart = ( currentPartCoordinates(3) - currentPartCoordinates(1) ) * ( currentPartCoordinates(4) - currentPartCoordinates(2) );
                
                if ( bestMatchLabels{k}(decisionMatchLabels(k), end) / areaPart > currentPartScore )
                    currentPartLabel = k;
                    currentPartScore = bestMatchLabels{k}(decisionMatchLabels(k), end) / areaPart;
                end                
            end                        
        end        
        flowFieldPartLabels(r,c) = currentPartLabel;                
    end
end

flowLinearX = NaN * ones(size(image1,1),size(image1,2));
flowLinearY = NaN * ones(size(image1,1),size(image1,2));

for r = 1:size(image1,1)
    for c = 1:size(image1,2)
        
        currentPartLabel = flowFieldPartLabels(r,c);
        
        if ( ~isnan(currentPartLabel) )
            
            currentPartCoordinates = box1Features{coarseSolutionProposalIds(currentPartLabel)}.coordinates;
            
            % estimated match location
            centerMatchedYCoordinate = bestMatchLabels{currentPartLabel}(decisionMatchLabels(currentPartLabel), 3);
            centerMatchedXCoordinate = bestMatchLabels{currentPartLabel}(decisionMatchLabels(currentPartLabel), 4);
                        
            part2Estimated = zeros(1,4);
            
            part2Estimated(2:-1:1) = (R * [currentPartCoordinates(2);currentPartCoordinates(1)] + T)';
            part2Estimated(4:-1:3) = (R * [currentPartCoordinates(4);currentPartCoordinates(3)] + T)';
            
            widthPart2 = part2Estimated(4) - part2Estimated(2);
            heightPart2 = part2Estimated(3) - part2Estimated(1);
            
            currentMatchedPartCoordinates = [centerMatchedYCoordinate - 0.5 * heightPart2, ...
                centerMatchedXCoordinate - 0.5 * widthPart2, ...
                centerMatchedYCoordinate + 0.5 * heightPart2, ...
                centerMatchedXCoordinate + 0.5 * widthPart2];
            
            term1 = (currentMatchedPartCoordinates(4) - currentMatchedPartCoordinates(2));
            term2 = (currentPartCoordinates(4) - currentPartCoordinates(2));
            term3 = (c - currentPartCoordinates(2));
            term4 = currentMatchedPartCoordinates(2);
            
            
            flowLinearX(r,c) = ( ( (term1 * term3) / term2 ) + term4 ) - c;
            
            
            term1 = (currentMatchedPartCoordinates(3) - currentMatchedPartCoordinates(1));
            term2 = (currentPartCoordinates(3) - currentPartCoordinates(1));
            term3 = (r - currentPartCoordinates(1));
            term4 = currentMatchedPartCoordinates(1);
            
            
            flowLinearY(r,c) = ( ( (term1 * term3) / term2 ) + term4 ) - r;
                        
        end        
    end
end
warpedLinear = uint8(warpImageModified(double(image2),flowLinearX, flowLinearY));

% Normalized cuts based interpolation
[flowLinearXCompleted, flowLinearYCompleted] = normalizedCutsFlowCompletion(image1, flowLinearX, flowLinearY);
warpedLinearCompleted = uint8(warpImageModified(double(image2),flowLinearXCompleted, flowLinearYCompleted));

% Bilateral filtering on linear interpolation
flowLinearXCompletedBilateralFiltered = bfilter2(flowLinearXCompleted, wBilateral, sigmaBilateral);
flowLinearYCompletedBilateralFiltered = bfilter2(flowLinearYCompleted, wBilateral, sigmaBilateral);
warpedLinearCompletedFiltered = uint8(warpImageModified(double(image2), flowLinearXCompletedBilateralFiltered, flowLinearYCompletedBilateralFiltered));


% obtain deformable flow field
% for every picked part proposal, lets find its deformation to the
% part it got matched with the highest score and keep {meshgrid, flowfield}
deformableMatchingCell = repmat({struct('meshgridX',[],'meshgridY',[],'flowx',[],'flowy',[])},numel(coarseSolutionProposalIds),1);

for k = 1:numel(coarseSolutionProposalIds)
    
    
    currentPartCoordinates = box1Features{coarseSolutionProposalIds(k)}.coordinates;
    
    % estimated match location
    centerMatchedYCoordinate = bestMatchLabels{k}(decisionMatchLabels(k), 3);
    centerMatchedXCoordinate = bestMatchLabels{k}(decisionMatchLabels(k), 4);
    
    
    part2Estimated = zeros(1,4);
    
    part2Estimated(2:-1:1) = (R * [currentPartCoordinates(2);currentPartCoordinates(1)] + T)';
    part2Estimated(4:-1:3) = (R * [currentPartCoordinates(4);currentPartCoordinates(3)] + T)';
    
    widthPart2 = part2Estimated(4) - part2Estimated(2);
    heightPart2 = part2Estimated(3) - part2Estimated(1);
    
    currentMatchedPartCoordinates = [centerMatchedYCoordinate - 0.5 * heightPart2, ...
        centerMatchedXCoordinate - 0.5 * widthPart2, ...
        centerMatchedYCoordinate + 0.5 * heightPart2, ...
        centerMatchedXCoordinate + 0.5 * widthPart2];
    
    currentMatchedPartCoordinates(1) = max(1, min(round(currentMatchedPartCoordinates(1)), size(image2,1)));
    currentMatchedPartCoordinates(2) = max(1, min(round(currentMatchedPartCoordinates(2)), size(image2,2)));
    currentMatchedPartCoordinates(3) = max(1, min(round(currentMatchedPartCoordinates(3)), size(image2,1)));
    currentMatchedPartCoordinates(4) = max(1, min(round(currentMatchedPartCoordinates(4)), size(image2,2)));
    
    
    [part1XGrid, part1YGrid] = meshgrid(currentPartCoordinates(2):currentPartCoordinates(4),...
        currentPartCoordinates(1):currentPartCoordinates(3));
    part1Patch = image1(currentPartCoordinates(1):currentPartCoordinates(3),currentPartCoordinates(2):currentPartCoordinates(4),:);
    part2Patch = image2(currentMatchedPartCoordinates(1):currentMatchedPartCoordinates(3),currentMatchedPartCoordinates(2):currentMatchedPartCoordinates(4),:);
    
        
    part1Feats = ExtractSIFT_WithPadding(part1Patch, [], 4);
    part2Feats = ExtractSIFT_WithPadding(part2Patch, [], 4);
    
    [vxFlow, vyFlow] = SimpleDSPMatch(part1Feats, part2Feats);
    
    vxFlow = vxFlow + (currentMatchedPartCoordinates(2) - currentPartCoordinates(2));
    vyFlow = vyFlow + (currentMatchedPartCoordinates(1) - currentPartCoordinates(1));
    
    deformableMatchingCell{k}.meshgridX = part1XGrid;
    deformableMatchingCell{k}.meshgridY = part1YGrid;
    deformableMatchingCell{k}.flowx = vxFlow;
    deformableMatchingCell{k}.flowy = vyFlow;
    
    disp(['Deformable matching: ' num2str(k) ' done...']);
    
        
end

% obtain deformable registration
flowDeformableX = NaN * ones(size(image1,1),size(image1,2));
flowDeformableY = NaN * ones(size(image1,1),size(image1,2));

for r = 1:size(image1,1)
    for c = 1:size(image1,2)
        
        currentPartLabel = flowFieldPartLabels(r,c);
        
        if ( ~isnan(currentPartLabel) )
                       
            [rowIndex, colIndex] = find( deformableMatchingCell{currentPartLabel}.meshgridX == c & ...
                deformableMatchingCell{currentPartLabel}.meshgridY == r);
            
            if ( ~(isempty(rowIndex) || isempty(colIndex)) )
                flowDeformableX(r,c) = deformableMatchingCell{currentPartLabel}.flowx(rowIndex, colIndex);
                flowDeformableY(r,c) = deformableMatchingCell{currentPartLabel}.flowy(rowIndex, colIndex);
            end
                                    
        end                
    end
end
warpedDeformable = uint8(warpImageModified(double(image2),flowDeformableX, flowDeformableY));


% Normalized cuts based interpolation
[flowDeformableXCompleted, flowDeformableYCompleted] = normalizedCutsFlowCompletion(image1, flowDeformableX, flowDeformableY);
warpedDeformableCompleted = uint8(warpImageModified(double(image2),flowDeformableXCompleted, flowDeformableYCompleted));

% Bilateral filtering on linear interpolation
flowDeformableXCompletedBilateralFiltered = bfilter2(flowDeformableXCompleted, wBilateral, sigmaBilateral);
flowDeformableYCompletedBilateralFiltered = bfilter2(flowDeformableYCompleted, wBilateral, sigmaBilateral);
warpedDeformableCompletedFiltered = uint8(warpImageModified(double(image2), flowDeformableXCompletedBilateralFiltered, flowDeformableYCompletedBilateralFiltered));


% obtain TPS deformation
TPSCenters = zeros(2,numel(decisionMatchLabels));
TPSValues = zeros(2,numel(decisionMatchLabels));

for k = 1:numel(decisionMatchLabels)   
    TPSCenters(:,k) = bestMatchLabels{k}(decisionMatchLabels(k),1:2)';
    TPSValues(:,k) = bestMatchLabels{k}(decisionMatchLabels(k),3:4)';
end

% lets remove duplicate points
trianglesTPS = delaunay(TPSCenters(2,:)', TPSCenters(1,:)');
validPointsTPS = unique(sort(trianglesTPS(:)));
TPSCenters = TPSCenters(:,validPointsTPS');
TPSValues = TPSValues(:,validPointsTPS');

[TPSInterpolant, pTPSInterpolant] = tpaps(TPSCenters, TPSValues);

[image1XGrid, image1YGrid] = meshgrid(1:size(image1,2),1:size(image1,1));
valsTPS = fnval(TPSInterpolant, [image1YGrid(:)'; image1XGrid(:)']);

flowTPSX = reshape(valsTPS(2,:)',size(image1,1),size(image1,2)) - image1XGrid;
flowTPSY = reshape(valsTPS(1,:)',size(image1,1),size(image1,2)) - image1YGrid;

warpedTPS = uint8(warpImageModified(double(image2), flowTPSX, flowTPSY));


% obtain natural neighbor interpolation
FXNaturalInterpolant = scatteredInterpolant(TPSCenters(2,:)',TPSCenters(1,:)',TPSValues(2,:)','natural');
FYNaturalInterpolant = scatteredInterpolant(TPSCenters(2,:)',TPSCenters(1,:)',TPSValues(1,:)','natural');

valsNatural = zeros(2,size(image1,1)*size(image1,2));
valsNatural(2,:) = FXNaturalInterpolant(image1XGrid(:),image1YGrid(:))';
valsNatural(1,:) = FYNaturalInterpolant(image1XGrid(:),image1YGrid(:))';

flowNaturalX = reshape(valsNatural(2,:)',size(image1,1),size(image1,2)) - image1XGrid;
flowNaturalY = reshape(valsNatural(1,:)',size(image1,1),size(image1,2)) - image1YGrid;

warpedNatural = uint8(warpImageModified(double(image2), flowNaturalX, flowNaturalY));


output.warpedLinear = warpedLinearCompleted;
output.flowLinearX = flowLinearXCompleted;
output.flowLinearY = flowLinearYCompleted;

output.warpedLinearBilateralFiltered = warpedLinearCompletedFiltered;
output.flowLinearXBilateralFiltered = flowLinearXCompletedBilateralFiltered;
output.flowLinearYBilateralFiltered = flowLinearYCompletedBilateralFiltered;

output.warpedDeformable = warpedDeformableCompleted;
output.flowDeformableX = flowDeformableXCompleted;
output.flowDeformableY = flowDeformableYCompleted;

output.warpedDeformableBilateralFiltered = warpedDeformableCompletedFiltered;
output.flowDeformableXBilateralFiltered = flowDeformableXCompletedBilateralFiltered;
output.flowDeformableYBilateralFiltered = flowDeformableYCompletedBilateralFiltered;

output.warpedTPS = warpedTPS;
output.flowTPSX = flowTPSX;
output.flowTPSY = flowTPSY;

output.warpedNatural = warpedNatural;
output.flowNaturalX = flowNaturalX;
output.flowNaturalY = flowNaturalY;


end


function [flowLinearXCompleted, flowLinearYCompleted] = normalizedCutsFlowCompletion(image1, flowLinearX, flowLinearY)

image1NTSC(:,:,1) = rgb2gray(image1);
image1NTSC(:,:,2) = rgb2gray(image1);
image1NTSC(:,:,3) = rgb2gray(image1);
image1NTSC = rgb2ntsc(double(image1NTSC/255));

flowMask = ( ~isnan(flowLinearX) ) & ( ~isnan(flowLinearY) );

% n : rowSizeNTSC
% m : colSizeNTSC
% imgSize

rowSizeNTSC = size(image1NTSC,1);
colSizeNTSC = size(image1NTSC,2);
imgSize=rowSizeNTSC * colSizeNTSC;

indsM=reshape([1:imgSize],rowSizeNTSC,colSizeNTSC);
wd=1;

len=0;
consts_len=0;
col_inds=zeros(imgSize*(2*wd+1)^2,1);
row_inds=zeros(imgSize*(2*wd+1)^2,1);
vals=zeros(imgSize*(2*wd+1)^2,1);
gvals=zeros(1,(2*wd+1)^2);


for jjjj=1:colSizeNTSC
    for iiii=1:rowSizeNTSC
        consts_len=consts_len+1;
        
        if (~flowMask(iiii,jjjj))
            tlen=0;
            for ii=max(1,iiii-wd):min(iiii+wd,rowSizeNTSC)
                for jj=max(1,jjjj-wd):min(jjjj+wd,colSizeNTSC)
                    
                    if (ii~=iiii)|(jj~=jjjj)
                        len=len+1; tlen=tlen+1;
                        row_inds(len)= consts_len;
                        col_inds(len)=indsM(ii,jj);
                        gvals(tlen)=image1NTSC(ii,jj,1);
                    end
                end
            end
            t_val=image1NTSC(iiii,jjjj,1);
            gvals(tlen+1)=t_val;
            c_var=mean((gvals(1:tlen+1)-mean(gvals(1:tlen+1))).^2);
            csig=c_var*0.6;
            mgv=min((gvals(1:tlen)-t_val).^2);
            if (csig<(-mgv/log(0.01)))
                csig=-mgv/log(0.01);
            end
            if (csig<0.000002)
                csig=0.000002;
            end
            
            gvals(1:tlen)=exp(-(gvals(1:tlen)-t_val).^2/csig);
            gvals(1:tlen)=gvals(1:tlen)/sum(gvals(1:tlen));
            vals(len-tlen+1:len)=-gvals(1:tlen);
        end
        
        
        len=len+1;
        row_inds(len)= consts_len;
        col_inds(len)=indsM(iiii,jjjj);
        vals(len)=1;
        
    end
end


vals=vals(1:len);
col_inds=col_inds(1:len);
row_inds=row_inds(1:len);


AAA=sparse(row_inds,col_inds,vals,consts_len,imgSize);
bbb=zeros(size(AAA,1),1);


validFlowXInds = find( ~isnan(flowLinearX) );
validFlowYInds = find( ~isnan(flowLinearY) );

bbb( validFlowXInds ) = flowLinearX( validFlowXInds );
flowLinearXCompleted = reshape( AAA\bbb, rowSizeNTSC, colSizeNTSC, 1);

bbb( validFlowYInds ) = flowLinearY( validFlowYInds );
flowLinearYCompleted = reshape( AAA\bbb, rowSizeNTSC, colSizeNTSC, 1);


end



