#include "mex.h"
#include <iostream>
#include <queue>
#include <vector>
#include <random>
#include <stdlib.h>     /* srand, rand */
#include <math.h>
#include <cfloat>
#include <set>
#include <chrono>
#include <omp.h>
#include "string.h"


/* NOTE: compile with mex -v -largeArrayDims CDEBUGFLAGS=" " COPTIMFLAGS="-O3" CXXDEBUGFLAGS=" " LDDEBUGFLAGS=" " LDCXXDEBUGFLAGS=" " LDCXXOPTIMFLAGS="-O3" LDOPTIMFLAGS="-O3" CXXOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -std=c++11 -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" ./computeSimilarityMatrix.cpp in MATLAB */

using namespace std;
using namespace std::chrono;


double maxOuterContrast(const double* hogSimilarityMatrixPointer, unsigned int totalNumberOfBoxes1, unsigned int totalNumberOfBoxes2, const mxArray* largerBoxIds1,
        vector<double>& maxMatchScores)
{
    vector<double> maxScores;
    
    const mwSize* largerBoxIds1Dimensions = mxGetDimensions(largerBoxIds1);
    unsigned numberOfBoxesToCheck = (unsigned int)largerBoxIds1Dimensions[0];
    double* largerBoxIds1Pointer = (double*)mxGetData(largerBoxIds1);
    
    for (unsigned int boxIterator1 = 0; boxIterator1 < numberOfBoxesToCheck; boxIterator1++)
    {
//         double currentMaxScore = 0.0;
//         for (unsigned int boxId2 = 0; boxId2 < totalNumberOfBoxes2; boxId2++)
//         {
//             unsigned int boxIdToCheck = (unsigned int)(largerBoxIds1Pointer[boxIterator1] - 1); // MATLAB to C++ conversion
//             double currentScore = hogSimilarityMatrixPointer[boxId2 * totalNumberOfBoxes1 + boxIdToCheck];
//             if ( currentScore > currentMaxScore )
//             {
//                 currentMaxScore = currentScore; 
//             }
//             
//         }
        
        unsigned int boxIdToCheck = (unsigned int)(largerBoxIds1Pointer[boxIterator1] - 1); // MATLAB to C++ conversion
        double currentMaxScore = maxMatchScores.at(boxIdToCheck);
        
        maxScores.push_back(currentMaxScore);
        
    }
    
    // return the max of max
    double globalMaxScore = 0.0;
    for (unsigned int i = 0; i < maxScores.size(); i++)
    {
        double currentMaxScore = maxScores.at(i);
        if (currentMaxScore > globalMaxScore)
        {
            globalMaxScore = currentMaxScore;
        }
    }
    
    return globalMaxScore;
        
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    
    // data order:
    // box1Features [0]
    // box2Features [1]
    // hogSimilarityMatrix [2]
    // spmSimilarityMatrix [3]
    
    // get total number of cells for box1Features
    mwSize totalNumberOfBoxes1;
    totalNumberOfBoxes1 = mxGetNumberOfElements(prhs[0]);
    cout << "totalNumberOfBoxes1: " << totalNumberOfBoxes1 << endl;
    
    // get total number of cells for box2Features
    mwSize totalNumberOfBoxes2;
    totalNumberOfBoxes2 = mxGetNumberOfElements(prhs[1]);
    cout << "totalNumberOfBoxes2: " << totalNumberOfBoxes2 << endl;
    
    // get hogSimilarityMatrixPointer
    double* hogSimilarityMatrixPointer = (double*)mxGetData(prhs[2]);
    
//     // get spmSimilarityMatrixPointer
//     double* spmSimilarityMatrixPointer = (double*)mxGetData(prhs[3]);
        
    // store similarity matrix
    vector< vector<double> > similarityMatrix(totalNumberOfBoxes1, vector<double>(totalNumberOfBoxes2, 0.0));
    
    // get field ids
    
    // saliency
    // detectionClass
    // detectionClassJaccardIndex
    // SPM
    // largerBoxIds
    

    const mxArray* firstBoxPointer = mxGetCell(prhs[0], 0);
    unsigned int numberOfFields = mxGetNumberOfFields(firstBoxPointer);
    cout << "numberOfFields: " << numberOfFields << endl;
    const char* fieldName;
    string fieldNameString;
    
    unsigned int saliencyFieldId;
    unsigned int detectionClassFieldId;
    unsigned int detectionClassJaccardIndexFieldId;
//     unsigned int SPMFieldId;
    unsigned int largerBoxIdsFieldId;
//     unsigned int skipSPMFieldId;
    

    for (unsigned int i = 0; i < numberOfFields; i++)
    {
        fieldName = mxGetFieldNameByNumber(firstBoxPointer, i);
        fieldNameString = string(fieldName);
        
        if (fieldNameString.compare("saliency") == 0)
        {
            saliencyFieldId = i;            
        }
        else if (fieldNameString.compare("detectionClass") == 0)
        {
            detectionClassFieldId = i;
        }
        else if (fieldNameString.compare("detectionClassJaccardIndex") == 0)
        {
            detectionClassJaccardIndexFieldId = i;
        }
//         else if (fieldNameString.compare("SPM") == 0)
//         {
//             SPMFieldId = i;
//         }
        else if (fieldNameString.compare("largerBoxIds") == 0)
        {
            largerBoxIdsFieldId = i;
        }
//         else if (fieldNameString.compare("skipSPM") == 0)
//         {
//             skipSPMFieldId = i;
//         }
    }
    cout << "saliencyFieldId: " << saliencyFieldId << endl;
    cout << "detectionClassFieldId: " << detectionClassFieldId << endl;
    cout << "detectionClassJaccardIndexFieldId: " << detectionClassJaccardIndexFieldId << endl;
//     cout << "SPMFieldId: " << SPMFieldId << endl;
    cout << "largerBoxIdsFieldId: " << largerBoxIdsFieldId << endl;
//     cout << "skipSPMFieldId: " << skipSPMFieldId << endl;
    
    // first find maximum HOG match response for every box in image 1
    vector<double> maxHOGMatchScores(totalNumberOfBoxes1, 0.0);    
    
    unsigned int box1Id;
    unsigned int box2Id;
    double currentScore;
    
    //omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());
        
    #pragma omp parallel for default(none) \
        shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, maxHOGMatchScores) \
                private(box1Id, box2Id, currentScore)
    for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
    {
        for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
        {
            currentScore = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id];
            if( currentScore > maxHOGMatchScores.at(box1Id) )
            {
                maxHOGMatchScores.at(box1Id) = currentScore; 
            }            
        }        
    }
    
//     // first find maximum SPM match response for every box in image 1
//     vector<double> maxSPMMatchScores(totalNumberOfBoxes1, 0.0);
//     
//     omp_set_num_threads(4);
//     
//     #pragma omp parallel for default(none) \
//         shared(totalNumberOfBoxes1, totalNumberOfBoxes2, spmSimilarityMatrixPointer, maxSPMMatchScores) \
//                 private(box1Id, box2Id, currentScore)
//     for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
//     {
//         for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
//         {
//             currentScore = spmSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id];
//             if( currentScore > maxSPMMatchScores.at(box1Id) )
//             {
//                 maxSPMMatchScores.at(box1Id) = currentScore;
//             }
//         }
//     }
            
//     // compute similarities
//         
//     auto start = system_clock::now();
//     for (unsigned int box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
//     {
//         
//         double saliency1 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, saliencyFieldId) ) )[0];
//         
//         const mxArray* detectionClass1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassFieldId);        
//         
// //         char* buf1;
// //         size_t buflen1 = mxGetN(detectionClass1) * sizeof(mxChar) + 1;
// //         buf1 = (char*)mxMalloc(buflen1);
// //         mxGetString(detectionClass1, buf1, (mwSize)buflen1);
//         
//         size_t buflen1 = mxGetN(detectionClass1);
//         string detectionClass1String(buflen1, ' ');
//         for (unsigned stringIterator1 = 0; stringIterator1 < buflen1; stringIterator1++)
//         {
//             detectionClass1String[stringIterator1] = ((char*)mxGetData(detectionClass1))[0];
//         }
//         
// 
//         double detectionClass1JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
//         
//         const mxArray* spm1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, SPMFieldId);
//         double* spm1Features = (double*)mxGetData(spm1);
//         
//         const mxArray* largerBoxIds1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, largerBoxIdsFieldId);
//         
//                 
//         for (unsigned int box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
//         {
//             
//             double saliency2 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, saliencyFieldId) ) )[0];
//             
//             const mxArray* detectionClass2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassFieldId);
//          
// //             char* buf2;
// //             size_t buflen2 = mxGetN(detectionClass2) * sizeof(mxChar) + 1;
// //             buf2 = (char*)mxMalloc(buflen2);
// //             mxGetString(detectionClass2, buf2, (mwSize)buflen2);
//             
//             size_t buflen2 = mxGetN(detectionClass2);
//             string detectionClass2String(buflen2, ' ');
//             for (unsigned stringIterator2 = 0; stringIterator2 < buflen2; stringIterator2++)
//             {
//                 detectionClass2String[stringIterator2] = ((char*)mxGetData(detectionClass2))[0];
//             }
//                         
//             double detectionClass2JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
//             
//             const mxArray* spm2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, SPMFieldId);
//             double* spm2Features = (double*)mxGetData(spm2);
//                         
//             // add saliency scores
//             similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (saliency1 + saliency2);
//             
//             // add detection scores
// //             if( !(strcmp(buf1, "background") == 0 || strcmp(buf2, "background") == 0) )
// //             {
// //                 if ( strcmp(buf1, buf2) == 0 )
// //                 {
// //                     similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (detectionClass1JaccardIndex + detectionClass2JaccardIndex);
// //                 }
// //             }
//             if( !(detectionClass1String.compare("background") == 0 || detectionClass2String.compare("background") == 0) )
//             {
//                 if ( detectionClass1String.compare(detectionClass2String) == 0 )
//                 {
//                     similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (detectionClass1JaccardIndex + detectionClass2JaccardIndex);
//                 }
//             }
//             
//             // add similarity between SPMs
//             const mwSize* spmDimensions = mxGetDimensions(spm1);
//             unsigned featureDimension = (unsigned int)spmDimensions[1];
//             
//             double spmDistance = 0;
//             for (unsigned int fDim = 0; fDim < featureDimension; fDim++)
//             {
//                 spmDistance +=  ( (spm1Features[fDim] - spm2Features[fDim]) * (spm1Features[fDim] - spm2Features[fDim]) ) /
//                         ( spm1Features[fDim] + spm2Features[fDim] + DBL_EPSILON);                
//             }
//             spmDistance  = 0.5 * spmDistance;
//             double spmSimilarity = 1.0 - spmDistance;
//             
//             similarityMatrix.at(box1Id).at(box2Id) += spmSimilarity;
//             
//             
//             // add similarity between HOGs
//             double contrastHOG = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
//                     maxOuterContrast(hogSimilarityMatrixPointer, totalNumberOfBoxes1, totalNumberOfBoxes2, largerBoxIds1);
//             
//             similarityMatrix.at(box1Id).at(box2Id) += contrastHOG;
//             
// //             mxFree(buf2);
//         }
//         
// //         mxFree(buf1);
//         //cout << "Box1Id: " << box1Id << " vs. all done..." << endl;
//     }
//     auto end = system_clock::now();
//     auto elapsed = duration_cast<milliseconds>(end-start);
//     
//     cout << "Time elapsed: " << elapsed.count() << " milliseconds." << endl;
    
    
    
    // compute similarities (parallel implementation)
    
    
    //unsigned int box1Id;
    double saliency1;
    const mxArray* detectionClass1;
    size_t buflen1;
    string detectionClass1String;
    unsigned stringIterator1;
    double detectionClass1JaccardIndex;
    //const mxArray* spm1;
    //double* spm1Features;
    const mxArray* largerBoxIds1;
    //unsigned int box2Id;
    double saliency2;
    const mxArray* detectionClass2;
    size_t buflen2;
    string detectionClass2String;
    unsigned int stringIterator2;
    double detectionClass2JaccardIndex;
    //const mxArray* spm2;
    //double* spm2Features;
    //const mwSize* spmDimensions;
    //unsigned int featureDimension;
    //double spmDistance;
    //unsigned int fDim;
    //double spmSimilarity;
    double contrastHOG;
    double contrastSPM;
//     double skipSPM1;
//     double skipSPM2;
    
    //omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());

  
    auto start = system_clock::now();
    #pragma omp parallel for default(none) \
                shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, /*spmSimilarityMatrixPointer,*/ similarityMatrix, saliencyFieldId, /*skipSPMFieldId,*/ \
                        detectionClassFieldId, detectionClassJaccardIndexFieldId, /*SPMFieldId,*/ largerBoxIdsFieldId, prhs, cout, maxHOGMatchScores/*, maxSPMMatchScores*/) \
                        private(box1Id, saliency1, detectionClass1, buflen1, detectionClass1String, stringIterator1, detectionClass1JaccardIndex, \
                                /*spm1, spm1Features,*/ largerBoxIds1, box2Id, saliency2, detectionClass2, buflen2, detectionClass2String, stringIterator2, \
                                detectionClass2JaccardIndex, /*spm2, spm2Features, spmDimensions, featureDimension, spmDistance, fDim, spmSimilarity,*/ contrastHOG/*, contrastSPM, skipSPM1, skipSPM2*/)                          
    for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
    {
        
        saliency1 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, saliencyFieldId) ) )[0];
        
        detectionClass1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassFieldId);        

        buflen1 = mxGetN(detectionClass1);
        detectionClass1String = string(buflen1, ' ');
        for (stringIterator1 = 0; stringIterator1 < buflen1; stringIterator1++)
        {
            detectionClass1String[stringIterator1] = ((char*)mxGetData(detectionClass1))[0];
        }
        

        detectionClass1JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
        
        //spm1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, SPMFieldId);
        //spm1Features = (double*)mxGetData(spm1);
        
        largerBoxIds1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, largerBoxIdsFieldId);
        
//         skipSPM1 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, skipSPMFieldId) ) )[0];
        
                
        for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
        {
            
            saliency2 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, saliencyFieldId) ) )[0];
            
            detectionClass2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassFieldId);
            
            buflen2 = mxGetN(detectionClass2);
            detectionClass2String = string(buflen2, ' ');
            for (stringIterator2 = 0; stringIterator2 < buflen2; stringIterator2++)
            {
                detectionClass2String[stringIterator2] = ((char*)mxGetData(detectionClass2))[0];
            }
                        
            detectionClass2JaccardIndex = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, detectionClassJaccardIndexFieldId) ) )[0];
            
//             skipSPM2 = ( (double*)mxGetData( mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, skipSPMFieldId) ) )[0];
            
            //spm2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, SPMFieldId);
            //spm2Features = (double*)mxGetData(spm2);
                        
            // add saliency scores
            similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (saliency1 + saliency2);
            
            // add detection scores
            if( !(detectionClass1String.compare("background") == 0 || detectionClass2String.compare("background") == 0) )
            {
                if ( detectionClass1String.compare(detectionClass2String) == 0 )
                {
                    similarityMatrix.at(box1Id).at(box2Id) += 0.5 * (detectionClass1JaccardIndex + detectionClass2JaccardIndex);
                }
            }
            
            // add similarity between SPMs
            //spmDimensions = mxGetDimensions(spm1);
            //featureDimension = (unsigned int)spmDimensions[1];
            
//             spmDistance = 0;
//             for (fDim = 0; fDim < featureDimension; fDim++)
//             {
//                 spmDistance +=  ( (spm1Features[fDim] - spm2Features[fDim]) * (spm1Features[fDim] - spm2Features[fDim]) ) /
//                         ( spm1Features[fDim] + spm2Features[fDim] + DBL_EPSILON);                
//             }
//             spmDistance  = 0.5 * spmDistance;
//             spmSimilarity = 1.0 - spmDistance;
            
//             contrastSPM = spmSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
//                     maxOuterContrast(spmSimilarityMatrixPointer, totalNumberOfBoxes1, totalNumberOfBoxes2, largerBoxIds1, maxSPMMatchScores);
//             
//             if (skipSPM1 == 1 || skipSPM2 == 1)
//             {
//                 contrastSPM = -1.0;
//             }
                        
            //similarityMatrix.at(box1Id).at(box2Id) += contrastSPM;
                        
            // add similarity between HOGs
            contrastHOG = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
                    maxOuterContrast(hogSimilarityMatrixPointer, totalNumberOfBoxes1, totalNumberOfBoxes2, largerBoxIds1, maxHOGMatchScores);
            
            similarityMatrix.at(box1Id).at(box2Id) += contrastHOG;
            
        }
        
        //cout << "Box1Id: " << box1Id << " vs. all done..." << endl;
    }
    auto end = system_clock::now();
    auto elapsed = duration_cast<milliseconds>(end-start);
    
    cout << "Time elapsed: " << elapsed.count() << " milliseconds." << endl;
    
    
      
    mwSize resultMatrixDims[2];
    resultMatrixDims[0] = totalNumberOfBoxes1;
    resultMatrixDims[1] = totalNumberOfBoxes2;
    plhs[0] = mxCreateNumericArray(2, resultMatrixDims, mxDOUBLE_CLASS, mxREAL);
    double* resultMatrixPointer = (double*)mxGetData(plhs[0]);
    
    for (unsigned int box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
    {
        for (unsigned int box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
        {
            resultMatrixPointer[totalNumberOfBoxes1 * box2Id + box1Id] = similarityMatrix.at(box1Id).at(box2Id);
        }
    }
    
    
}
