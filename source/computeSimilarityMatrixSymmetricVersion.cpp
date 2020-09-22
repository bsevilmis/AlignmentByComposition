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


/* NOTE: compile with mex -v -largeArrayDims CDEBUGFLAGS=" " COPTIMFLAGS="-O3" CXXDEBUGFLAGS=" " LDDEBUGFLAGS=" " LDCXXDEBUGFLAGS=" " LDCXXOPTIMFLAGS="-O3" LDOPTIMFLAGS="-O3" CXXOPTIMFLAGS="-O3" CXXFLAGS="\$CXXFLAGS -std=c++11 -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" ./computeSimilarityMatrixSymmetricVersion.cpp in MATLAB */

using namespace std;
using namespace std::chrono;


double maxOuterContrast(const double* hogSimilarityMatrixPointer, const mxArray* largerBoxIds,
        vector<double>& maxMatchScores)
{
    vector<double> maxScores;
    
    const mwSize* largerBoxIdsDimensions = mxGetDimensions(largerBoxIds);
    unsigned numberOfBoxesToCheck = (unsigned int)largerBoxIdsDimensions[0];
    double* largerBoxIdsPointer = (double*)mxGetData(largerBoxIds);
    
    for (unsigned int boxIterator = 0; boxIterator < numberOfBoxesToCheck; boxIterator++)
    {        
        unsigned int boxIdToCheck = (unsigned int)(largerBoxIdsPointer[boxIterator] - 1); // MATLAB to C++ conversion
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
    unsigned int largerBoxIdsFieldId;
    

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
        else if (fieldNameString.compare("largerBoxIds") == 0)
        {
            largerBoxIdsFieldId = i;
        }
    }
    cout << "saliencyFieldId: " << saliencyFieldId << endl;
    cout << "detectionClassFieldId: " << detectionClassFieldId << endl;
    cout << "detectionClassJaccardIndexFieldId: " << detectionClassJaccardIndexFieldId << endl;
    cout << "largerBoxIdsFieldId: " << largerBoxIdsFieldId << endl;
    
    // first find maximum HOG match response for every box in image 1
    unsigned int box1Id;
    unsigned int box2Id;
    double currentScore;
    
    vector<double> maxHOGMatchScores1(totalNumberOfBoxes1, 0.0);    
            
    //omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());
        
    #pragma omp parallel for default(none) \
        shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, maxHOGMatchScores1) \
                private(box1Id, box2Id, currentScore)
    for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
    {
        for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
        {
            currentScore = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id];
            if( currentScore > maxHOGMatchScores1.at(box1Id) )
            {
                maxHOGMatchScores1.at(box1Id) = currentScore; 
            }            
        }        
    }
    
    // first find maximum HOG match response for every box in image 2
    vector<double> maxHOGMatchScores2(totalNumberOfBoxes2, 0.0);    
            
    //omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());
        
    #pragma omp parallel for default(none) \
        shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, maxHOGMatchScores2) \
                private(box1Id, box2Id, currentScore)
    for (box2Id = 0; box2Id < totalNumberOfBoxes2; box2Id++)
    {
        for (box1Id = 0; box1Id < totalNumberOfBoxes1; box1Id++)
        {
            currentScore = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id];
            if( currentScore > maxHOGMatchScores2.at(box2Id) )
            {
                maxHOGMatchScores2.at(box2Id) = currentScore; 
            }            
        }        
    }
                        
    // compute similarities (parallel implementation)            
    double saliency1;
    const mxArray* detectionClass1;
    size_t buflen1;
    string detectionClass1String;
    unsigned stringIterator1;
    double detectionClass1JaccardIndex;    
    const mxArray* largerBoxIds1;    
    double saliency2;
    const mxArray* detectionClass2;
    size_t buflen2;
    string detectionClass2String;
    unsigned int stringIterator2;
    double detectionClass2JaccardIndex;
    const mxArray* largerBoxIds2;            
    double contrastHOG;
    double contrastSPM;

    
    //omp_set_num_threads(4);
    omp_set_num_threads(omp_get_max_threads());

  
    auto start = system_clock::now();
    #pragma omp parallel for default(none) \
                shared(totalNumberOfBoxes1, totalNumberOfBoxes2, hogSimilarityMatrixPointer, similarityMatrix, saliencyFieldId, \
                        detectionClassFieldId, detectionClassJaccardIndexFieldId, largerBoxIdsFieldId, prhs, cout, maxHOGMatchScores1, maxHOGMatchScores2) \
                        private(box1Id, saliency1, detectionClass1, buflen1, detectionClass1String, stringIterator1, detectionClass1JaccardIndex, \
                                largerBoxIds1, largerBoxIds2, box2Id, saliency2, detectionClass2, buflen2, detectionClass2String, stringIterator2, \
                                detectionClass2JaccardIndex, contrastHOG)                          
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
               
        largerBoxIds1 = mxGetFieldByNumber(mxGetCell(prhs[0], box1Id), 0, largerBoxIdsFieldId);
        
                        
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
            
            largerBoxIds2 = mxGetFieldByNumber(mxGetCell(prhs[1], box2Id), 0, largerBoxIdsFieldId);

                        
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
                                    
            // add similarity between HOGs
            contrastHOG = hogSimilarityMatrixPointer[box2Id * totalNumberOfBoxes1 + box1Id] - 
                    0.5*maxOuterContrast(hogSimilarityMatrixPointer, largerBoxIds1, maxHOGMatchScores1) -
                    0.5*maxOuterContrast(hogSimilarityMatrixPointer, largerBoxIds2, maxHOGMatchScores2);
            
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
