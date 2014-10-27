#include "swod/hog_piotr_toolbox.h"
#include "swod/feature_hog_dollar.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/internal.hpp"

using namespace cv;
using namespace std;


namespace
{
    DataTypeTime SOURCE_IMAGE("image", 0);
}


PiotrHOGParams::PiotrHOGParams()
    : winSizeW(64),
      winSizeH(128),
      spatialStride(8),
      orientBins(9),
      featureSubsetType(PiotrHOG::ALL_FEATURES)
{}


PiotrHOGParams::PiotrHOGParams(const PiotrHOGParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    spatialStride = p.spatialStride;
    orientBins = p.orientBins;
    featureSubsetType = p.featureSubsetType;
    p.mask.copyTo(mask);
}


PiotrHOGParams & PiotrHOGParams::operator= (const PiotrHOGParams & p)
{
    winSizeW = p.winSizeW;
    winSizeH = p.winSizeH;
    spatialStride = p.spatialStride;
    orientBins = p.orientBins;
    featureSubsetType = p.featureSubsetType;
    p.mask.copyTo(mask);
    return *this;
}


void PiotrHOG::setParams(const PiotrHOGParams & p)
{
    params = p;
}


Size PiotrHOG::getNumOfSpatialSteps() const
{
    CV_Assert(!hog.empty());
    Size hogDetWinSize = Size(params.winSizeW / params.spatialStride - 2,
                              params.winSizeH / params.spatialStride - 2);
    CV_Assert(0 < hogDetWinSize.width && 0 < hogDetWinSize.height);
    return hog.size() - hogDetWinSize + Size(1, 1);
}


void PiotrHOG::getFeatureVector(int positionX,
                                int positionY,
                                Mat & featureVector) const
{	
    CV_Assert(!hog.empty());
    CV_Assert(positionX >= 0);
    CV_Assert(positionY >= 0);
    Size hogDetWinSize = Size(params.winSizeW / params.spatialStride - 2,
                              params.winSizeH / params.spatialStride - 2);
    CV_Assert(positionY + hogDetWinSize.height <= hog.rows);
    CV_Assert(positionX + hogDetWinSize.width <= hog.cols);

    featureVector.create(1, getFeatureVectorLength(), CV_32F);

    switch (params.featureSubsetType)
    {
    case SUBSET_10:
    {
        int totalRowLen = 4 * params.orientBins * hog.cols;
        float * fvData = (float*)(featureVector.data);
        float * hogData2 = (float*)(hog.data) +
                           positionY * totalRowLen +
                           4 * params.orientBins * positionX;
        // LET IT BE HARD CODE!
        fvData[0] = hogData2[2];
        fvData[1] = hogData2[20];
        fvData[2] = hogData2[38];
        fvData[3] = hogData2[43];
        fvData[4] = hogData2[74];
        fvData[5] = hogData2[79];
        fvData[6] = hogData2[92];
        fvData[7] = hogData2[97];
        fvData[8] = hogData2[110];
        fvData[9] = hogData2[122];
        fvData[10] = hogData2[128];
        fvData[11] = hogData2[133];
        fvData[12] = hogData2[140];
        fvData[13] = hogData2[151];
        fvData[14] = hogData2[164];
        fvData[15] = hogData2[169];
        hogData2 += totalRowLen;
        fvData[16] = hogData2[63];
        fvData[17] = hogData2[73];
        fvData[18] = hogData2[74];
        fvData[19] = hogData2[75];
        fvData[20] = hogData2[82];
        fvData[21] = hogData2[83];
        fvData[22] = hogData2[84];
        fvData[23] = hogData2[90];
        fvData[24] = hogData2[91];
        fvData[25] = hogData2[92];
        fvData[26] = hogData2[93];
        fvData[27] = hogData2[94];
        fvData[28] = hogData2[98];
        fvData[29] = hogData2[100];
        fvData[30] = hogData2[101];
        fvData[31] = hogData2[102];
        fvData[32] = hogData2[103];
        fvData[33] = hogData2[113];
        fvData[34] = hogData2[114];
        fvData[35] = hogData2[115];
        fvData[36] = hogData2[116];
        fvData[37] = hogData2[122];
        fvData[38] = hogData2[123];
        fvData[39] = hogData2[124];
        fvData[40] = hogData2[125];
        fvData[41] = hogData2[126];
        fvData[42] = hogData2[131];
        fvData[43] = hogData2[132];
        fvData[44] = hogData2[133];
        fvData[45] = hogData2[134];
        fvData[46] = hogData2[141];
        fvData[47] = hogData2[142];
        fvData[48] = hogData2[153];
        hogData2 += totalRowLen;
        fvData[49] = hogData2[74];
        fvData[50] = hogData2[75];
        fvData[51] = hogData2[79];
        fvData[52] = hogData2[83];
        fvData[53] = hogData2[90];
        fvData[54] = hogData2[92];
        fvData[55] = hogData2[93];
        fvData[56] = hogData2[97];
        fvData[57] = hogData2[98];
        fvData[58] = hogData2[101];
        fvData[59] = hogData2[102];
        fvData[60] = hogData2[115];
        fvData[61] = hogData2[116];
        fvData[62] = hogData2[124];
        fvData[63] = hogData2[127];
        fvData[64] = hogData2[132];
        fvData[65] = hogData2[133];
        fvData[66] = hogData2[134];
        fvData[67] = hogData2[142];
        fvData[68] = hogData2[153];
        hogData2 += totalRowLen;
        fvData[69] = hogData2[37];
        fvData[70] = hogData2[38];
        fvData[71] = hogData2[47];
        fvData[72] = hogData2[55];
        fvData[73] = hogData2[56];
        fvData[74] = hogData2[74];
        fvData[75] = hogData2[81];
        fvData[76] = hogData2[82];
        fvData[77] = hogData2[99];
        fvData[78] = hogData2[100];
        fvData[79] = hogData2[132];
        fvData[80] = hogData2[133];
        fvData[81] = hogData2[135];
        fvData[82] = hogData2[152];
        fvData[83] = hogData2[160];
        fvData[84] = hogData2[169];
        fvData[85] = hogData2[170];
        hogData2 += totalRowLen;
        fvData[86] = hogData2[36];
        fvData[87] = hogData2[37];
        fvData[88] = hogData2[46];
        fvData[89] = hogData2[69];
        fvData[90] = hogData2[83];
        fvData[91] = hogData2[84];
        fvData[92] = hogData2[85];
        fvData[93] = hogData2[93];
        fvData[94] = hogData2[94];
        fvData[95] = hogData2[95];
        fvData[96] = hogData2[100];
        fvData[97] = hogData2[101];
        fvData[98] = hogData2[102];
        fvData[99] = hogData2[103];
        fvData[100] = hogData2[104];
        fvData[101] = hogData2[105];
        fvData[102] = hogData2[111];
        fvData[103] = hogData2[112];
        fvData[104] = hogData2[113];
        fvData[105] = hogData2[114];
        fvData[106] = hogData2[119];
        fvData[107] = hogData2[120];
        fvData[108] = hogData2[121];
        fvData[109] = hogData2[122];
        fvData[110] = hogData2[123];
        fvData[111] = hogData2[139];
        fvData[112] = hogData2[162];
        fvData[113] = hogData2[170];
        fvData[114] = hogData2[179];
        hogData2 += totalRowLen;
        fvData[115] = hogData2[46];
        hogData2 += totalRowLen;
        fvData[116] = hogData2[36];
        fvData[117] = hogData2[37];
        fvData[118] = hogData2[45];
        fvData[119] = hogData2[46];
        fvData[120] = hogData2[162];
        fvData[121] = hogData2[170];
        fvData[122] = hogData2[179];
        hogData2 += totalRowLen;
        fvData[123] = hogData2[23];
        fvData[124] = hogData2[36];
        fvData[125] = hogData2[37];
        fvData[126] = hogData2[41];
        fvData[127] = hogData2[43];
        fvData[128] = hogData2[44];
        fvData[129] = hogData2[45];
        fvData[130] = hogData2[46];
        fvData[131] = hogData2[130];
        fvData[132] = hogData2[162];
        fvData[133] = hogData2[163];
        fvData[134] = hogData2[166];
        fvData[135] = hogData2[185];
        hogData2 += totalRowLen;
        fvData[136] = hogData2[22];
        fvData[137] = hogData2[23];
        fvData[138] = hogData2[36];
        fvData[139] = hogData2[37];
        fvData[140] = hogData2[46];
        fvData[141] = hogData2[55];
        fvData[142] = hogData2[93];
        fvData[143] = hogData2[102];
        fvData[144] = hogData2[105];
        fvData[145] = hogData2[114];
        fvData[146] = hogData2[123];
        fvData[147] = hogData2[143];
        fvData[148] = hogData2[162];
        fvData[149] = hogData2[185];
        fvData[150] = hogData2[214];
        hogData2 += totalRowLen;
        fvData[151] = hogData2[0];
        fvData[152] = hogData2[9];
        fvData[153] = hogData2[16];
        fvData[154] = hogData2[22];
        fvData[155] = hogData2[23];
        fvData[156] = hogData2[27];
        fvData[157] = hogData2[34];
        fvData[158] = hogData2[37];
        fvData[159] = hogData2[46];
        fvData[160] = hogData2[55];
        fvData[161] = hogData2[73];
        fvData[162] = hogData2[82];
        fvData[163] = hogData2[83];
        fvData[164] = hogData2[100];
        fvData[165] = hogData2[116];
        fvData[166] = hogData2[133];
        fvData[167] = hogData2[143];
        fvData[168] = hogData2[152];
        fvData[169] = hogData2[155];
        fvData[170] = hogData2[170];
        fvData[171] = hogData2[184];
        fvData[172] = hogData2[185];
        fvData[173] = hogData2[191];
        fvData[174] = hogData2[198];
        fvData[175] = hogData2[200];
        fvData[176] = hogData2[207];
        fvData[177] = hogData2[208];
        fvData[178] = hogData2[209];
        hogData2 += totalRowLen;
        fvData[179] = hogData2[9];
        fvData[180] = hogData2[16];
        fvData[181] = hogData2[17];
        fvData[182] = hogData2[34];
        fvData[183] = hogData2[38];
        fvData[184] = hogData2[42];
        fvData[185] = hogData2[47];
        fvData[186] = hogData2[51];
        fvData[187] = hogData2[52];
        fvData[188] = hogData2[55];
        fvData[189] = hogData2[56];
        fvData[190] = hogData2[69];
        fvData[191] = hogData2[70];
        fvData[192] = hogData2[96];
        fvData[193] = hogData2[105];
        fvData[194] = hogData2[111];
        fvData[195] = hogData2[120];
        fvData[196] = hogData2[123];
        fvData[197] = hogData2[133];
        fvData[198] = hogData2[138];
        fvData[199] = hogData2[151];
        fvData[200] = hogData2[152];
        fvData[201] = hogData2[155];
        fvData[202] = hogData2[156];
        fvData[203] = hogData2[165];
        fvData[204] = hogData2[169];
        fvData[205] = hogData2[173];
        fvData[206] = hogData2[174];
        fvData[207] = hogData2[178];
        fvData[208] = hogData2[185];
        fvData[209] = hogData2[191];
        fvData[210] = hogData2[200];
        fvData[211] = hogData2[207];
        fvData[212] = hogData2[208];
        fvData[213] = hogData2[209];
        hogData2 += totalRowLen;
        fvData[214] = hogData2[37];
        fvData[215] = hogData2[38];
        fvData[216] = hogData2[46];
        fvData[217] = hogData2[55];
        fvData[218] = hogData2[56];
        fvData[219] = hogData2[64];
        fvData[220] = hogData2[72];
        fvData[221] = hogData2[73];
        fvData[222] = hogData2[81];
        fvData[223] = hogData2[87];
        fvData[224] = hogData2[91];
        fvData[225] = hogData2[104];
        fvData[226] = hogData2[105];
        fvData[227] = hogData2[108];
        fvData[228] = hogData2[116];
        fvData[229] = hogData2[117];
        fvData[230] = hogData2[120];
        fvData[231] = hogData2[121];
        fvData[232] = hogData2[122];
        fvData[233] = hogData2[126];
        fvData[234] = hogData2[134];
        fvData[235] = hogData2[135];
        fvData[236] = hogData2[139];
        fvData[237] = hogData2[140];
        fvData[238] = hogData2[151];
        fvData[239] = hogData2[152];
        fvData[240] = hogData2[161];
        fvData[241] = hogData2[169];
        fvData[242] = hogData2[170];
        fvData[243] = hogData2[179];
        hogData2 += totalRowLen;
        fvData[244] = hogData2[9];
        fvData[245] = hogData2[22];
        fvData[246] = hogData2[23];
        fvData[247] = hogData2[36];
        fvData[248] = hogData2[44];
        fvData[249] = hogData2[54];
        fvData[250] = hogData2[55];
        fvData[251] = hogData2[59];
        fvData[252] = hogData2[61];
        fvData[253] = hogData2[62];
        fvData[254] = hogData2[72];
        fvData[255] = hogData2[73];
        fvData[256] = hogData2[80];
        fvData[257] = hogData2[81];
        fvData[258] = hogData2[82];
        fvData[259] = hogData2[89];
        fvData[260] = hogData2[90];
        fvData[261] = hogData2[91];
        fvData[262] = hogData2[98];
        fvData[263] = hogData2[99];
        fvData[264] = hogData2[100];
        fvData[265] = hogData2[108];
        fvData[266] = hogData2[109];
        fvData[267] = hogData2[116];
        fvData[268] = hogData2[117];
        fvData[269] = hogData2[118];
        fvData[270] = hogData2[125];
        fvData[271] = hogData2[126];
        fvData[272] = hogData2[127];
        fvData[273] = hogData2[134];
        fvData[274] = hogData2[135];
        fvData[275] = hogData2[136];
        fvData[276] = hogData2[143];
        fvData[277] = hogData2[144];
        fvData[278] = hogData2[145];
        fvData[279] = hogData2[146];
        fvData[280] = hogData2[152];
        fvData[281] = hogData2[155];
        fvData[282] = hogData2[162];
        fvData[283] = hogData2[163];
        fvData[284] = hogData2[170];
        fvData[285] = hogData2[184];
        fvData[286] = hogData2[185];
        fvData[287] = hogData2[207];
        hogData2 += totalRowLen;
        fvData[288] = hogData2[0];
        fvData[289] = hogData2[4];
        fvData[290] = hogData2[5];
        fvData[291] = hogData2[9];
        fvData[292] = hogData2[10];
        fvData[293] = hogData2[17];
        fvData[294] = hogData2[18];
        fvData[295] = hogData2[22];
        fvData[296] = hogData2[23];
        fvData[297] = hogData2[27];
        fvData[298] = hogData2[28];
        fvData[299] = hogData2[36];
        fvData[300] = hogData2[37];
        fvData[301] = hogData2[40];
        fvData[302] = hogData2[41];
        fvData[303] = hogData2[45];
        fvData[304] = hogData2[46];
        fvData[305] = hogData2[53];
        fvData[306] = hogData2[54];
        fvData[307] = hogData2[58];
        fvData[308] = hogData2[59];
        fvData[309] = hogData2[63];
        fvData[310] = hogData2[64];
        fvData[311] = hogData2[71];
        fvData[312] = hogData2[72];
        fvData[313] = hogData2[76];
        fvData[314] = hogData2[77];
        fvData[315] = hogData2[81];
        fvData[316] = hogData2[82];
        fvData[317] = hogData2[89];
        fvData[318] = hogData2[94];
        fvData[319] = hogData2[95];
        fvData[320] = hogData2[99];
        fvData[321] = hogData2[100];
        fvData[322] = hogData2[107];
        fvData[323] = hogData2[112];
        fvData[324] = hogData2[113];
        fvData[325] = hogData2[117];
        fvData[326] = hogData2[119];
        fvData[327] = hogData2[125];
        fvData[328] = hogData2[130];
        fvData[329] = hogData2[131];
        fvData[330] = hogData2[135];
        fvData[331] = hogData2[136];
        fvData[332] = hogData2[143];
        fvData[333] = hogData2[144];
        fvData[334] = hogData2[148];
        fvData[335] = hogData2[149];
        fvData[336] = hogData2[153];
        fvData[337] = hogData2[162];
        fvData[338] = hogData2[166];
        fvData[339] = hogData2[167];
        fvData[340] = hogData2[171];
        fvData[341] = hogData2[172];
        fvData[342] = hogData2[179];
        fvData[343] = hogData2[180];
        fvData[344] = hogData2[184];
        fvData[345] = hogData2[185];
        fvData[346] = hogData2[189];
        fvData[347] = hogData2[193];
        fvData[348] = hogData2[198];
        fvData[349] = hogData2[202];
        fvData[350] = hogData2[203];
        fvData[351] = hogData2[207];
        fvData[352] = hogData2[208];
        fvData[353] = hogData2[215];
        break;
    }
    case (ALL_FEATURES):
    {
        const int blockDescriptionLength = 4 * params.orientBins;
        int pos = 0;
        int rowLength = blockDescriptionLength * hogDetWinSize.width * sizeof(float);
        for (int i = positionY; i < positionY + hogDetWinSize.height; ++i)
        {
            memcpy(featureVector.data + pos,
                   &(hog.at<float>(i, positionX * blockDescriptionLength)),
                   rowLength);
            pos += rowLength;
        }
        break;
    }
    case (SUBSET_BY_MASK):
    {
        int rowLength = 4 * params.orientBins * hogDetWinSize.width;
        int totalRowLen = 4 * params.orientBins * hog.cols;
        float * fvData = (float*)(featureVector.data);
        uchar * maskData = (uchar*)(params.mask.data);
        float * hogData = (float*)(hog.data) + (4 * params.orientBins * positionX);

        for (int i = positionY; i < positionY + hogDetWinSize.height; ++i)
        {
            float * hogData2 = hogData + i * totalRowLen;
            for (int j = 0; j < rowLength; ++j)
            {
                int maskValue = (*maskData);
                (*fvData) = (*hogData2) * maskValue;
                fvData += maskValue;
                hogData2++;
                maskData++;
            }
        }
        break;
    }
    default:
    {
        CV_Error(CV_StsBadArg, "Unknown feature mask type.");
        break;
    }
    } // switch (featureSubsetType)
}


int PiotrHOG::getFeatureVectorLength() const
{
    int n = 0;
    switch (params.featureSubsetType)
    {
    case SUBSET_10:
        n = 354;
        break;
    case ALL_FEATURES:
        n = piotrhog::getHogDescriptorSize(params.winSizeW,
                                           params.winSizeH,
                                           params.spatialStride,
                                           params.orientBins);
        break;
    case SUBSET_BY_MASK:
        n = static_cast<int>(sum(params.mask)[0]);
        break;
    default:
        n = 0;
        break;
    }
    return n;
}


void PiotrHOG::computeOnNewImage(const SourcesMap & sources)
{
    CV_Assert(sources.count(SOURCE_IMAGE));
    sources.at(SOURCE_IMAGE).copyTo(img);
}


void PiotrHOG::computeOnNewScale(const float scale)
{
    CV_Assert(!img.empty());
    CV_Assert(0.0f < scale);

    Mat scaledImg;
    Size scaledImageSize((int)(img.cols / scale), (int)(img.rows / scale));
    int method = (scale <= 1.0f) ? INTER_CUBIC : INTER_AREA;
    resize(img, scaledImg, scaledImageSize, 0.0, 0.0, method);
    hog = piotrhog::hog(scaledImg, params.spatialStride, params.orientBins);
}


void PiotrHOG::getROIDescription(Mat & featureDescription,
                                 const SourcesMap & sources,
                                 const Rect & roi)
{
    CV_Assert(sources.count(SOURCE_IMAGE));
    CV_Assert(0 < roi.width);
    CV_Assert(0 < roi.height);
    const Mat & im = sources.at(SOURCE_IMAGE);
    CV_Assert(!im.empty());

    featureDescription.create(1, getFeatureVectorLength(), CV_32F);
    Mat imageROI;
    getROI(im, imageROI, roi);
    resize(imageROI, imageROI, Size(params.winSizeW, params.winSizeH));

    SourcesMap imageSource;
    imageSource[SOURCE_IMAGE] = imageROI;
    computeOnNewImage(imageSource);
    computeOnNewScale(1.0f);
    getFeatureVector(0, 0, featureDescription);
}


void PiotrHOG::getROIDescription(Mat & featureDescription,
                                 const SourcesMap & sources,
                                 const vector<Rect> & roi)
{
    featureDescription.create(roi.size(), getFeatureVectorLength(), CV_32F);
    for (size_t i = 0; i < roi.size(); ++i)
    {
        Mat featureVector = featureDescription.row(i);
        getROIDescription(featureVector, sources, roi[i]);
    }
}


vector<vector<DataTypeTime> > PiotrHOG::getRequiredSources() const
{
    vector<vector<DataTypeTime> > sources(1);
    sources[0].push_back(SOURCE_IMAGE);
    return sources;
}
