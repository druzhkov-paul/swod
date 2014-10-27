#include "swod/swod.hpp"
#include <iostream>

using namespace std;
using namespace cv;


namespace
{
    void listComponent(const string & prefix, bool createConfigFiles = false)
    {
        vector<string> algorithmsList;
        Algorithm::getList(algorithmsList);
        for (size_t i = 0; i < algorithmsList.size(); ++i)
        {
            if (algorithmsList[i].find(prefix) == 0)
            {
                cout << "  - " << algorithmsList[i] << endl;
                if (createConfigFiles)
                {
                    Ptr<Algorithm> a = Algorithm::create<Algorithm>(algorithmsList[i]);
                    FileStorage fs(algorithmsList[i] + ".yml", FileStorage::WRITE);
                    a->write(fs);
                    fs.release();
                }
            }
        }
    }


    void listAvailableComponents(bool createConfigFiles = false)
    {
        cout << "Available features:" << endl;
        listComponent("SWOD.Feature.", createConfigFiles);

        cout << "Available classifiers:" << endl;
        listComponent("SWOD.Classifier.", createConfigFiles);

        cout << "Available nonmaximum suppressors:" << endl;
        listComponent("SWOD.NMS.", createConfigFiles);

        cout << "Available data providers: " << endl;
        listComponent("SWOD.DataProvider.", createConfigFiles);

        cout << endl;
    }
}


int main(int argc, char ** argv)
{
    bool createConfigFiles = (argc == 2 && strcmp(argv[1], "--configs") == 0);

    initFeatures();
    initClassifiers();
    initDataProviders();
    initNonMaximumSuppressors();

    listAvailableComponents(createConfigFiles);

    return 0;
}
