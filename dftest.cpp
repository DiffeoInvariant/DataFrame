#include <string>
#include <vector>
#include <iostream>
#include <functional>
#include <Eigen/Core>
#include "DataFrame.h"
using namespace std;

int main()
{
    Eigen::MatrixXd data(4,2);
    data << 1.2,2.1,
            1.1,4.2,
            7.4,3.8,
            9.1,4.2;
    string sv("col1");
    string sv2("another");
    string sv3("r1");
    string sv4("cats");
    vector<string> cls, rws;
    cls.push_back(sv);
    cls.push_back(sv2);
    
    rws.push_back(sv3);
    rws.push_back(sv4);
    rws.push_back(sv3);
    rws.push_back(sv4);
    
    cout << "Testing DataFrame ctor with matrix and column names only.\n";
    auto df = DataFrame::DataFrame(data, cls,rws);
    cout << "Printing df:\n";
    cout << df << "\n";
    cout << "Shape: (" << df.shape().first <<"," <<df.shape().second << ")\n";
    
    return 0;
}
