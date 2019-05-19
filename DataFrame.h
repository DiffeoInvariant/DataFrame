/**
 @author: Zane Jakobs
 @file: DataFrame.h: Pandas-like dataframes for C++
 build on the Eigen library and C++17 stdlib algorithms
 and data structures
 */
#include <string>
#include <vector>
#include <optional>
#include <utility>
#include <typeinfo>
#include <memory>
#include <type_traits>
#include <functional>
#include <Eigen/Core>
#include <ostream>
#include <fstream>
#include <sstream>

using namespace std;


namespace DataFrame
{
    
//Python range functionality
    struct range
    {
        vector<int> operator() (int start=0, int end=1)
        {
            if(end <= start){
                throw "Error: end must be greater than start";
            }
            vector<int> list;
            for(int i = start; i < end; i++){
                list.push_back(i);
            }
            return list;
        }
    };
    
    //check if type is a range
    template<typename T>
    bool is_range(vector<T>& vec)
    {
        if(!is_integral<T>::value)
        {
            return false;
        }
        else
        {
            for(auto &it = vec.begin(); it++ != vec.end(); it++){
                //check if each element is one more than the last
                if( *(it++) != (*it) + 1){
                    return false;
                }
            }
            //if all elements are one more than previous, it's a range
            return true;
        }
    }
    
using svec = vector<string>;
/**
 @author: Zane Jakobs
 @brief: class to implement Pandas-like dataframes in C++
 */
template<typename T>
class DataFrame
{
protected:
    
    svec column_names;
    
    svec row_names;
    //numeric data
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> nData;
public:
    
    DataFrame() {};
    //ctor for Eigen matrix data and optional names
    DataFrame(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& dat,
              optional<svec>          colNames,
              optional<svec>         rowNames)
    {
        static_assert(is_arithmetic<T>::value);
        
        if( colNames )
        {
            column_names = *colNames;
        }
        else
        {
            auto ncol = dat.cols();
            svec cls;
            for(int i = 0; i < ncol; i++)
            {
                string name(to_string(i));
                cls.push_back(name);
            }
            column_names = cls;
        }//end else
        if( rowNames )
        {
            row_names = *rowNames;
        }
        else
        {
            auto nrow = dat.rows();
            svec rws;
            for(int i = 0; i < nrow; i++)
            {
                string name(to_string(i));
                rws.push_back(name);
            }
            row_names = rws;
        }//end else

        nData = dat;
    }
    
    auto data() const noexcept
    {
        return nData;
    }
    
    void setData(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> newData)
    {
        nData = newData;
    }
    //if arg passed, set colum_names, else get column names
    optional<svec> colnames(unique_ptr<svec> newNames = nullptr) noexcept
    {
        if(newNames)
        {
            column_names = *newNames;
            return {};
        }
        else
        {
            return column_names;
        }
            
    }
    
    //same as colnames, but for rows
    optional<svec> rownames(unique_ptr<svec> newNames = nullptr) noexcept
    {
        if(newNames)
        {
            row_names = *(newNames);
            return {};
        }
        else
        {
            return row_names;
        }
    }
    
    //returns a pair of number of rows, number of columns
    pair<size_t, size_t> shape() const
    {
        return make_pair(nData.rows(), nData.cols());
    }
    
    //returns DataFrame of data within specified row and column range
    auto slice(size_t initRow, size_t finRow,
               size_t initCol, size_t finCol) const
    {
        if(finRow >= nData.rows())
        {
            throw "Error: finRow greater than number of rows.";
        }
        if(finCol >= nData.cols())
        {
            throw "Error: finCol greater than number of columns.";
        }
        if(initRow >= finRow)
        {
            throw "Error: initRow greater than or equal to finRow";
        }
        if(initCol >= finCol)
        {
            throw "Error: initCol greater than or equal to finCol";
        }
        //pre-initialize subset matrix
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> subset = Eigen::MatrixXd::Zero(finRow - initRow, finCol - initCol);
        
        svec rNames, cNames;
        
        for(auto row = initRow; row <= finRow; row++){
            rNames.push_back(row_names[row]);
            for(auto col= initCol; col <= finCol; col++){
                
                if(row == initRow)
                {
                    cNames.push_back(column_names[col]);
                }
                auto i = row - initRow;
                auto j = col - initCol;
                                     
                subset(i,j) = nData(row, col);
            }//end inner for
        }//end outer for
        
        return DataFrame(subset, cNames, rNames);
    }
    
    //bracket operator, takes either ranges or
    //pairs of first, last
    template<typename rowType, typename colType>
    auto operator[] (pair<rowType, colType>& indices) const
    {
        auto rows = indices.first;
        auto cols = indices.second;
        if(is_range(rows))
        {
            if( rows.front() < 0 && rows.back() > 0)
                throw "Error: cannot subset with both positive and negative row indices";
            if(is_range(cols))
            {
                //if both rows and cols are ranges
                if( cols.front() < 0 && cols.back() > 0)
                    throw "Error: cannot subset with both positive and negative column indices";
                    
                size_t initRow = rows.front();
                size_t finRow = rows.back();
                
                size_t initCol = cols.front();
                size_t finCol = cols.back();
                
                return slice(initRow, finRow, initCol, finCol);
            }//end if cols is range
            else
            {
                //not implemented yet
                throw "Error: Currently, the [] operator must take two range types.";
            }
        }//end if rows is range
        else
        {
            //not implemented yet
            throw "Error: Currently, the [] operator must take two range types.";
        }//end else
    }
    
    
    friend ostream& operator<<(ostream &os, DataFrame& df)
    {
        os << "  |_ ";
        auto dfshape = df.shape();
        size_t cls = dfshape.second;
        
        auto columns = (df.colnames());
        auto rows = (df.rownames());
        
        auto dataMat = df.data();
        
        for(auto &cit : *columns)
        {
            os << "|_" << cit.data() << "_| ";
        }
        os << "\n";
        size_t cRow = 0;
        for(auto &rit : *rows)
        {
            os << rit.data() << "|  ";
            for(size_t i = 0; i < cls; i++)
            {
                os << dataMat(cRow, i) << "  |  ";
            }
            os << "\n";
            cRow++;
        }
        return os;
    }
    
};
    /**
     T is the type of the DataFrame's contents
     */
    template<typename T = double>
    auto read_csv(string& filename)
    {
        fstream                                             fileIn;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>    dataMatrix = Eigen::MatrixXd::Zero(100,1);
        size_t                                              colSize;//num cols
        size_t                                              rowSize = 100;//num rows
        svec                                                cols, row;
        string                                              line, elem;
        size_t                                              i=0, j=0;
        bool                                                first = true;//is first iteration?
        
        fileIn.open(filename, ios::in);
        
        while(!fileIn.eof()){
            
            row.clear();
            
            getline(fileIn, line);
            stringstream ss(line);
            //handle line
            while(getline(ss, elem, ',')){
                row.push_back(elem);
            }
            //if it's the first pass, allocate matrix size, store columns
            if(first)
            {
                colSize = row.size();
                dataMatrix.conservativeResize(rowSize, colSize);
                //save column names
                cols = row;
                first = false;
            }
            //if we hit the row size limit, allocate more memory
            if(i == rowSize - 1)
            {
                rowSize *= 1.5;
                dataMatrix.conservativeResize(rowSize, colSize);
            }
            if(!first)
            {
                for(j=0; j < colSize; j++){
                    dataMatrix(i,j) = stod(row[j]);
                }
                i++;
            }
        }
        
        return DataFrame(dataMatrix, cols);
    }
    
    
    
    
    
}//end namespace cpPandas
