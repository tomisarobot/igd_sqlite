#ifndef DATA_H
#define DATA_H

#include <Eigen/Dense>

typedef Eigen::MatrixXd mat;
typedef Eigen::VectorXd vec;

// awkward

struct Data {
	Data(size_t r, size_t c)
		: x(r, c)
		, y(r)
	{}
	mat x;
	vec y;
};

const Data &GetHomeData();
const Data &GetFoodData();

class DataSet {
public:
	virtual bool next() = 0;
	virtual vec get_x() const = 0;
	virtual double get_y() const = 0;
	virtual size_t rows() const = 0;
	virtual size_t cols() const = 0;
	virtual void reset() = 0;
};

class MemoryDataSet : public DataSet {
public:
	MemoryDataSet(const mat &x_, const vec &y_) 
		: x(x_), y(y_), index(0), first(true) {}
	virtual bool next() {
		if (index+1 < rows()) {
			if (!first)
				index++;
			first = false;
			return true;
		} else {
			return false;
		}
	}
	virtual vec get_x() const {return x.row(index);}
	virtual double get_y() const {return y(index);}
	virtual size_t rows() const {return x.rows();}
	virtual size_t cols() const {return x.cols();}
	virtual void reset() {index = 0; first = true;}
private:
	const mat x;
	const vec y;
	size_t index;
	bool first;
};

MemoryDataSet GetFoodDataSet();

#endif
