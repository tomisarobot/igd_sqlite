
#include <UnitTest++.h>

#include <algorithm>
#include <boost/utility.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <vector>
#include <iostream>
#include <stdexcept>
using std::cout;
using std::cerr;
using std::endl;

#include <sqlite3.h>
#include <Eigen/Dense>

#include <keen/sqlite.h>
#include <keen/str.h>

#include "train.h"
#include "test_data.h"

namespace igd {
namespace {

	namespace kq = keen::sqlite;
	using keen::str;
	using keen::str_trans;

	std::string build_table(kq::Database &db, const mat &x, const vec &y)
	{
		assert(x.rows() == y.rows());

		kq::Transaction trans(db);

		size_t xr = x.rows();
		size_t xc = x.cols();
		size_t yr = y.rows();
		size_t yc = y.cols();

		std::string root = keen::next_ident("igd");

		std::string dat = str("%1%_data", root);

		std::string create_dat_sql = str(
				"CREATE TABLE %1% (%2%,%3%);"
				, dat
				, str_trans("y%1% REAL", 1u, yc + 1)
				, str_trans("x%1% REAL", 1u, xc + 1)
			);
		kq::execute_once(db, create_dat_sql);

		std::string insert_dat_sql = str(
				"INSERT INTO %1% (%2%,%3%) VALUES (%4%)"
				, dat
				, str_trans("y%1%", 1u, yc + 1)
				, str_trans("x%1%", 1u, xc + 1)
				, str_trans("?%1%", 1u, yc + xc + 1)
			);
		kq::Statement insert_dat(db, insert_dat_sql);
		for (size_t i = 0; i < xr; i++) {
			size_t b = 1;
			for (size_t j = 0; j < yc; j++) insert_dat.bind(b++, y(i, j));
			for (size_t j = 0; j < xc; j++) insert_dat.bind(b++, x(i, j));
			insert_dat.execute();
			insert_dat.reset();
			insert_dat.clear_bindings();
		}

		trans.commit();

		return dat;
	}

	class SimpleIGD : public AggregateVisitor {
	public:
		SimpleIGD(size_t m_, size_t n_, double alpha_, size_t iterations_)
				: m(m_)
				, n(n_)
				, alpha(alpha_)
				, iterations_max(iterations_)
				, iterations_cur(0) {
			theta = s = vec::Zero(n);	
		}

		virtual void step(const std::vector<double> &record) {
			const Eigen::Map<vec> xi((double *)&record[1], record.size() - 1);
			s += (alpha / (double)m) * xi * (xi.dot(theta) - record[0]);
		}
		virtual void stop() {
			theta -= s;
			iterations_cur++;
			s = vec::Zero(n);
		}
		virtual bool done() {
			return iterations_cur >= iterations_max;
		}

		size_t m;
		size_t n;
		double alpha;
		size_t iterations_max;
		size_t iterations_cur;
		vec theta;
		vec s;
	};

	class ResevoirIGD : public AggregateVisitor {
	public:
		ResevoirIGD(size_t m_, size_t k_, size_t n_, double alpha_, size_t iterations_)
				: m(m_)
				, k(k_)
				, n(n_)
				, c(0)
				, x(k_, n_)
				, y(k_)
				, alpha(alpha_)
				, iterations_max(iterations_)
				, iterations_cur(0) {
			theta = vec::Zero(n);	
		}

		virtual void step(const std::vector<double> &record) {
			size_t i = c;
			if (c >= k) {
				static boost::mt19937 rng;
				boost::uniform_int<> dist(0, c - 1);
				boost::variate_generator< boost::mt19937, boost::uniform_int<> > gen(rng, dist);
				i = gen();
			}
			if (i < k) {
				const Eigen::Map<vec> xi((double *)&record[1], record.size() - 1);
				x.row(i) = xi;
				y(i) = record[0];
			}
			c++;
		}
		virtual void stop() {
			vec s = vec::Zero(n);
			for (size_t i = 0; i < k; i++) {
				const vec &xi = x.row(i);
				s += (alpha / (double)m) * xi * (xi.dot(theta) - y(i));
			}
			theta -= s;
			iterations_cur++;
			c = 0;
		}
		virtual bool done() {
			return iterations_cur >= iterations_max;
		}

		size_t m;
		size_t k;
		size_t n;
		size_t c;
		mat x;
		vec y;
		double alpha;
		size_t iterations_max;
		size_t iterations_cur;
		vec theta;
	};

	TEST(reservoir)
	{
		return;
		kq::Database db(":memory:", SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE);
		const Data &data = GetFoodData();
		TrainArgs args;
		std::string table = build_table(db, data.x, data.y);
		size_t wrong = 0;
		for (size_t i = 0; i < 20; i++) {{
			TrainArgs train_args;
			train_args.db_ptr = db.get_ptr();
			train_args.tabname = table;
			train_args.ycol = "y1";
			train_args.xcols = { "x1" };
			ResevoirIGD vis(data.x.rows(), 100, data.x.cols()+1, 0.01, 650);
			train(train_args, &vis);
			CHECK_CLOSE(-3.630291, vis.theta(0), 0.000001);
			CHECK_CLOSE(1.166362, vis.theta(1), 0.000001);
		}}
	}

	void igd_row(const DataSet &data, double alpha, const vec &theta, vec &s) {
		const vec &xi = data.get_x();
		s += (alpha / (double)data.rows()) * xi * (xi.dot(theta) - data.get_y());
	}

	void igd_one(DataSet &data, double alpha, vec &theta) {
		size_t m = data.rows();
		size_t n = data.cols();
		vec s = vec::Zero(n);
		while (data.next())
			igd_row(data, alpha, theta, s);
		theta -= s;
	}

	vec igd_all(DataSet &data, double alpha, size_t iters) {
		size_t n = data.cols();
		vec theta = vec::Zero(n);
		for (size_t i = 0; i < iters; i++) {
			igd_one(data, alpha, theta);
			data.reset();
		}
		return theta;
	}

	TEST(cpp_only) {
		MemoryDataSet data = GetFoodDataSet();
		for (size_t i = 0; i < 20; i++) {
			vec theta = igd_all(data, 0.01, 1500);
			CHECK_CLOSE(-3.630291, theta(0), 0.000001);
			CHECK_CLOSE(1.166362, theta(1), 0.000001);
		}
	}

}} // igd anon

int main(int argc, char ** argv) {
	int retval = UnitTest::RunAllTests();
	return 0;
}

