
#include <keen/str.h>
#include <keen/sqlite.h>

#include "train.h"

namespace igd {
namespace {

	namespace kq = keen::sqlite;

	using keen::str;
	using keen::str_trans;
	using keen::next_ident;

	template<typename T>
		void agg_step(sqlite3_context *context, int argc, sqlite3_value **argv) {
			std::vector<double> values(argc, 0.0);
			for (int i = 0; i < argc; i++) {
				if (!kq::value_check<double, kq::is_numeric>(argv[i], values[i]))
					throw std::runtime_error("non-numeric number found");
			}

			T *vis = (T *)sqlite3_user_data(context);
			if (!vis)
				throw std::runtime_error("bad aggregate visitor");
			vis->step(values);
		}

	template<typename T>
		void agg_stop(sqlite3_context *context) {
			T *vis = (T *)sqlite3_user_data(context);
			if (!vis)
				throw std::runtime_error("bad aggregate visitor");
			vis->stop();
		}

	void train_populate(const TrainArgs &args) {

		kq::Transaction trans(args.db_ptr);

		std::string create_sql = str(
				"CREATE TEMPORARY TABLE %1% ("
					"%2% REAL,"
					"%3% REAL,"
					"%4%"
				");"
				, args.calc_tab
				, args.ycol
				, args.icol
				, str_trans("%1% REAL", args.xcols.begin(), args.xcols.end())
			);
		kq::execute_once(args.db_ptr, create_sql);

		std::string insert_sql = str(
				"INSERT INTO %1% "
					"(%2%,%3%,%4%) "
				"SELECT %5%,1.0,%6% "
				"FROM %7% "
				"ORDER BY RANDOM();"
				, args.calc_tab
				, args.ycol
				, args.icol
				, str_trans("%1%", args.xcols.begin(), args.xcols.end())
				, args.ycol
				, str_trans("%1%", args.xcols.begin(), args.xcols.end())
				, args.tabname
			);
		kq::execute_once(args.db_ptr, insert_sql);

		trans.commit();
	}

	void train_aggregate(const TrainArgs &args, AggregateVisitor *vis) {

		// create
		sqlite3_create_function_v2(args.db_ptr, args.calc_agg.c_str(), -1, SQLITE_ANY, vis, 
				NULL, agg_step<AggregateVisitor>, agg_stop<AggregateVisitor>, NULL);

		static const char *sql_template = 
			"SELECT %1%(%2%,%3%,%4%) FROM %5%;";
		std::string sql = str(sql_template
				, args.calc_agg
				, args.ycol
				, args.icol
				, str_trans("%1%", args.xcols.begin(), args.xcols.end())
				, args.calc_tab
			);
		kq::Statement statement(args.db_ptr, sql);

		do {
			statement.execute();
			statement.reset();
		} while (!vis->done());

		// delete
		sqlite3_create_function_v2(args.db_ptr, args.calc_agg.c_str(), -1, SQLITE_ANY, vis, 
				NULL, NULL, NULL, NULL);
	}

	void train_clean(const TrainArgs &args) {
		std::string create_sql = str(
				"DROP TABLE %1%;"
				, args.calc_tab
			);
		kq::execute_once(args.db_ptr, create_sql);
	}

}} // igd anon

namespace igd {

	void train(const TrainArgs &args, AggregateVisitor *vis) {
		train_populate(args);
		train_aggregate(args, vis);
		train_clean(args);
	}

} // igd

